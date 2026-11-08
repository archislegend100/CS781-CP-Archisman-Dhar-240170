"""Phase 6: abstractive legal summarization with the Phase-1 legal LMs.

Long judgments are summarised chunk by chunk (Shukla et al., AACL 2022, IN-Abs): the document is cut into sentence-
aligned chunks of `chunk` tokens; every reference-summary sentence is assigned to the chunk it overlaps most (content-
word overlap), giving chunk -> chunk-summary training pairs (a chunk with no aligned sentence learns to emit only EOS).
Decoders (LegalGPT, LegalHybrid, LongLegal) learn  chunk + " SUMMARY :" -> summary + EOS  (loss on the summary only);
LegalSeq learns encoder(chunk) -> decoder(summary). At test time the chunk summaries are generated with KV caching
(greedy + no-repeat-n-gram + repetition penalty), concatenated in document order, de-duplicated across chunks (trigram
blocking, Paulus et al. 2018; Liu & Lapata 2019) and cut to a document-level word budget tuned on a held-out split.

python sm.py phase6/configs/<cfg>.yaml download | prep | train | predict | mbr
"""
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import ckptpath
from config import args, loadcfg, path, plain, subdir, wrap
from data import writejson
from generate import Picker, autocast
from models import build as mbuild
from peft import (LoRA, LoRMA, SVDLinear, AdaLoRAController, addadapters, count, layers, qlora, wraplinears)
from tokenizer import gettok

STOP = set("the a an of to in and or for on by with is was are were be been as at that this which it its from has have had "
           "not no shall may said such any all their there these those than then into upon under also he she they his her "
           "him them we our you your i".split())
MARK = " SUMMARY :"


# ----------------------------------------------------------------------------- data

def readjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def loadsplit(p):
    d = readjson(p)
    recs = d if isinstance(d, list) else [{"id": k, **(v if isinstance(v, dict) else {"document": v})} for k, v in d.items()]
    out = []
    for i, r in enumerate(recs):
        doc = r.get("document", r.get("text", r.get("judgment", "")))
        summ = r.get("summary", r.get("reference", None))
        out.append((str(r.get("id", i)), doc if isinstance(doc, str) else " ".join(doc),
                    None if summ is None else (summ if isinstance(summ, str) else " ".join(summ))))
    return out


def files(cfg):
    c = cfg.sm
    return {k: path(c.root) / v for k, v in c.files.items()}


def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    c = cfg.sm
    have = list_repo_files(c.repo, repo_type="dataset")
    print("remote Phase-6 files:", *[f for f in have if f.startswith("Phase_6")], sep="\n  ")
    for k, f in c.files.items():
        if f not in have:
            hits = [x for x in have if x.startswith("Phase_6") and Path(x).name == Path(f).name]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix sm.files.{k}")
            f = hits[0]
        hf_hub_download(c.repo, f, repo_type="dataset", local_dir=str(path(c.root)))
        print(f"ok {f}")


SENT = re.compile(r"(?<=[.?!;])\s+(?=[A-Z0-9(\"'])")


def sentences(text):
    return [s.strip() for s in SENT.split(" ".join(text.split())) if s.strip()]


def content(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def chunkdoc(tok, text, C):
    """Sentence-aligned chunks of at most C tokens (a longer sentence is split on its own): [(sentences, ids)]."""
    out, cur, ids = [], [], []
    for s in sentences(text):
        t = tok.encode(s)
        while len(t) > C:                       # very long "sentence" (tables, lists): hard split
            if cur:
                out.append((cur, ids))
                cur, ids = [], []
            out.append(([tok.decode(t[:C])], t[:C]))
            t = t[C:]
        if ids and len(ids) + len(t) > C:
            out.append((cur, ids))
            cur, ids = [], []
        cur.append(s)
        ids = ids + t
    if ids:
        out.append((cur, ids))
    return out or [([""], tok.encode(".") or [3])]


def align(chunks, summary):
    """Assign every summary sentence to the chunk with the largest content-word overlap -> per-chunk target text."""
    cw = [content(" ".join(s)) for s, _ in chunks]
    tgt = [[] for _ in chunks]
    for s in sentences(summary):
        w = content(s)
        if not w:
            continue
        sc = [len(w & c) / len(w) for c in cw]
        tgt[int(np.argmax(sc))].append(s)
    return [" ".join(t) for t in tgt]


_TOK, _C = None, None


def _prep_one(rec):
    i, doc, summ = rec
    ch = chunkdoc(_TOK, doc, _C)
    tg = align(ch, summ) if summ is not None else [None] * len(ch)
    return {"id": i, "chunks": [ids for _, ids in ch],
            "targets": [None if t is None else _TOK.encode(t) for t in tg],
            "doc_words": len(doc.split()), "summary": summ}


def prep(cfg):
    """Chunk + align every document once for a chunk size -> runs/sm/data_c<C>.pt; held-out split; statistics."""
    import multiprocessing as mp
    global _TOK, _C
    c = cfg.sm
    _TOK, _C = gettok(loadcfg(c.backbone.config)), int(c.chunk)
    out = path(c.datadir) / f"data_c{_C}.pt"
    if out.exists():
        print(f"exists: {out}")
        return
    f = files(cfg)
    tr, ts = loadsplit(f["train"]), loadsplit(f["test"])
    rng = random.Random(int(c.get("valseed", 781)))
    ids = sorted(r[0] for r in tr)
    rng.shuffle(ids)
    val = set(ids[:int(c.get("valdocs", 100))])
    t0 = time.time()
    with mp.get_context("fork").Pool(int(c.get("workers", 32))) as pool:
        P = pool.map(_prep_one, tr + ts, chunksize=8)
    data = {"train": [], "val": [], "test": []}
    for r, p in zip(tr + ts, P):
        data["test" if r[2] is None else ("val" if r[0] in val else "train")].append(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    dw = np.array([p["doc_words"] for p in data["train"]])
    sw = np.array([len(p["summary"].split()) for p in data["train"]])
    nc = np.array([len(p["chunks"]) for p in data["train"]])
    tl = np.array([len(t) for p in data["train"] for t in p["targets"]])
    print(f"chunk {_C}: train {len(data['train'])} val {len(data['val'])} test {len(data['test'])} docs | doc words median "
          f"{np.median(dw):.0f} p90 {np.percentile(dw, 90):.0f} | summary words median {np.median(sw):.0f} p90 "
          f"{np.percentile(sw, 90):.0f} | words ratio median {np.median(sw / np.maximum(dw, 1)):.3f} | chunks/doc median "
          f"{np.median(nc):.0f} max {nc.max()} | chunk-target tokens median {np.median(tl):.0f} p95 {np.percentile(tl, 95):.0f} "
          f"max {tl.max()} | empty chunk targets {np.mean(tl == 0):.1%} | {time.time() - t0:.0f}s")


def loaddata(cfg):
    return torch.load(path(cfg.sm.datadir) / f"data_c{int(cfg.sm.chunk)}.pt", map_location="cpu", weights_only=False)


# ----------------------------------------------------------------------------- model

def make(cfg, dev):
    """Phase-1 generator (decoder or full encoder-decoder, causal masks kept) + the adaptation method."""
    c = cfg.sm
    b = c.backbone
    obj = torch.load(ckptpath(path(b.run), b.ckpt), map_location="cpu", weights_only=False)
    mcfg = wrap(obj["cfg"]["model"])
    m = mbuild(mcfg)
    m.load_state_dict(obj["model"])
    del obj
    family = getattr(m, "family", "causal")
    if c.get("layers") and family != "seq2seq":
        m.blocks = torch.nn.ModuleList(list(m.blocks)[:int(c.layers)])
    meth = c.method
    for p in m.parameters():
        p.requires_grad = meth in ("full", "kd")
    T = tuple(c.targets)
    if meth == "lora":
        wraplinears(m, LoRA, T, r=int(c.rank), alpha=float(c.alpha), drop=float(c.get("loradrop", 0.0)))
    elif meth == "lorma":
        wraplinears(m, LoRMA, T, r=int(c.rank), alpha=float(c.get("alpha", c.rank)))
    elif meth == "qlora":
        qlora(m, T, r=int(c.rank), alpha=float(c.alpha))
    elif meth == "adalora":
        wraplinears(m, SVDLinear, T, r=int(c.get("rank0", 12)), alpha=float(c.alpha))
    elif meth == "houlsby":
        addadapters(m, int(mcfg.dim), int(c.bottleneck))
    elif meth == "pfeiffer":
        addadapters(m, int(mcfg.dim), int(c.bottleneck), names=("ffn",))
    elif meth not in ("full", "kd"):
        raise ValueError(meth)
    m.ckpt = bool(c.get("ckpt", False))
    return m.to(dev), mcfg, family


def loadsm(run, dev):
    obj = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = wrap(obj["cfg"])
    m, mcfg, fam = make(cfg, dev)
    m.load_state_dict(obj["model"])
    return m.eval(), obj, fam


# ----------------------------------------------------------------------------- training

def examples(docs, tok, c, family, emptykeep=None, seed=0):
    """(input ids, labels) for decoders or (src, decoder input, labels) for LegalSeq, one per chunk. Chunks with no
    aligned summary sentence (target = EOS only) are down-sampled to `emptykeep` so the model does not learn to stop
    immediately."""
    mk = tok.encode(MARK)
    cap = int(c.get("maxtarget", 640))
    keep = float(c.get("emptykeep", 0.25) if emptykeep is None else emptykeep)
    rng = random.Random(seed)
    out = []
    for d in docs:
        for ch, tg in zip(d["chunks"], d["targets"]):
            if not tg and rng.random() > keep:
                continue
            t = tg[:cap] + [tok.eos]
            if family == "seq2seq":
                out.append((ch, [tok.bos] + t[:-1], t))
            else:
                out.append((ch + mk + t, [-100] * (len(ch) + len(mk)) + t))
    return out


def collate(batch, family, dev, pad=0):
    if family == "seq2seq":
        S = max(len(x[0]) for x in batch)
        T = max(len(x[1]) for x in batch)
        src = torch.tensor([x[0] + [pad] * (S - len(x[0])) for x in batch], device=dev)
        mask = torch.tensor([[1] * len(x[0]) + [0] * (S - len(x[0])) for x in batch], device=dev, dtype=torch.bool)
        ids = torch.tensor([x[1] + [pad] * (T - len(x[1])) for x in batch], device=dev)
        y = torch.tensor([x[2] + [-100] * (T - len(x[2])) for x in batch], device=dev)
        return (src, mask, ids), y
    L = max(len(x[0]) for x in batch)
    ids = torch.tensor([x[0] + [pad] * (L - len(x[0])) for x in batch], device=dev)
    y = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch], device=dev)
    return (ids,), y


def logits_of(m, inp, family):
    if family == "seq2seq":
        return m(inp[0], inp[1], inp[2])["logits"]
    return m(inp[0])["logits"]


def batchify(ex, B, rng, shuffle=True):
    idx = sorted(range(len(ex)), key=lambda i: len(ex[i][0]) + (len(ex[i][1]) if len(ex[i]) == 3 else 0))
    bs = [idx[i:i + B] for i in range(0, len(idx), B)]
    if shuffle:
        rng.shuffle(bs)
    return bs


def shiftloss(lg, y, family):
    """Next-token CE on labelled positions (decoder labels are aligned with inputs, so shift; LegalSeq is pre-shifted)."""
    if family != "seq2seq":
        lg, y = lg[:, :-1], y[:, 1:]
    return F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100), lg, y


def train(cfg):
    c = cfg.sm
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = gettok(loadcfg(c.backbone.config))
    data = loaddata(cfg)
    m, mcfg, family = make(cfg, dev)
    ex = examples(data["train"], tok, c, family)
    vx = examples(data["val"], tok, c, family, emptykeep=1.0)
    print(f"chunk examples: {len(ex)} train (empty-target chunks kept at {float(c.get('emptykeep', 0.25)):.0%}), {len(vx)} held-out")
    teacher = None
    if c.get("kd", {}).get("teacher"):
        teacher, _, tfam = loadsm(path(c.kd.teacher), dev)
        assert tfam == family, "word-level KD needs teacher and student of the same family (same tokenisation of inputs)"
        for p in teacher.parameters():
            p.requires_grad = False
    groups = [{"params": [p for p in m.parameters() if p.requires_grad], "lr": float(c.lr)}]
    opt = torch.optim.AdamW(groups, weight_decay=float(c.get("wd", 0.01)))
    B, acc, E = int(c.batch), int(c.get("accum", 1)), float(c.epochs)
    steps = max(1, int(len(batchify(ex, B, random.Random(0))) * E / acc))
    warm = int(float(c.get("warmup", 0.05)) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    ada = AdaLoRAController(m, float(c.get("target", 4)), steps, int(0.1 * steps), int(0.3 * steps)) if c.method == "adalora" else None
    tot, trn = count(m)
    run = path(f"runs/sm/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | {mcfg.name} from {c.backbone.run}:{c.backbone.ckpt} | method {c.method} | chunk {c.chunk} | params "
          f"{tot / 1e6:.1f}M, trainable {trn / 1e6:.2f}M ({100 * trn / tot:.2f}%) | {len(ex)} chunk examples, {steps} steps"
          + (" | word-level KD" if teacher is not None else ""), flush=True)
    pr = c.get("precision", "bf16")
    T_, alpha = float(c.get("kd", {}).get("T", 2.0)), float(c.get("kd", {}).get("alpha", 0.5))
    every = int(c.get("evalevery", 0)) or max(1, steps // 4)
    best, step, micro, t0, tl = float("inf"), 0, 0, time.time(), 0.0
    done = False
    while not done:
        for b in batchify(ex, B, rng):
            inp, y = collate([ex[i] for i in b], family, dev)
            with autocast(dev, pr):
                lg = logits_of(m, inp, family)
            loss, lgs, ys = shiftloss(lg, y, family)
            if teacher is not None:
                with torch.no_grad(), autocast(dev, pr):
                    tlg = logits_of(teacher, inp, family)
                tlg = tlg[:, :-1] if family != "seq2seq" else tlg
                msk = ys != -100
                kl = F.kl_div(F.log_softmax(lgs.float()[msk] / T_, -1), F.softmax(tlg.float()[msk] / T_, -1),
                              reduction="batchmean") * T_ * T_
                loss = alpha * loss + (1 - alpha) * kl
            (loss / acc).backward()
            tl += float(loss.detach())
            micro += 1
            if micro % acc:
                continue
            if ada is not None:
                (float(c.get("orth", 0.1)) * ada.penalty()).backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], float(c.get("clip", 1.0)))
            if ada is not None:
                ada.update(step)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} | loss {tl / (50 * acc):.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                tl = 0.0
            if step % every == 0 or step >= steps:
                vl = valloss(m, vx, family, dev, pr, int(c.get("evalbatch", 16)))
                print(f"eval step {step}: held-out chunk loss {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f})", flush=True)
                if vl < best:
                    best = vl
                    t1, n1 = count(m)
                    torch.save({"model": m.state_dict(), "cfg": plain(cfg), "step": step, "val_loss": vl, "params": t1,
                                "trainable": n1, "family": family}, run / "best.pt")
                    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": c.method, "step": step,
                                                  "val_loss": vl, "params": t1, "trainable": n1, "model": mcfg.name}, 1)
            if step >= steps:
                done = True
                break
    print(f"best held-out chunk loss {best:.4f}")


@torch.no_grad()
def valloss(m, vx, family, dev, pr, B):
    was = m.training
    m.eval()
    tot, n = 0.0, 0
    for b in batchify(vx, B, None, shuffle=False):
        inp, y = collate([vx[i] for i in b], family, dev)
        with autocast(dev, pr):
            lg = logits_of(m, inp, family)
        loss, _, ys = shiftloss(lg, y, family)
        k = int((ys != -100).sum())
        tot += float(loss) * k
        n += k
    m.train(was)
    return tot / max(1, n)


# ----------------------------------------------------------------------------- generation

@torch.no_grad()
def generate(m, family, tok, prompts, g, dev, pr, B, ctx):
    """Greedy (or sampled) chunk summaries with KV caching; each row stops at EOS or g.maxnew tokens. Decoder prompts
    in a batch are cut to the batch's shortest length from the left (prompts are sorted by length, so little is cut)."""
    mk = tok.encode(MARK)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    maxnew = int(g.maxnew)
    for a in range(0, len(order), B):
        idx = order[a:a + B]
        n = len(idx)
        with autocast(dev, pr):
            if family == "seq2seq":
                S = max(len(prompts[i]) for i in idx)
                src = torch.tensor([prompts[i] + [0] * (S - len(prompts[i])) for i in idx], device=dev)
                mask = torch.tensor([[1] * len(prompts[i]) + [0] * (S - len(prompts[i])) for i in idx], device=dev, dtype=torch.bool)
                enc, _ = m.encode(src, mask)
                cache = m.newcache()
                step = lambda ids: m.decode(ids, enc, mask, cache, last=True)[0]
                logits = step(torch.full((n, 1), tok.bos, device=dev))[:, -1]
            else:
                L = min(min(len(prompts[i]) for i in idx), ctx - maxnew - len(mk))
                x = torch.tensor([prompts[i][-L:] + mk for i in idx], device=dev)
                cache = m.newcache()
                step = lambda ids: m(ids, cache=cache, last=True)["logits"]
                logits = step(x)[:, -1]
        pick = Picker(g, n, logits.size(-1), [tok.pad, tok.bos, tok.unk], dev, 0)
        hist = [[] for _ in range(n)]
        active = torch.ones(n, dtype=torch.bool, device=dev)
        minnew = int(g.get("minnew", 0))
        for t in range(maxnew):
            if t < minnew:                      # minimum chunk-summary length: no EOS before `minnew` tokens
                logits = logits.clone()
                logits[:, tok.eos] = float("-inf")
            nxt = pick(logits, hist, active, tok.eos)
            nxt = torch.where(active, nxt, torch.full_like(nxt, tok.eos))
            for r, v in enumerate(nxt.tolist()):
                if active[r]:
                    if v == tok.eos:
                        active[r] = False
                    else:
                        hist[r].append(v)
            pick.update(hist)
            if not active.any():
                break
            with autocast(dev, pr):
                logits = step(nxt[:, None])[:, -1]
        for r, i in enumerate(idx):
            outs[i] = tok.decode(hist[r])
    return outs


def tri(s):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def assemble(chunk_summaries, maxwords, block=0.5):
    """Concatenate chunk summaries in order, drop sentences whose trigrams are mostly already present (trigram blocking),
    cut to `maxwords` words at a sentence boundary when possible."""
    seen, keep = set(), []
    for cs in chunk_summaries:
        for s in sentences(cs):
            t = tri(s)
            if t and len(t & seen) / len(t) > block:
                continue
            keep.append(s)
            seen |= t
    out, n = [], 0
    for s in keep:
        w = s.split()
        if n + len(w) > maxwords:
            if n < 0.8 * maxwords:
                out.append(" ".join(w[:maxwords - n]))
            break
        out.append(s)
        n += len(w)
    return " ".join(out).strip() or " ".join(" ".join(chunk_summaries).split()[:maxwords])


# ----------------------------------------------------------------------------- metrics

def lcs(a, b):
    if len(a) > len(b):
        a, b = b, a
    masks = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    s = 0
    for t in b:
        mm = masks.get(t, 0)
        x = mm | s
        s = x & ~(x - ((s << 1) | 1))
    return bin(s).count("1")


def rouge(pred, ref):
    """ROUGE-1 / ROUGE-L F1 on lower-cased word tokens (no stemming; a close proxy of the official scorer)."""
    p, r = re.findall(r"[a-z0-9]+", pred.lower()), re.findall(r"[a-z0-9]+", ref.lower())
    if not p or not r:
        return 0.0, 0.0
    ov = sum((Counter(p) & Counter(r)).values())
    f = lambda o: 0.0 if o == 0 else 2 * (o / len(p)) * (o / len(r)) / (o / len(p) + o / len(r))
    return f(ov), f(lcs(p, r))


def evaluate(preds, refs):
    sc = [rouge(preds[k], refs[k]) for k in refs]
    return float(np.mean([s[0] for s in sc])), float(np.mean([s[1] for s in sc]))


def official(cfg, run, preds, refs):
    import subprocess
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase6.py"))
    if not ev.exists():
        return
    writejson(run / "val_truth.json", refs)
    writejson(run / "val_pred.json", [{"id": k, "summary": preds[k]} for k in refs])
    r = subprocess.run([sys.executable, str(ev), "--truth", str(run / "val_truth.json"), "--prediction", str(run / "val_pred.json")],
                       capture_output=True, text=True)
    print("official evaluator on the held-out split: " + " | ".join(x.strip() for x in (r.stdout + r.stderr).splitlines() if x.strip())[-400:])


# ----------------------------------------------------------------------------- prediction

def predict(cfg):
    """Generate chunk summaries for the held-out and test documents, assemble, tune the document word budget (ratio of
    the document length) on the held-out split for ROUGE-L, write the submission."""
    c = cfg.sm
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = gettok(loadcfg(c.backbone.config))
    data = loaddata(cfg)
    run = path(f"runs/sm/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    m, obj, family = loadsm(path(f"runs/sm/{c.get('fromrun', cfg.name)}"), dev)
    g = wrap({"mode": "greedy", **plain(c.gen)})
    ctx = int(c.get("ctx", 2048))
    lead = c.get("leadonly", False)            # ablation: summarise the first chunk only
    raw = {}
    for split in ("val", "test"):
        docs = data[split]
        prompts, owner = [], []
        for d in docs:
            for j, ch in enumerate(d["chunks"][:1] if lead else d["chunks"]):
                prompts.append(ch)
                owner.append((d["id"], j))
        t0 = time.time()
        outs = generate(m, family, tok, prompts, g, dev, c.get("precision", "bf16"), int(c.get("genbatch", 16)), ctx)
        per = {}
        for (i, j), o in zip(owner, outs):
            per.setdefault(i, []).append((j, o))
        raw[split] = {i: [o for _, o in sorted(v)] for i, v in per.items()}
        print(f"{split}: {len(prompts)} chunks from {len(docs)} docs generated in {time.time() - t0:.0f}s", flush=True)
    torch.save({"raw": raw, "doc_words": {d["id"]: d["doc_words"] for s in ("val", "test") for d in data[s]}}, run / "raw.pt")
    finish(cfg, run, raw, data, {"method": c.method, "params": obj["params"], "trainable": obj["trainable"], "step": obj["step"],
                                 "val_loss": obj["val_loss"]})


def finish(cfg, run, raw, data, meta):
    c = cfg.sm
    dw = {d["id"]: d["doc_words"] for s in ("val", "test") for d in data[s]}
    refs = {d["id"]: d["summary"] for d in data["val"]}
    tr = data["train"]
    ratio = float(np.median([len(d["summary"].split()) / max(1, d["doc_words"]) for d in tr]))
    lo, hi = int(c.get("minwords", 80)), int(c.get("maxwords", 1500))
    budget = lambda i, mult: int(np.clip(mult * ratio * dw[i], lo, hi))
    natural = {i: assemble(v, 10 ** 6) for i, v in raw["val"].items()}
    nat_len = np.mean([len(natural[i].split()) for i in refs])
    best = None
    for mult in c.get("lengrid", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]):
        preds = {i: assemble(raw["val"][i], budget(i, mult)) for i in refs}
        r1, rl = evaluate(preds, refs)
        print(f"  word budget {mult:.2f} x ratio: held-out ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} | mean words "
              f"{np.mean([len(preds[i].split()) for i in refs]):.0f}")
        if best is None or rl + r1 > best[0]:
            best = (rl + r1, mult, r1, rl)
    _, mult, r1, rl = best
    preds = {i: assemble(raw["val"][i], budget(i, mult)) for i in refs}
    print(f"chosen budget {mult} x median summary/document ratio ({ratio:.3f}) | held-out ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} | "
          f"natural (EOS-stopped) length {nat_len:.0f} words vs reference {np.mean([len(refs[i].split()) for i in refs]):.0f}")
    official(cfg, run, preds, refs)
    test = {i: assemble(v, budget(i, mult)) for i, v in raw["test"].items()}
    for i in test:
        if not test[i].strip():
            test[i] = "The appeal is disposed of."      # never submit an empty summary (the evaluator rejects it)
    res = [{"id": d["id"], "summary": test[d["id"]]} for d in data["test"]]
    sd = subdir(cfg)
    writejson(sd / "prediction.json", res, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, **meta, "budget_mult": mult, "ratio": ratio,
                                 "val_rouge1": r1, "val_rougel": rl, "test_docs": len(res),
                                 "test_mean_words": float(np.mean([len(r["summary"].split()) for r in res]))}, 1)
    torch.save({"val": preds, "test": test}, run / "summaries.pt")
    print(f"ok: {len(res)} test summaries, mean {np.mean([len(r['summary'].split()) for r in res]):.0f} words -> {sd / 'submission.zip'}")


def mbr(cfg):
    """Document-level minimum-Bayes-risk selection over the members' final summaries (utility: ROUGE-1 + ROUGE-L F1
    against the other candidates; Kumar & Byrne 2004; Freitag et al. 2022)."""
    e = cfg.ens
    data = loaddata(cfg)
    ms = []
    for r in e.runs:
        p = path(r) / "summaries.pt"
        if p.exists():
            ms.append((r, torch.load(p, map_location="cpu", weights_only=False)))
        else:
            print(f"skip {r}: no summaries.pt")
    refs = {d["id"]: d["summary"] for d in data["val"]}
    q = []
    for (r, m) in ms:
        r1, rl = evaluate({i: m["val"][i] for i in refs}, refs)
        q.append(r1 + rl)
        print(f"  member {Path(r).name}: held-out ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f}")
    # weak members would win MBR by agreeing with each other: keep members within `keep` of the best held-out quality and
    # weight every candidate's agreement by its member's held-out quality
    keepf = float(e.get("keep", 0.8))
    sel = [k for k in range(len(ms)) if q[k] >= keepf * max(q)]
    ms, q = [ms[k] for k in sel], [q[k] for k in sel]
    print(f"MBR members (>= {keepf:.0%} of the best held-out ROUGE-1 + ROUGE-L): {[Path(r).name for r, _ in ms]}")
    out, picks = {}, Counter()
    for split in ("val", "test"):
        out[split] = {}
        for i in ms[0][1][split]:
            cands = [m[split][i] for _, m in ms]
            u = [q[j] * sum(q[k] * sum(rouge(a, b)) for k, b in enumerate(cands) if k != j) for j, a in enumerate(cands)] \
                if len(cands) > 1 else [1.0]
            j = int(np.argmax(u))
            out[split][i] = cands[j]
            if split == "test":
                picks[Path(ms[j][0]).name] += 1
    r1, rl = evaluate(out["val"], refs)
    print(f"MBR over {len(ms)} members: held-out ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} | test picks {dict(picks)}")
    run = path(f"runs/sm/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    official(cfg, run, out["val"], refs)
    torch.save({"val": out["val"], "test": out["test"]}, run / "summaries.pt")
    res = [{"id": d["id"], "summary": out["test"][d["id"]]} for d in data["test"]]
    sd = subdir(cfg)
    writejson(sd / "prediction.json", res, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": "document-level MBR over member summaries",
                                 "members": [r for r, _ in ms], "val_rouge1": r1, "val_rougel": rl, "test_picks": dict(picks)}, 1)
    print(f"ok: {len(res)} test summaries -> {sd / 'submission.zip'}")


def hybrid(cfg):
    """Abstract-then-extract fill: the member's generated summary stays as the backbone; if it is shorter than a word budget
    (multiplier x median summary/document ratio x document words, tuned on the held-out split), the most salient document
    sentences are appended in document order -- salience = content-word overlap with the generated summary (summary-
    guided extraction) + a position prior for the opening and closing parts of the judgment -- skipping near-duplicates
    (trigram blocking). Cf. extract-then-abstract pipelines (Pilault et al. 2020; Extract-Explain-Abstract, NLLP 2025)."""
    h = cfg.hyb
    f = files(cfg)
    docs = {r[0]: r[1] for r in loadsplit(f["train"]) + loadsplit(f["test"])}
    data = loaddata(cfg)
    refs = {d["id"]: d["summary"] for d in data["val"]}
    dw = {d["id"]: d["doc_words"] for s in ("val", "test") for d in data[s]}
    ratio = float(np.median([len(d["summary"].split()) / max(1, d["doc_words"]) for d in data["train"]]))
    src = path(f"runs/sm/{h.fromrun}")
    if h.get("source", "raw") == "raw":
        raw = torch.load(src / "raw.pt", map_location="cpu", weights_only=False)["raw"]
        base = {s: {i: assemble(v, 10 ** 6) for i, v in raw[s].items()} for s in ("val", "test")}
    else:
        base = torch.load(src / "summaries.pt", map_location="cpu", weights_only=False)
    sents = {i: sentences(docs[i]) for s in ("val", "test") for i in base[s]}

    def build(i, gen, mult, pw):
        T = int(np.clip(mult * ratio * dw[i], int(h.get("minwords", 80)), int(h.get("maxwords", 2000))))
        out = [x for x in sentences(gen)]
        n = sum(len(x.split()) for x in out)
        if n >= T:
            return assemble([gen], T)
        gw = content(gen)
        S = sents[i]
        N = max(1, len(S))
        sc = []
        for k, x in enumerate(S):
            w = content(x)
            if len(x.split()) < 6 or not w:
                continue
            pos = k / N
            sc.append((len(w & gw) / len(w) + pw * (1.0 if pos < 0.15 or pos > 0.85 else 0.0), k))
        seen = set().union(*[tri(x) for x in out]) if out else set()
        chosen = []
        for _, k in sorted(sc, reverse=True):
            t = tri(S[k])
            if not t or len(t & seen) / len(t) > 0.5:
                continue
            chosen.append(k)
            seen |= t
            n += len(S[k].split())
            if n >= T:
                break
        return " ".join(out + [S[k] for k in sorted(chosen)])

    best = None
    for mult in h.get("lengrid", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]):
        for pw in h.get("posgrid", [0.0, 0.2]):
            pv = {i: build(i, base["val"][i], mult, pw) for i in refs}
            r1, rl = evaluate(pv, refs)
            print(f"  budget {mult:.1f} x ratio, position weight {pw}: held-out ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} | mean words "
                  f"{np.mean([len(pv[i].split()) for i in refs]):.0f}")
            if best is None or r1 + rl > best[0]:
                best = (r1 + rl, mult, pw, r1, rl)
    _, mult, pw, r1, rl = best
    gr1, grl = evaluate({i: base["val"][i] for i in refs}, refs)
    print(f"generated only: ROUGE-1 {gr1:.4f} ROUGE-L {grl:.4f} ({np.mean([len(base['val'][i].split()) for i in refs]):.0f} words) | "
          f"chosen fill: budget {mult} x ratio {ratio:.3f}, position weight {pw}: ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} | reference "
          f"{np.mean([len(refs[i].split()) for i in refs]):.0f} words")
    run = path(f"runs/sm/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    pv = {i: build(i, base["val"][i], mult, pw) for i in refs}
    official(cfg, run, pv, refs)
    pt = {i: build(i, base["test"][i], mult, pw) or base["test"][i] for i in base["test"]}
    res = [{"id": d["id"], "summary": pt[d["id"]] if pt[d["id"]].strip() else "The appeal is disposed of."} for d in data["test"]]
    sd = subdir(cfg)
    writejson(sd / "prediction.json", res, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": "abstract-then-extract fill", "from": h.fromrun,
                                 "source": h.get("source", "raw"), "budget_mult": mult, "position_weight": pw, "val_rouge1": r1,
                                 "val_rougel": rl, "gen_only_rouge1": gr1, "gen_only_rougel": grl,
                                 "test_mean_words": float(np.mean([len(r["summary"].split()) for r in res]))}, 1)
    torch.save({"val": pv, "test": pt}, run / "summaries.pt")
    print(f"ok: {len(res)} test summaries, mean {np.mean([len(r['summary'].split()) for r in res]):.0f} words -> {sd / 'submission.zip'}")


class EmbScorer:
    """Local BERTScore proxy: greedy cosine matching of token states of a FROZEN Phase-1 LM (no idf), F1 averaged over
    several layers (as our Phase-1 embedding utility, F2 / Z13); long texts are encoded in `seg`-token segments."""

    def __init__(self, cfg, dev):
        q = cfg.proxy
        obj = torch.load(ckptpath(path(q.run), q.ckpt), map_location="cpu", weights_only=False)
        self.m = mbuild(wrap(obj["cfg"]["model"]))
        self.m.load_state_dict(obj["model"])
        self.m.to(dev).eval()
        self.tok = gettok(loadcfg(q.config))
        self.layers, self.seg, self.maxtok, self.dev = list(q.get("layers", [10, 13, 16])), int(q.get("seg", 1024)), \
            int(q.get("maxtok", 3072)), dev
        self.cache = {}

    @torch.no_grad()
    def emb(self, text):
        if text in self.cache:
            return self.cache[text]
        ids = self.tok.encode(text)[:self.maxtok] or [self.tok.eos]
        parts = [[] for _ in self.layers]
        for a in range(0, len(ids), self.seg):
            x = torch.tensor([ids[a:a + self.seg]], device=self.dev)
            with autocast(self.dev, "bf16"):
                hs = self.m(x, hiddens=True)["hiddens"]
            for k, l in enumerate(self.layers):
                parts[k].append(F.normalize(hs[l][0].float(), dim=-1))
        e = [torch.cat(p_) for p_ in parts]
        if len(self.cache) < 5000:
            self.cache[text] = e
        return e

    def f1(self, pred, ref):
        A, B = self.emb(pred), self.emb(ref)
        out = 0.0
        for a, b in zip(A, B):
            S = a @ b.T
            P, R = S.max(1).values.mean(), S.max(0).values.mean()
            out += float(2 * P * R / (P + R).clamp_min(1e-6))
        return out / len(A)

    def score(self, preds, refs):
        return float(np.mean([self.f1(preds[i], refs[i]) for i in refs]))


def hybrid2(cfg):
    """Hybrid tuned on the local BERTScore proxy: faithfulness filter on generated sentences (drop those whose content
    overlap with the source document is below tau; cf. Maynez et al. 2020), fraction of the (filtered) generated text
    kept, then summary-guided extractive fill to a word budget; the configuration with the best held-out proxy wins."""
    h = cfg.hyb
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = files(cfg)
    docs = {r[0]: r[1] for r in loadsplit(f["train"]) + loadsplit(f["test"])}
    data = loaddata(cfg)
    refs = {d["id"]: d["summary"] for d in data["val"]}
    dw = {d["id"]: d["doc_words"] for s in ("val", "test") for d in data[s]}
    ratio = float(np.median([len(d["summary"].split()) / max(1, d["doc_words"]) for d in data["train"]]))
    src = path(f"runs/sm/{h.fromrun}")
    if h.get("source", "raw") == "raw":
        raw = torch.load(src / "raw.pt", map_location="cpu", weights_only=False)["raw"]
        base = {s: {i: assemble(v, 10 ** 6) for i, v in raw[s].items()} for s in ("val", "test")}
    else:
        base = torch.load(src / "summaries.pt", map_location="cpu", weights_only=False)
    sents = {i: sentences(docs[i]) for s in ("val", "test") for i in base[s]}
    dwords = {i: content(docs[i]) for s in ("val", "test") for i in base[s]}
    pw = float(h.get("posweight", 0.2))

    def build(i, gen, mult, tau, frac):
        T = int(np.clip(mult * ratio * dw[i], int(h.get("minwords", 80)), int(h.get("maxwords", 2000))))
        g = [x for x in sentences(gen) if not content(x) or len(content(x) & dwords[i]) / len(content(x)) >= tau]
        keepw = frac * T
        out, n = [], 0
        for x in g:
            if n >= keepw:
                break
            out.append(x)
            n += len(x.split())
        gw = content(gen)
        S, N = sents[i], max(1, len(sents[i]))
        sc = []
        for k, x in enumerate(S):
            w = content(x)
            if len(x.split()) < 6 or not w:
                continue
            sc.append((len(w & gw) / len(w) + pw * (1.0 if k / N < 0.15 or k / N > 0.85 else 0.0), k))
        seen = set().union(*[tri(x) for x in out]) if out else set()
        chosen = []
        for _, k in sorted(sc, reverse=True):
            if n >= T:
                break
            t = tri(S[k])
            if not t or len(t & seen) / len(t) > 0.5:
                continue
            chosen.append(k)
            seen |= t
            n += len(S[k].split())
        text = " ".join(out + [S[k] for k in sorted(chosen)])
        return " ".join(text.split()[:int(T * 1.1)]) or gen

    es = EmbScorer(cfg, dev)
    gen_only = {i: base["val"][i] for i in refs}
    print(f"proxy calibration | generated only: proxy {es.score(gen_only, refs):.4f} ROUGE-1/L {evaluate(gen_only, refs)} | "
          f"round-2 fill (budget 1.0, tau 0, frac 1.0): proxy {es.score({i: build(i, base['val'][i], 1.0, 0.0, 1.0) for i in refs}, refs):.4f}",
          flush=True)
    best = None
    for mult in h.get("lengrid", [0.75, 1.0, 1.25]):
        for tau in h.get("taugrid", [0.0, 0.5, 0.7]):
            for frac in h.get("fracgrid", [0.0, 0.25, 0.5, 1.0]):
                pv = {i: build(i, base["val"][i], mult, tau, frac) for i in refs}
                px = es.score(pv, refs)
                r1, rl = evaluate(pv, refs)
                print(f"  budget {mult:.2f} tau {tau:.1f} generated-fraction {frac:.2f}: proxy {px:.4f} | ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} | "
                      f"words {np.mean([len(pv[i].split()) for i in refs]):.0f}", flush=True)
                if best is None or px > best[0]:
                    best = (px, mult, tau, frac, r1, rl)
    px, mult, tau, frac, r1, rl = best
    print(f"chosen: budget {mult} tau {tau} generated-fraction {frac} | proxy {px:.4f} ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f}")
    run = path(f"runs/sm/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    pv = {i: build(i, base["val"][i], mult, tau, frac) for i in refs}
    pt = {i: build(i, base["test"][i], mult, tau, frac) for i in base["test"]}
    res = [{"id": d["id"], "summary": pt[d["id"]] if pt[d["id"]].strip() else "The appeal is disposed of."} for d in data["test"]]
    sd = subdir(cfg)
    writejson(sd / "prediction.json", res, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": "hybrid tuned on a Phase-1 embedding BERTScore proxy",
                                 "from": h.fromrun, "budget_mult": mult, "tau": tau, "gen_fraction": frac, "val_proxy": px,
                                 "val_rouge1": r1, "val_rougel": rl,
                                 "test_mean_words": float(np.mean([len(r["summary"].split()) for r in res]))}, 1)
    torch.save({"val": pv, "test": pt}, run / "summaries.pt")
    print(f"ok: {len(res)} test summaries, mean {np.mean([len(r['summary'].split()) for r in res]):.0f} words -> {sd / 'submission.zip'}")


class OfficialScorer:
    """The official metric for held-out model selection only: bert_score with roberta-large (layer 17), from a local
    model directory (cfg.proxy.bsmodel). Never used to produce or pick test outputs per document."""

    def __init__(self, cfg, dev):
        import bert_score
        _bmm = torch.bmm
        if not getattr(torch.bmm, "_outer_safe", False):
            # torch 2.14 routes outer-product bmm (inner dim 1, used by bert_score for its masks) to a Triton kernel that
            # needs Python headers at compile time; that product is just a broadcast multiply
            def safe_bmm(a, b, *args, **kw):
                if a.dim() == 3 and a.shape[-1] == 1 and b.shape[-2] == 1:
                    return a * b
                return _bmm(a, b, *args, **kw)
            safe_bmm._outer_safe = True
            torch.bmm = safe_bmm
        self.bs, self.q, self.dev = bert_score, cfg.proxy, str(dev)

    def score(self, preds, refs):
        ks = list(refs)
        _, _, f = self.bs.score([preds[k] for k in ks], [refs[k] for k in ks], model_type=str(path(self.q.bsmodel)),
                                num_layers=int(self.q.get("bslayer", 17)), batch_size=16, device=self.dev, verbose=False)
        return float(f.mean())


class TruncScorer(EmbScorer):
    """Phase-1 embedding proxy compared on the first `words` words of each text, like BERTScore's 512-token window."""

    def score(self, preds, refs):
        w = int(self.q_words)
        cut = lambda t: " ".join(t.split()[:w])
        return float(np.mean([self.f1(cut(preds[i]), cut(refs[i])) for i in refs]))


def hybrid3(cfg):
    """Chunk-interleaved hybrid built for BERTScore's 512-token window: walking the judgment chunk by chunk, each chunk
    contributes its faithfulness-filtered generated summary followed by its most salient sentences (overlap with the
    generated summary), within a per-chunk word allowance; document order, trigram blocking. The allowance multiplier, the
    generated share and the filter are chosen on the held-out split by the official scorer if available, else by the
    truncated Phase-1 proxy."""
    h = cfg.hyb
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = files(cfg)
    docs = {r[0]: r[1] for r in loadsplit(f["train"]) + loadsplit(f["test"])}
    data = loaddata(cfg)
    refs = {d["id"]: d["summary"] for d in data["val"]}
    ratio = float(np.median([len(d["summary"].split()) / max(1, d["doc_words"]) for d in data["train"]]))
    raw = torch.load(path(f"runs/sm/{h.fromrun}") / "raw.pt", map_location="cpu", weights_only=False)["raw"]
    tok = gettok(loadcfg(cfg.sm.backbone.config))
    C = int(cfg.sm.chunk)
    chs = {i: [s_ for s_, _ in chunkdoc(tok, docs[i], C)] for sp in ("val", "test") for i in raw[sp]}
    for sp in ("val", "test"):
        for i in raw[sp]:
            assert len(chs[i]) == len(raw[sp][i]), f"chunking mismatch for {i}: use the member's chunk size"

    def build(i, gens, mult, frac, tau, maxw):
        out, seen = [], set()

        def add(x):
            t = tri(x)
            if t and len(t & seen) / len(t) > 0.5:
                return 0
            out.append(x)
            seen.update(t)
            return len(x.split())
        for csents, gen in zip(chs[i], gens):
            cw = content(" ".join(csents))
            A = mult * ratio * sum(len(x.split()) for x in csents)
            n = 0
            gs = [x for x in sentences(gen) if content(x) and len(content(x) & cw) / len(content(x)) >= tau]
            for x in gs:
                if n >= frac * A:
                    break
                n += add(x)
            gw = content(gen) or cw
            ranked = sorted(((len(content(x) & gw) / max(1, len(content(x))), k) for k, x in enumerate(csents)
                             if len(x.split()) >= 6), reverse=True)
            pick = []
            for _, k in ranked:
                if n >= A:
                    break
                pick.append(k)
                n += len(csents[k].split())
            for k in sorted(pick):
                add(csents[k])
            if sum(len(x.split()) for x in out) >= maxw:
                break
        return " ".join(" ".join(out).split()[:maxw]) or " ".join(gens)

    if h.get("official") and Path(path(cfg.proxy.bsmodel)).exists():
        sc = OfficialScorer(cfg, dev)
        kind = "official BERTScore (roberta-large L17)"
    else:
        sc = TruncScorer(cfg, dev)
        sc.q_words = int(h.get("window", 400))
        kind = f"Phase-1 proxy on the first {sc.q_words} words"
    maxw = int(h.get("maxwords", 1200))
    gen_only = {i: assemble(raw["val"][i], 10 ** 6) for i in refs}
    print(f"scorer: {kind} | generated only: {sc.score(gen_only, refs):.4f}", flush=True)
    best = None
    for mult in h.get("lengrid", [0.5, 0.75, 1.0, 1.5]):
        for frac in h.get("fracgrid", [0.25, 0.5, 1.0]):
            for tau in h.get("taugrid", [0.0, 0.5]):
                pv = {i: build(i, raw["val"][i], mult, frac, tau, maxw) for i in refs}
                v = sc.score(pv, refs)
                r1, rl = evaluate(pv, refs)
                print(f"  allowance {mult:.2f} x ratio, generated share {frac:.2f}, tau {tau:.1f}: score {v:.4f} | ROUGE-1 {r1:.4f} "
                      f"ROUGE-L {rl:.4f} | words {np.mean([len(pv[i].split()) for i in refs]):.0f}", flush=True)
                if best is None or v > best[0]:
                    best = (v, mult, frac, tau, r1, rl)
    v, mult, frac, tau, r1, rl = best
    print(f"chosen: allowance {mult} x ratio, generated share {frac}, tau {tau} | score {v:.4f} ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f}")
    pt = {i: build(i, raw["test"][i], mult, frac, tau, maxw) for i in raw["test"]}
    res = [{"id": d["id"], "summary": pt[d["id"]] if pt[d["id"]].strip() else "The appeal is disposed of."} for d in data["test"]]
    run = path(f"runs/sm/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"val": {i: build(i, raw["val"][i], mult, frac, tau, maxw) for i in refs}, "test": pt}, run / "summaries.pt")
    sd = subdir(cfg)
    writejson(sd / "prediction.json", res, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": "chunk-interleaved hybrid for the BERTScore window",
                                 "scorer": kind, "from": h.fromrun, "allowance_mult": mult, "gen_share": frac, "tau": tau,
                                 "val_score": v, "val_rouge1": r1, "val_rougel": rl,
                                 "test_mean_words": float(np.mean([len(r["summary"].split()) for r in res]))}, 1)
    print(f"ok: {len(res)} test summaries, mean {np.mean([len(r['summary'].split()) for r in res]):.0f} words -> {sd / 'submission.zip'}")


def bsscan(cfg):
    """Official BERTScore (local roberta-large) of every saved held-out output, to see what actually moves the metric."""
    data = loaddata(cfg)
    refs = {d["id"]: d["summary"] for d in data["val"]}
    sc = OfficialScorer(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for r in sorted(path("runs/sm").glob("*/summaries.pt")):
        s_ = torch.load(r, map_location="cpu", weights_only=False)["val"]
        if set(refs) <= set(s_):
            print(f"{r.parent.name:28s} BERTScore {sc.score({i: s_[i] for i in refs}, refs):.4f} | ROUGE-1/L "
                  f"{tuple(round(x, 4) for x in evaluate({i: s_[i] for i in refs}, refs))}", flush=True)


def main():
    cfg, modes = args()
    {"download": download, "prep": prep, "train": train, "predict": predict, "mbr": mbr, "hybrid": hybrid, "hybrid2": hybrid2, "hybrid3": hybrid3, "bsscan": bsscan}[modes[0] if modes else "prep"](cfg)


if __name__ == "__main__":
    main()
