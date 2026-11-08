"""Phase 5: legal statute identification (LSI, multi-label) with the Phase-1 legal LMs.

Model: a Phase-1 encoder (any adaptation technique, causal mask removed) reads a window of the document; the head is
either a label-wise attention network (LWAN: one attention query per statute over the token states -> a statute-
specific document vector -> its score; CAML, Mullenbach et al. 2018; Chalkidis et al. 2019) or attention + mean pooling
with a linear multi-label layer. Loss: asymmetric loss (ASL, Ridnik et al. 2021) or BCE with capped positive weights.
Decisions: per-statute thresholds tuned on dev for F1 (global threshold for statutes with few dev positives), plus an
"at least one statute" rule when it helps dev macro-F1. Ensembles average member probabilities.

python lsi.py phase5/configs/<cfg>.yaml download | prep | train | predict | ensemble | teach
"""
import hashlib
import json
import math
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner
from cj import batches, inputs, shapekey
from config import args, loadcfg, path, plain, subdir, wrap
from data import writejson
from generate import autocast
from peft import AdaLoRAController, count
from tokenizer import gettok

SPLITS = ("train", "dev", "test")


# ----------------------------------------------------------------------------- data

def readjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def loadsplit(p):
    """[(id, text, [statute ids] or None)]."""
    d = readjson(p)
    recs = d if isinstance(d, list) else [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in d.items()]
    out = []
    for i, r in enumerate(recs):
        lab = next((r[k] for k in ("labels", "gold", "statutes", "label") if k in r), None)
        if lab is not None and not isinstance(lab, list):
            lab = [lab]
        text = r.get("text", r.get("document", ""))
        if isinstance(text, list):
            text = " ".join(text)
        out.append((str(r.get("id", i)), text, None if lab is None else sorted({int(x) for x in lab})))
    return out


def files(cfg):
    c = cfg.lsi
    return {k: path(c.root) / v for k, v in c.files.items()}


def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    c = cfg.lsi
    have = list_repo_files(c.repo, repo_type="dataset")
    print("remote Phase-5 files:", *[f for f in have if f.startswith("Phase_5")], sep="\n  ")
    for k, f in c.files.items():
        if f not in have:
            hits = [x for x in have if x.startswith("Phase_5") and Path(x).name == Path(f).name]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix lsi.files.{k}")
            f = hits[0]
        hf_hub_download(c.repo, f, repo_type="dataset", local_dir=str(path(c.root)))
        print(f"ok {f}")


_TOK = None


def _enc(text):
    return np.asarray(_TOK.encode(text), dtype=np.int16)


def prep(cfg):
    """Tokenise every split once -> runs/lsi/tok/<split>.pt + meta.json; print statistics (lengths, label frequencies,
    labels per doc, overlap of train / dev with test)."""
    import multiprocessing as mp
    global _TOK
    _TOK = gettok(loadcfg(cfg.lsi.backbone.config))
    out = path(cfg.lsi.tokdir)
    out.mkdir(parents=True, exist_ok=True)
    f = files(cfg)
    data = {}
    for split in SPLITS:
        t0 = time.time()
        recs = loadsplit(f[split])
        with mp.get_context("fork").Pool(int(cfg.lsi.get("workers", 32))) as pool:
            toks = pool.map(_enc, [r[1] for r in recs], chunksize=16)
        data[split] = {"ids": [r[0] for r in recs], "tok": toks, "labels": [r[2] for r in recs]}
        torch.save(data[split], out / f"{split}.pt")
        n = np.array([len(t) for t in toks])
        msg = f"{split}: {len(recs)} docs | tokens/doc median {np.median(n):.0f} mean {n.mean():.0f} p90 {np.percentile(n, 90):.0f} " \
              f"max {n.max()} | <=2048: {np.mean(n <= 2048):.1%}"
        if recs[0][2] is not None:
            nl = [len(r[2]) for r in recs]
            msg += f" | labels/doc mean {np.mean(nl):.2f} max {max(nl)} | empty {sum(x == 0 for x in nl)}"
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)
    allab = [l for s in ("train", "dev") for ls in data[s]["labels"] for l in ls]
    K = max(allab) + 1
    ctr = {s: Counter(l for ls in data[s]["labels"] for l in ls) for s in ("train", "dev")}
    th = {hashlib.md5(x.tobytes()).hexdigest() for x in data["test"]["tok"]}
    for s in ("train", "dev"):
        ov = sum(hashlib.md5(x.tobytes()).hexdigest() in th or i in set(data["test"]["ids"]) for i, x in zip(data[s]["ids"], data[s]["tok"]))
        print(f"{s} cases also in test (id or identical text): {ov}")
    tr = np.array([ctr["train"][k] for k in range(K)])
    print(f"statutes K = {K} | train positives per statute: min {tr.min()} median {np.median(tr):.0f} max {tr.max()} | "
          f"statutes with < 20 train docs: {(tr < 20).sum()} | absent from dev: {sum(ctr['dev'][k] == 0 for k in range(K))}")
    writejson(out / "meta.json", {"K": K, "train_counts": tr.tolist(), "dev_counts": [ctr["dev"][k] for k in range(K)]}, 1)


_TESTKEYS = {}


def loadtok(cfg, split):
    """Tokenised split; train / dev cases that also occur in test are removed (`notest`, default on)."""
    d = torch.load(path(cfg.lsi.tokdir) / f"{split}.pt", map_location="cpu", weights_only=False)
    if split == "test" or not cfg.lsi.get("notest", True):
        return d
    key = str(path(cfg.lsi.tokdir))
    if key not in _TESTKEYS:
        t = torch.load(path(cfg.lsi.tokdir) / "test.pt", map_location="cpu", weights_only=False)
        _TESTKEYS[key] = (set(t["ids"]), {hashlib.md5(x.tobytes()).hexdigest() for x in t["tok"]})
    ids, hs = _TESTKEYS[key]
    keep = [i for i, (k, x) in enumerate(zip(d["ids"], d["tok"])) if k not in ids and hashlib.md5(x.tobytes()).hexdigest() not in hs]
    if len(keep) < len(d["ids"]):
        print(f"[notest] {split}: removed {len(d['ids']) - len(keep)} of {len(d['ids'])} cases that also occur in the test set")
    return {k: [d[k][i] for i in keep] for k in ("ids", "tok", "labels")}


def meta(cfg):
    return readjson(path(cfg.lsi.tokdir) / "meta.json")


def onehot(labels, K):
    Y = np.zeros((len(labels), K), dtype=np.uint8)
    for i, ls in enumerate(labels):
        Y[i, [l for l in ls if l < K]] = 1
    return Y


# ----------------------------------------------------------------------------- metrics and thresholds

def scores(Y, P):
    """Macro-F1 (sklearn average='macro', zero_division=0 over all K columns), micro-F1, subset accuracy, per-label F1."""
    tp = (Y & P).sum(0).astype(float)
    fp = ((1 - Y) & P).sum(0).astype(float)
    fn = (Y & (1 - P)).sum(0).astype(float)
    pr = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    rc = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(tp), where=pr + rc > 0)
    mi = 2 * tp.sum() / max(1.0, 2 * tp.sum() + fp.sum() + fn.sum())
    return {"macro": float(f1.mean()), "micro": float(mi), "subset": float((Y == P).all(1).mean()), "per": f1}


GRID = np.round(np.arange(0.02, 0.96, 0.02), 2)


def decide(S, thr, atleast1):
    P = (S > thr[None]).astype(np.uint8)
    if atleast1:
        empty = P.sum(1) == 0
        P[empty, S[empty].argmax(1)] = 1
    return P


def tune(Y, S, minpos=5):
    """Global threshold maximising macro-F1, then per-statute thresholds maximising that statute's F1 when it has at
    least `minpos` dev positives (others keep the global one); finally whether the at-least-one rule helps."""
    K = Y.shape[1]
    best = max((scores(Y, decide(S, np.full(K, t), False))["macro"], t) for t in GRID)
    g = best[1]
    thr = np.full(K, g)
    pos = Y.sum(0)
    for k in range(K):
        if pos[k] < minpos:
            continue
        pk = S[:, k][:, None] > GRID[None]
        y = Y[:, k][:, None].astype(bool)
        tp = (pk & y).sum(0)
        f1 = 2 * tp / np.maximum(1, 2 * tp + (pk & ~y).sum(0) + (~pk & y).sum(0))
        thr[k] = GRID[int(f1.argmax())]
    a0 = scores(Y, decide(S, thr, False))["macro"]
    a1 = scores(Y, decide(S, thr, True))["macro"]
    return thr, a1 > a0, {"global": float(g), "global_macro": best[0], "perlabel_macro": a0, "atleast1_macro": a1}


# ----------------------------------------------------------------------------- model

class LSIModel(nn.Module):
    def __init__(self, enc, dim, K, c):
        super().__init__()
        self.enc = enc
        self.enc.cls = nn.Identity()
        self.headkind = c.get("head", "lwan")
        dr = float(c.get("hdrop", 0.1))
        self.drop = nn.Dropout(dr)
        if self.headkind == "lwan":
            self.q = nn.Parameter(torch.randn(K, dim) * 0.02)        # one attention query per statute
            self.w = nn.Parameter(torch.randn(K, dim) * 0.02)        # one output vector per statute
            self.b = nn.Parameter(torch.zeros(K))
            self.ln = nn.LayerNorm(dim)
        else:
            self.att = nn.Linear(dim, 1)
            self.head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dr), nn.Linear(dim, K))

    def forward(self, ids):
        if ids.dim() == 3:                     # multiple-instance: (docs, chunks, tokens) -> LSE pooling of chunk logits
            B, k, T = ids.shape
            z = self.forward(ids.reshape(B * k, T)).reshape(B, k, -1)
            return torch.logsumexp(z, 1) - math.log(k)
        h = self.enc.hidden(ids).float()
        if self.headkind == "lwan":
            h = self.ln(h)
            a = torch.softmax(torch.einsum("btd,kd->bkt", h, self.q) / math.sqrt(h.size(-1)), -1)
            v = self.drop(torch.einsum("bkt,btd->bkd", a, h))
            return (v * self.w[None]).sum(-1) + self.b
        a = torch.softmax(self.att(h).squeeze(-1), -1)
        return self.head(self.drop(torch.cat([(a[..., None] * h).sum(1), h.mean(1)], -1)))


def milstack(t, N, W):
    """Up to N windows of exactly W tokens evenly covering the first min(len, N * W) tokens (one window if len <= W)."""
    n = len(t)
    if n <= W:
        return t[None]
    span = min(n, N * W)
    k = min(N, math.ceil(span / W))
    if k == 1:
        return t[:W][None]
    starts = [round(j * (span - W) / (k - 1)) for j in range(k)]
    return np.stack([t[s_:s_ + W] for s_ in starts])


def seqsfor(toks, c, off=0):
    if int(c.get("milchunks", 0)) > 1:
        return [milstack(t, int(c.milchunks), int(c.window)) for t in toks]
    return inputs(toks, c, off)


def build(cfg, dev, K):
    t, info = ner.make(wrap({**cfg, "ner": cfg.lsi}), dev, ["x"])
    return LSIModel(t, int(info["mcfg"]["dim"]), K, cfg.lsi).to(dev), info


def loadlsi(run, dev):
    obj = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = wrap(obj["cfg"])
    m, _ = build(cfg, dev, obj["K"])
    m.load_state_dict(obj["model"])
    return m.eval(), obj


DBSTATS = {}


def lossfn(lg, Y, c, posw):
    if c.get("loss", "asl") == "db":
        # distribution-balanced loss (Wu et al. ECCV 2020; Huang et al. EMNLP 2021): rebalanced weights
        # r = (1/n_i) / mean_{pos j}(1/n_j), smoothed alpha + sigmoid(beta (r - mu)); negative-tolerant regularisation with
        # scale lambda and class-prior logit bias kappa * logit(p_i)
        invn, nu = DBSTATS["invn"], DBSTATS["nu"]
        a_, b_, mu_ = float(c.get("dbalpha", 0.1)), float(c.get("dbbeta", 10.0)), float(c.get("dbmu", 0.05))
        lam = float(c.get("dblambda", 2.0))
        pi = (Y @ invn) / Y.sum(1).clamp(min=1)
        w = a_ + torch.sigmoid(b_ * (invn[None] / pi[:, None].clamp(min=1e-12) - mu_))
        z = lg - nu[None]
        l_ = Y * F.softplus(-z) + (1 - Y) * F.softplus(lam * z) / lam
        return (w * l_).sum(1).mean()
    if c.get("loss", "asl") == "asl":
        # asymmetric loss (Ridnik et al. 2021): no focusing on positives, gamma- on negatives, probability shift m
        gp, gn, m = float(c.get("gpos", 0.0)), float(c.get("gneg", 4.0)), float(c.get("clipm", 0.05))
        p = torch.sigmoid(lg)
        pn = (p - m).clamp(min=0)
        lp = Y * (1 - p).pow(gp) * torch.log(p.clamp(min=1e-8))
        ln_ = (1 - Y) * pn.pow(gn) * torch.log((1 - pn).clamp(min=1e-8))
        return -(lp + ln_).sum(1).mean()
    return F.binary_cross_entropy_with_logits(lg, Y, pos_weight=posw, reduction="none").sum(1).mean()


@torch.no_grad()
def probs(model, toks, cfg, dev):
    c = cfg.lsi
    was = model.training
    model.eval()
    offs = list(c.get("tta", [0])) if c.get("tta") else [0]
    K = model.b.numel() if model.headkind == "lwan" else model.head[-1].out_features
    out = np.zeros((len(toks), K), dtype=np.float32)
    if int(c.get("milchunks", 0)) > 1:
        seqs = seqsfor(toks, c)
        for b in batches([shapekey(s_) for s_ in seqs], max(1, int(c.get("evalbatch", 16)) // 2), None, shuffle=False):
            ids = torch.tensor(np.stack([seqs[i] for i in b]).astype(np.int64), device=dev)
            with autocast(dev, c.get("precision", "bf16")):
                out[b] = torch.sigmoid(model(ids).float()).cpu().numpy()
        model.train(was)
        return out
    if int(c.get("chunks", 1)) > 1:
        # chunk-level statute predictions + document-level aggregation: consecutive windows of `window` tokens from the
        # start (up to `chunks`), per-statute max (or mean) over the chunks
        W, N, agg = int(c.window), int(c.chunks), c.get("agg", "max")
        jobs = []
        for di, t in enumerate(toks):
            for j in range(min(N, max(1, math.ceil(len(t) / W)))):
                piece = t[j * W:(j + 1) * W]
                if len(piece) >= 32 or j == 0:
                    jobs.append((di, piece))
        res = [[] for _ in toks]
        for b in batches([len(p_) for _, p_ in jobs], int(c.get("evalbatch", 16)), None, shuffle=False):
            ids = torch.tensor(np.stack([jobs[i][1] for i in b]).astype(np.int64), device=dev)
            with autocast(dev, c.get("precision", "bf16")):
                pr_ = torch.sigmoid(model(ids).float()).cpu().numpy()
            for i, row in zip(b, pr_):
                res[jobs[i][0]].append(row)
        for di, r in enumerate(res):
            out[di] = np.max(r, 0) if agg == "max" else np.mean(r, 0)
        model.train(was)
        return out
    for off in offs:
        seqs = inputs(toks, c, off)
        for b in batches([shapekey(s) for s in seqs], int(c.get("evalbatch", 16)), None, shuffle=False):
            ids = torch.tensor(np.stack([seqs[i] for i in b]).astype(np.int64), device=dev)
            with autocast(dev, c.get("precision", "bf16")):
                lg = model(ids)
            out[b] += torch.sigmoid(lg.float()).cpu().numpy() / len(offs)
    model.train(was)
    return out


def train(cfg):
    c = cfg.lsi
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = int(meta(cfg)["K"])
    tr, va = loadtok(cfg, "train"), loadtok(cfg, "dev")
    if c.get("alldocs"):
        tr = {k: tr[k] + va[k] for k in ("ids", "tok", "labels")}
    if c.get("devfold") is not None:      # dev cross-fitting: train + the other dev fold(s); select on this fold
        fo = devfolds(va["ids"], int(c.get("devfolds", 2)))
        k = int(c.devfold)
        keep = [i for i, d in enumerate(va["ids"]) if fo[d] != k]
        held = [i for i, d in enumerate(va["ids"]) if fo[d] == k]
        tr = {key: tr[key] + [va[key][i] for i in keep] for key in ("ids", "tok", "labels")}
        va = {key: [va[key][i] for i in held] for key in ("ids", "tok", "labels")}
        print(f"dev fold {k}: +{len(keep)} dev docs in training, {len(held)} held out for selection")
    seqs = seqsfor(tr["tok"], c)
    views = [inputs(tr["tok"], c, o) for o in c.augoffsets] if c.get("augoffsets") else None
    Y = onehot(tr["labels"], K).astype(np.float32)
    Yv = onehot(va["labels"], K)
    pos = Y.sum(0)
    posw = torch.tensor(np.minimum(((len(Y) - pos) / np.maximum(pos, 1)) ** 0.5, 20.0), dtype=torch.float32, device=dev)
    prior = np.clip(pos / len(Y), 1e-6, 1 - 1e-6)
    DBSTATS.update(invn=torch.tensor(1.0 / np.maximum(pos, 1), dtype=torch.float32, device=dev),
                   nu=torch.tensor(float(c.get("dbkappa", 0.05)) * np.log(prior / (1 - prior)), dtype=torch.float32, device=dev))
    model, info = build(cfg, dev, K)
    teacher = None
    if c.get("kd", {}).get("teacher"):
        tp = torch.load(path(c.kd.teacher), map_location="cpu", weights_only=False)["train"]
        teacher = np.stack([tp[i] for i in tr["ids"]]).astype(np.float32)
    meth = c.method
    groups = [{"params": [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith("enc.backbone.")], "lr": float(c.lr)},
              {"params": [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("enc.backbone.")], "lr": float(c.headlr)}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=float(c.get("wd", 0.01)))
    B, acc, E = int(c.batch), int(c.get("accum", 1)), float(c.epochs)
    nb = len(batches([shapekey(s) for s in seqs], B, random.Random(0)))
    steps = max(1, int(nb * E / acc))
    warm = int(float(c.get("warmup", 0.06)) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    ada = AdaLoRAController(model, float(c.get("target", 4)), steps, int(0.1 * steps), int(0.3 * steps)) if meth == "adalora" else None
    tot, trn = count(model)
    run = path(f"runs/lsi/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | {info['phase1_model']} from {info['phase1']} | method {meth} | head {c.get('head', 'lwan')} | "
          f"{'MIL x' + str(c.milchunks) + ' chunks | ' if int(c.get('milchunks', 0)) > 1 else ''}"
          f"loss {c.get('loss', 'asl')} | window {c.window} ({c.get('take', 'first')}) | params {tot / 1e6:.1f}M, trainable "
          f"{trn / 1e6:.2f}M ({100 * trn / tot:.2f}%) | train {len(Y)} docs, K {K} | {steps} steps" + (" | KD" if teacher is not None else ""), flush=True)
    pr = c.get("precision", "bf16")
    alpha = float(c.get("kd", {}).get("alpha", 0.5))
    every = int(c.get("evalevery", 0)) or max(1, steps // 4)
    best, step, micro, t0, tl = -1.0, 0, 0, time.time(), 0.0
    log = open(run / "metrics.jsonl", "a")
    done = False
    while not done:
        if views is not None:
            seqs = [views[rng.randrange(len(views))][i] for i in range(len(Y))]
        for b in batches([shapekey(s) for s in seqs], B, rng):
            ids = torch.tensor(np.stack([seqs[i] for i in b]).astype(np.int64), device=dev)
            yy = torch.tensor(Y[b], device=dev)
            with autocast(dev, pr):
                lg = model(ids)
            lg = lg.float()
            loss = lossfn(lg, yy, c, posw)
            if teacher is not None:
                tt = torch.tensor(teacher[b], device=dev)
                loss = alpha * loss + (1 - alpha) * F.binary_cross_entropy_with_logits(lg, tt, reduction="none").sum(1).mean()
            (loss / acc).backward()
            tl += float(loss.detach())
            micro += 1
            if micro % acc:
                continue
            if ada is not None:
                (float(c.get("orth", 0.1)) * ada.penalty()).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(c.get("clip", 1.0)))
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
                if not c.get("alldocs"):
                    S = probs(model, va["tok"], cfg, dev)
                    thr, a1, st = tune(Yv, S, int(c.get("minpos", 5)))
                    sc = scores(Yv, decide(S, thr, a1))
                    print(f"eval step {step}: dev macro-F1 {sc['macro']:.4f} micro {sc['micro']:.4f} subset {sc['subset']:.4f} "
                          f"(global thr {st['global']:.2f}: {st['global_macro']:.4f})", flush=True)
                    log.write(json.dumps({"step": step, "macro": sc["macro"], "micro": sc["micro"]}) + "\n")
                    log.flush()
                    if sc["macro"] > best:
                        best = sc["macro"]
                        save(run, model, cfg, info, step, K, {"macro": sc["macro"], "micro": sc["micro"], "subset": sc["subset"]})
                elif step >= steps:
                    save(run, model, cfg, info, step, K, None)
            if step >= steps:
                done = True
                break
    log.close()
    print(f"best dev macro-F1 {best:.4f}")


def save(run, model, cfg, info, step, K, dev):
    tot, trn = count(model)
    torch.save({"model": model.state_dict(), "cfg": plain(cfg), "step": step, "dev": dev, "K": K,
                "info": {k: v for k, v in info.items() if k != "mcfg"}, "params": tot, "trainable": trn}, run / "best.pt")
    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": cfg.lsi.method, "step": step, "dev": dev,
                                  "params": tot, "trainable": trn, **{k: v for k, v in info.items() if k != "mcfg"}}, 1)


# ----------------------------------------------------------------------------- prediction, ensemble, teacher

def finish(cfg, run, out, dv, ts, K, extra):
    Yv = onehot(dv["labels"], K)
    Sv = np.stack([out["dev"][i] for i in dv["ids"]])
    thr, a1, st = tune(Yv, Sv, int(cfg.lsi.get("minpos", 5)))
    sc = scores(Yv, decide(Sv, thr, a1))
    tc = np.array(meta(cfg)["train_counts"])
    rare, freq = tc < np.percentile(tc, 33), tc >= np.percentile(tc, 67)
    print(f"dev macro-F1 {sc['macro']:.4f} micro {sc['micro']:.4f} subset {sc['subset']:.4f} | global thr {st['global']:.2f} "
          f"({st['global_macro']:.4f}) -> per-statute ({st['perlabel_macro']:.4f}) -> at-least-one {'on' if a1 else 'off'} | "
          f"rare-third F1 {sc['per'][rare].mean():.3f}, frequent-third F1 {sc['per'][freq].mean():.3f}")
    officialdev(cfg, run, dv["ids"], dv["labels"], decide(Sv, thr, a1))
    St = np.stack([out["test"][i] for i in ts["ids"]])
    Pt = decide(St, thr, a1)
    maxpred = int(cfg.lsi.get("maxpred", K - 1))
    preds = [[int(k) for k in np.flatnonzero(p) if k <= maxpred] for p in Pt]
    res = [{"id": i, "pred": p} for i, p in zip(ts["ids"], preds)]
    sd = subdir(cfg)
    writejson(sd / "prediction.json", res, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, **extra, "dev_macro": sc["macro"], "dev_micro": sc["micro"],
                                 "dev_subset": sc["subset"], "thresholds": thr.tolist(), "atleast1": bool(a1), "tuning": st,
                                 "dev_per_label_f1": sc["per"].tolist(), "test_cases": len(res),
                                 "test_labels_per_doc": float(np.mean([len(p) for p in preds]))}, 1)
    print(f"ok: {len(res)} test cases, {np.mean([len(p) for p in preds]):.2f} statutes/doc -> {sd / 'submission.zip'}")


def officialdev(cfg, run, ids, labels, P):
    import subprocess
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase5.py"))
    if not ev.exists():
        return
    seen, truth, pred = set(), [], []
    for i, ls, p in zip(ids, labels, P):
        if i in seen:
            continue
        seen.add(i)
        truth.append({"id": i, "gold": [int(x) for x in ls]})
        pred.append({"id": i, "pred": [int(x) for x in np.flatnonzero(p)]})
    writejson(run / "dev_truth.json", truth)
    writejson(run / "dev_pred.json", pred)
    r = subprocess.run([sys.executable, str(ev), "--truth", str(run / "dev_truth.json"), "--prediction", str(run / "dev_pred.json")],
                       capture_output=True, text=True)
    print("official evaluator on dev: " + " | ".join(x.strip() for x in (r.stdout + r.stderr).splitlines() if x.strip())[-300:])


def devfolds(ids, K):
    """Deterministic K-fold split of the dev cases by id (repeated ids stay together)."""
    u = sorted(set(ids))
    random.Random(0).shuffle(u)
    return {d: i % K for i, d in enumerate(u)}


def devjoin(cfg):
    """Cross-fitted member from dev-fold runs: each dev case takes the probabilities of the run that held it out, test the
    mean of the fold runs; per-statute thresholds tuned on these honest dev predictions."""
    q = cfg.devjoin
    dv, ts = loadtok(cfg, "dev"), loadtok(cfg, "test")
    ms = [torch.load(path(r) / "probs.pt", map_location="cpu", weights_only=False) for r in q.runs]
    fo = devfolds(dv["ids"], len(ms))
    K = ms[0]["K"]
    out = {"K": K, "dev": {d: ms[fo[d]]["dev"][d] for d in set(dv["ids"])},
           "test": {d: np.mean([m["test"][d] for m in ms], 0) for d in ts["ids"]},
           "params": ms[0]["params"], "trainable": ms[0]["trainable"], "phase1": ms[0]["phase1"], "method": ms[0]["method"]}
    for k, r in enumerate(q.runs):
        held = [i for i, d in enumerate(dv["ids"]) if fo[d] == k]
        Y = onehot([dv["labels"][i] for i in held], K)
        S = np.stack([out["dev"][dv["ids"][i]] for i in held])
        thr, a1, _ = tune(Y, S, int(cfg.lsi.get("minpos", 5)))
        print(f"  fold {k} ({r}): held-out dev macro-F1 {scores(Y, decide(S, thr, a1))['macro']:.4f}")
    run = path(f"runs/lsi/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save(out, run / "probs.pt")
    finish(cfg, run, out, dv, ts, K, {"method": f"dev cross-fitted {out['method']} ({len(ms)} folds)", "folds": list(q.runs)})


def predict(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = path(f"runs/lsi/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    model, obj = loadlsi(path(f"runs/lsi/{cfg.lsi.get('fromrun', cfg.name)}"), dev)
    K = obj["K"]
    out = {"params": obj["params"], "trainable": obj["trainable"], "phase1": obj["info"]["phase1"], "method": cfg.lsi.method, "K": K}
    splits = SPLITS if cfg.lsi.get("trainprobs") else ("dev", "test")
    data = {s: loadtok(cfg, s) for s in splits}
    for s in splits:
        t0 = time.time()
        out[s] = dict(zip(data[s]["ids"], probs(model, data[s]["tok"], cfg, dev)))
        print(f"{s}: {len(out[s])} docs scored in {time.time() - t0:.0f}s", flush=True)
    torch.save(out, run / "probs.pt")
    finish(cfg, run, out, data["dev"], data["test"], K, {"method": cfg.lsi.method, "phase1": obj["info"]["phase1"],
                                                         "params": obj["params"], "trainable": obj["trainable"], "step": obj["step"]})


def ensemble(cfg):
    """Weighted mean of member probabilities; member weights tuned greedily for dev macro-F1 (thresholds re-tuned each
    trial); per-statute thresholds on the final mixture."""
    e = cfg.ens
    dv, ts = loadtok(cfg, "dev"), loadtok(cfg, "test")
    ms = []
    for r in e.runs:
        p = path(r) / "probs.pt"
        if p.exists():
            ms.append((r, torch.load(p, map_location="cpu", weights_only=False)))
        else:
            print(f"skip {r}: no probs.pt")
    K = ms[0][1]["K"]
    Yv = onehot(dv["labels"], K)
    P = np.stack([np.stack([m["dev"][i] for i in dv["ids"]]) for _, m in ms])
    mp_ = int(cfg.lsi.get("minpos", 5))
    f = lambda w: scores(Yv, decide((w[:, None, None] * P).sum(0) / w.sum(), *tune(Yv, (w[:, None, None] * P).sum(0) / w.sum(), mp_)[:2]))["macro"]
    w = np.ones(len(ms))
    best = f(w)
    print(f"{len(ms)} members, equal weights: dev {best:.4f} | singles: " +
          ", ".join(f"{Path(r).name} {f(np.eye(len(ms))[j]):.4f}" for j, (r, _) in enumerate(ms)))
    for _ in range(2):
        for j in range(len(ms)):
            for v in e.get("wgrid", [0.0, 0.5, 2.0, 3.0]):
                t_ = w.copy()
                t_[j] = v
                if t_.sum() == 0:
                    continue
                s_ = f(t_)
                if s_ > best + 1e-4:
                    best, w = s_, t_
    print(f"weights {dict(zip([Path(r).name for r, _ in ms], w.tolist()))}: dev {best:.4f}")
    out = {"K": K, "params": 0, "trainable": 0, "phase1": "ensemble", "method": "ensemble"}
    for s, d in (("dev", dv), ("test", ts)):
        out[s] = dict(zip(d["ids"], (w[:, None, None] * np.stack([np.stack([m[s][i] for i in d["ids"]]) for _, m in ms])).sum(0) / w.sum()))
    if all("train" in m for _, m in ms):
        ids = list(ms[0][1]["train"])
        out["train"] = dict(zip(ids, (w[:, None, None] * np.stack([np.stack([m["train"][i] for i in ids]) for _, m in ms])).sum(0) / w.sum()))
    run = path(f"runs/lsi/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save(out, run / "probs.pt")
    finish(cfg, run, out, dv, ts, K, {"method": "ensemble (weighted mean probabilities) + per-statute dev thresholds",
                                      "members": [r for r, _ in ms], "weights": w.tolist()})


def stack(cfg):
    """Stacked binary relevance with label correlations (Godbole & Sarawagi 2004; Montanes et al. 2014): a small network
    maps every member's probabilities for ALL statutes to each statute's score (residual on the mean member logit), so
    co-occurring statutes inform each other. Trained on dev with K-fold cross-fitting (honest out-of-fold dev scores for
    thresholds); test = mean of the fold models."""
    e = cfg.stack
    dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dv, ts = loadtok(cfg, "dev"), loadtok(cfg, "test")
    ms = []
    for r in e.runs:
        p = path(r) / "probs.pt"
        if p.exists():
            ms.append((r, torch.load(p, map_location="cpu", weights_only=False)))
        else:
            print(f"skip {r}: no probs.pt")
    K = ms[0][1]["K"]
    lg = lambda P: np.log(np.clip(P, 1e-5, 1 - 1e-5) / (1 - np.clip(P, 1e-5, 1 - 1e-5)))
    feats = lambda split, ids: np.concatenate([lg(np.stack([m[split][i] for i in ids])) for _, m in ms], 1).astype(np.float32)
    Xd, Xt = feats("dev", dv["ids"]), feats("test", ts["ids"])
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-4
    base_d = Xd.reshape(len(Xd), len(ms), K).mean(1)
    base_t = Xt.reshape(len(Xt), len(ms), K).mean(1)
    Xd, Xt = (Xd - mu) / sd, (Xt - mu) / sd
    Y = onehot(dv["labels"], K).astype(np.float32)
    F_ = int(e.get("folds", 5))
    u = sorted(set(dv["ids"]))
    random.Random(0).shuffle(u)
    fo = {d: i % F_ for i, d in enumerate(u)}
    fold = np.array([fo[d] for d in dv["ids"]])
    oof = np.zeros_like(Y)
    tst = np.zeros((len(Xt), K), dtype=np.float32)
    H = int(e.get("hid", 512))

    class Net(nn.Module):
        def __init__(s_):
            super().__init__()
            s_.f = nn.Sequential(nn.Dropout(float(e.get("indrop", 0.2))), nn.Linear(Xd.shape[1], H), nn.GELU(),
                                 nn.Dropout(float(e.get("drop", 0.3))), nn.Linear(H, K))
            nn.init.zeros_(s_.f[-1].weight)
            nn.init.zeros_(s_.f[-1].bias)

        def forward(s_, x, b):
            return b + s_.f(x)                    # residual on the mean member logit: starts as the plain ensemble

    T = lambda a: torch.tensor(a, device=dev_)
    for k in range(F_):
        tr_, ho = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        torch.manual_seed(k)
        net = Net().to(dev_)
        opt = torch.optim.AdamW(net.parameters(), lr=float(e.get("lr", 1e-3)), weight_decay=float(e.get("wd", 1e-2)))
        for ep in range(int(e.get("epochs", 20))):
            net.train()
            perm = np.random.RandomState(100 * k + ep).permutation(tr_)
            for a in range(0, len(perm), 256):
                bi = perm[a:a + 256]
                loss = F.binary_cross_entropy_with_logits(net(T(Xd[bi]), T(base_d[bi])), T(Y[bi]), reduction="none").sum(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            oof[ho] = torch.sigmoid(net(T(Xd[ho]), T(base_d[ho]))).cpu().numpy()
            tst += torch.sigmoid(net(T(Xt), T(base_t))).cpu().numpy() / F_
        thr, a1, _ = tune(Y[ho].astype(np.uint8), oof[ho], int(cfg.lsi.get("minpos", 5)))
        print(f"  fold {k + 1}/{F_}: held-out dev macro-F1 {scores(Y[ho].astype(np.uint8), decide(oof[ho], thr, a1))['macro']:.4f}", flush=True)
    plain_ = 1 / (1 + np.exp(-base_d))
    thr0, a10, _ = tune(Y.astype(np.uint8), plain_, int(cfg.lsi.get("minpos", 5)))
    print(f"mean-logit ensemble of {len(ms)} members (same dev, same thresholding): "
          f"{scores(Y.astype(np.uint8), decide(plain_, thr0, a10))['macro']:.4f}")
    out = {"K": K, "params": 0, "trainable": 0, "phase1": "stack", "method": "stack",
           "dev": dict(zip(dv["ids"], oof)), "test": dict(zip(ts["ids"], tst))}
    run = path(f"runs/lsi/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save(out, run / "probs.pt")
    finish(cfg, run, out, dv, ts, K, {"method": "stacked binary relevance over member probabilities (dev cross-fitted)",
                                      "members": [r for r, _ in ms], "folds": F_})


def teach(cfg):
    p = torch.load(path(f"runs/lsi/{cfg.teach.fromrun}") / "probs.pt", map_location="cpu", weights_only=False)
    assert p.get("train"), "the teacher ensemble needs train probabilities (trainprobs: true on its members)"
    out = path(cfg.teach.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tp = dict(p["train"])
    if cfg.teach.get("withdev"):          # soft targets for dev docs too (students trained on train + dev folds)
        tp.update(p["dev"])
    torch.save({"train": tp}, out)
    print(f"ok: teacher probabilities for {len(tp)} docs (train{' + dev' if cfg.teach.get('withdev') else ''}) -> {out}")


def main():
    cfg, modes = args()
    {"download": download, "prep": prep, "train": train, "predict": predict, "ensemble": ensemble,
     "teach": teach, "stack": stack, "devjoin": devjoin}[modes[0] if modes else "prep"](cfg)


if __name__ == "__main__":
    main()
