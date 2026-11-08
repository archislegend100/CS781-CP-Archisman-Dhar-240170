"""Phase 3: rhetorical role prediction (RR) with the Phase-1 legal LMs.

Model (hierarchical sequence labelling, cf. HSLN, Brack et al. 2021; Kalamkar et al. 2022; Malik et al. 2022):
  consecutive sentences of a document are packed into a chunk of <= `window` tokens and encoded together by the
  Phase-1 backbone (causal mask removed, Dukic & Snajder 2024), so every sentence sees its neighbours; a sentence is the
  mean of its token states; a learned relative-position bucket and a domain (CL/IT) embedding are added; a BiGRU over
  the chunk's sentences gives document context; a linear layer gives per-sentence emissions; a linear-chain CRF over
  the role sequence (Lafferty et al. 2001) decodes the whole document. Auxiliary label-shift head (Malik et al. 2022).
  `mode: sent` is the sentence-only ablation (each sentence encoded alone, no position / GRU / CRF).
Long documents: overlapping sentence-aligned chunks; each sentence takes its emission from the chunk where it is
most central; CRF Viterbi then runs over the whole document.

python rr.py phase3/configs/<cfg>.yaml check | train | predict | ensemble | teach
"""
import json
import math
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner
from config import args, loadcfg, path, plain, subdir, wrap
from data import writejson
from generate import autocast
from peft import AdaLoRAController, count
from tokenizer import gettok

DOMAINS = ("CL", "IT")


# ----------------------------------------------------------------------------- data

def readjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def loadvocab(p):
    d = readjson(p)
    if isinstance(d, list):
        return list(d)
    vals = list(d.values())
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]
    return [d[k] for k in sorted(d, key=lambda k: int(k))] if all(str(k).isdigit() for k in d) else vals


def loaddocs(p):
    """{doc_id: (sentences, labels or None)} from a list of docs or a dict keyed by id."""
    d = readjson(p)
    items = d.items() if isinstance(d, dict) and not ("text" in d and "id" in d) else \
        [(x.get("id", str(i)), x) for i, x in enumerate(d if isinstance(d, list) else [d])]
    out = {}
    for k, x in items:
        if isinstance(x, list):          # id -> list of sentences
            x = {"text": x}
        sents = x.get("text", x.get("sentences"))
        if isinstance(sents, str):
            sents = [sents]
        labs = x.get("labels", x.get("label"))
        out[str(x.get("id", k))] = (list(sents), list(labs) if labs is not None else None)
    return out


def files(cfg):
    r = cfg.rr
    return {k: path(r.root) / v for k, v in r.files.items()}


def loadall(cfg):
    f = files(cfg)
    labels = loadvocab(f["labels"])
    data = {}
    for dom in DOMAINS:
        for split in ("train", "dev", "test"):
            key = f"{dom}_{split}"
            if key in f and f[key].exists():
                for d, (s, l) in loaddocs(f[key]).items():
                    data[d] = {"sents": s, "labels": l, "domain": dom, "split": split}
    return labels, data


def macro(gold, pred, labels):
    """sklearn f1_score(labels=labels, average='macro', zero_division=0) over all sentences, as evaluate_phase3.py."""
    st = {l: [0, 0, 0] for l in labels}
    for g, p in zip(gold, pred):
        if g == p:
            st[g][0] += 1
        else:
            st[p][1] += 1
            st[g][2] += 1
    per = {}
    for l, (tp, fp, fn) in st.items():
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[l] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    k = len(labels)
    return {"f1": sum(v[2] for v in per.values()) / k, "p": sum(v[0] for v in per.values()) / k,
            "r": sum(v[1] for v in per.values()) / k, "per": per}


def flat(docs, preds, labels, data):
    g, p = [], []
    for d in docs:
        g += data[d]["labels"]
        p += [labels[i] for i in preds[d]]
    return g, p


# ----------------------------------------------------------------------------- tokenisation and chunks

def encode(tok, data, maxsent):
    """Token ids per sentence (first `maxsent` pieces; an empty sentence becomes one '.' piece)."""
    dot = tok.encode(".")[:1] or [3]
    for x in data.values():
        x["ids"] = [(tok.encode(s)[:maxsent] or dot) for s in x["sents"]]


def chunks(lens, L, overlap):
    """Sentence-aligned chunks [s, e) of at most L tokens; consecutive chunks overlap by about `overlap` of a chunk."""
    n, out, s = len(lens), [], 0
    while s < n:
        e, tot = s, 0
        while e < n and tot + lens[e] <= L:
            tot += lens[e]
            e += 1
        e = max(e, s + 1)
        out.append((s, e))
        if e >= n:
            break
        acc, nxt = 0, s
        while nxt < e - 1 and acc < (1 - overlap) * tot:
            acc += lens[nxt]
            nxt += 1
        s = max(nxt, s + 1)
    return out


def posbucket(i, n, k):
    return min(k - 1, int(k * i / max(1, n)))


# ----------------------------------------------------------------------------- model

class CRF(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(k, k))
        self.start = nn.Parameter(torch.zeros(k))
        self.end = nn.Parameter(torch.zeros(k))

    def nll(self, em, y):
        """Negative log-likelihood of one tag sequence y (S,) under emissions em (S, K)."""
        em = em.float()
        score = self.start[y[0]] + em[torch.arange(len(y), device=em.device), y].sum() + self.end[y[-1]]
        if len(y) > 1:
            score = score + self.trans[y[:-1], y[1:]].sum()
        a = self.start + em[0]
        for t in range(1, em.size(0)):
            a = torch.logsumexp(a[:, None] + self.trans + em[t][None], 0)
        return torch.logsumexp(a + self.end, 0) - score


def viterbi(em, trans, start, end):
    S, K = em.shape
    if S == 0:
        return []
    dp = start + em[0]
    back = np.zeros((S, K), dtype=np.int64)
    for t in range(1, S):
        sc = dp[:, None] + trans
        back[t] = sc.argmax(0)
        dp = sc.max(0) + em[t]
    dp = dp + end
    path = [int(dp.argmax())]
    for t in range(S - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


class RRModel(nn.Module):
    def __init__(self, enc, dim, k, r):
        super().__init__()
        self.enc, self.mode = enc, r.get("mode", "ctx")
        self.enc.cls = nn.Identity()
        ctxm = self.mode == "ctx"
        self.npos = int(r.get("posbuckets", 10)) if ctxm and r.get("pos", True) else 0
        self.pos = nn.Embedding(self.npos, dim) if self.npos else None
        self.dom = nn.Embedding(len(DOMAINS), dim) if r.get("domain", True) else None
        for e in (self.pos, self.dom):
            if e is not None:
                nn.init.normal_(e.weight, 0.0, 0.02)
        self.drop = nn.Dropout(float(r.get("sdrop", 0.2)))
        hid = int(r.get("gru", 256)) if ctxm else 0
        self.gru = nn.GRU(dim, hid, batch_first=True, bidirectional=True) if hid else None
        out = 2 * hid if hid else dim
        self.cls = nn.Linear(out, k)
        self.crf = CRF(k) if ctxm and r.get("crf", True) else None
        self.shift = nn.Linear(2 * out, 1) if ctxm and float(r.get("shift", 0.0)) else None

    def sentvecs(self, ids, spans):
        """ids (1, T) of a packed chunk -> mean token state per sentence span -> (S, D)."""
        h = self.enc.hidden(ids)[0].float()
        seg = torch.repeat_interleave(torch.arange(len(spans), device=h.device),
                                      torch.tensor([e - s for s, e in spans], device=h.device))
        v = torch.zeros(len(spans), h.size(-1), device=h.device).index_add_(0, seg, h)
        return v / torch.tensor([e - s for s, e in spans], device=h.device, dtype=v.dtype)[:, None]

    def forward(self, ids, spans, pos, dom):
        """One chunk -> (sentence features g (S, H), emissions (S, K), sentence vectors v (S, D))."""
        v = self.sentvecs(ids, spans)
        x = v
        if self.pos is not None:
            x = x + self.pos(pos)
        if self.dom is not None:
            x = x + self.dom(dom)
        x = self.drop(x)
        g = self.gru(x[None])[0][0] if self.gru is not None else x
        return g, self.cls(self.drop(g)), v

    def sentonly(self, ids, dom):
        """Sentence-only ablation: a batch of equal-length single sentences (B, T) -> emissions (B, K)."""
        v = self.enc.hidden(ids).float().mean(1)
        if self.dom is not None:
            v = v + self.dom(dom)
        return self.cls(self.drop(v)), v


def build(cfg, dev, labels):
    t, info = ner.make(wrap({**cfg, "ner": cfg.rr}), dev, labels)
    m = RRModel(t, int(info["mcfg"]["dim"]), len(labels), cfg.rr)
    return m.to(dev), info


def loadrr(run, dev):
    obj = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = wrap(obj["cfg"])
    m, _ = build(cfg, dev, obj["labels"])
    m.load_state_dict(obj["model"])
    return m.eval(), obj


# ----------------------------------------------------------------------------- training

def chunkitems(data, docs, L, overlap):
    out = []
    for d in docs:
        lens = [len(i) for i in data[d]["ids"]]
        out += [(d, s, e) for s, e in chunks(lens, L, overlap)]
    return out


def chunkinput(x, s, e, k, dev):
    ids, spans, p = [], [], 0
    for i in range(s, e):
        ids += x["ids"][i]
        spans.append((p, p + len(x["ids"][i])))
        p += len(x["ids"][i])
    n = len(x["ids"])
    pos = torch.tensor([posbucket(i, n, k) for i in range(s, e)], device=dev)
    dom = torch.full((e - s,), DOMAINS.index(x["domain"]), device=dev)
    return torch.tensor([ids], device=dev), spans, pos, dom


def classweights(data, docs, labels, gamma):
    c = Counter(l for d in docs for l in data[d]["labels"])
    n = np.array([max(1, c[l]) for l in labels], dtype=np.float64)
    return torch.tensor(np.minimum((n.max() / n) ** gamma, 50.0), dtype=torch.float32)


def kdload(r, labels):
    if not r.get("kd", {}).get("teacher"):
        return None
    t = torch.load(path(r.kd.teacher), map_location="cpu", weights_only=False)
    assert t["labels"] == labels
    return t


def train(cfg):
    r = cfg.rr
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, data = loadall(cfg)
    tok = gettok(loadcfg(r.backbone.config))
    encode(tok, data, int(r.get("maxsent", 128)))
    li = {l: i for i, l in enumerate(labels)}
    trs = ("train", "dev") if r.get("alldocs") else ("train",)
    tr = [d for d, x in data.items() if x["split"] in trs and x["labels"] is not None]
    va = [] if r.get("alldocs") else [d for d, x in data.items() if x["split"] == "dev" and x["labels"] is not None]
    if r.get("foldk") is not None:      # cross-fitting member: all labelled docs outside fold k, fixed epochs
        fold = cvfolds(data, int(r.get("folds", 5)))
        tr, va = [d for d in fold if fold[d] != int(r.foldk)], []
    cw = classweights(data, tr, labels, float(r.get("cw", 0.5))).to(dev)
    model, info = build(cfg, dev, labels)
    teacher = kdload(r, labels)
    tot, trn = count(model)
    mode, L, K = model.mode, int(r.window), int(r.get("posbuckets", 10))
    if mode == "ctx":
        items = chunkitems(data, tr, L, float(r.get("overlap", 0.5)))
    else:
        items = [(d, i, i + 1) for d in tr for i in range(len(data[d]["ids"]))]
    T, alpha = float(r.get("kd", {}).get("T", 2.0)), float(r.get("kd", {}).get("alpha", 0.5))
    beta = float(r.get("kd", {}).get("hidden", 0.0))
    proj = nn.Linear(int(info["mcfg"]["dim"]), teacher["vdim"]).to(dev) if teacher is not None and beta else None
    hp = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("enc.backbone.")]
    hp += list(proj.parameters()) if proj is not None else []
    bp = [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith("enc.backbone.")]
    opt = torch.optim.AdamW([g for g in ({"params": bp, "lr": float(r.lr)}, {"params": hp, "lr": float(r.headlr)})
                             if g["params"]], weight_decay=float(r.get("wd", 0.01)))
    acc = int(r.get("accum", 8)) if mode == "ctx" else 1
    B = int(r.get("batch", 32))
    nb = math.ceil(len(items) / acc) if mode == "ctx" else math.ceil(len(items) / B) + len({len(data[d]["ids"][i]) for d, i, _ in items})
    steps = nb * int(r.epochs)
    warm = int(float(r.get("warmup", 0.1)) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    ada = AdaLoRAController(model, float(r.get("target", 4)), steps, int(0.1 * steps), int(0.3 * steps)) \
        if r.method == "adalora" else None
    run = path(f"runs/rr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | {info['phase1_model']} from {info['phase1']} | method {r.method} | mode {mode} | "
          f"crf {model.crf is not None} | params {tot / 1e6:.1f}M, trainable {trn / 1e6:.2f}M ({100 * trn / tot:.2f}%)")
    print(f"docs train {len(tr)} dev {len(va)} | {'chunks' if mode == 'ctx' else 'sentences'} {len(items)} | "
          f"class weights {np.round(cw.cpu().numpy(), 2).tolist()}" + (" | KD on" if teacher else ""))
    pr = r.get("precision", "bf16")
    best, bad, log, step = -1.0, 0, open(run / "metrics.jsonl", "a"), 0
    for ep in range(int(r.epochs)):
        model.train()
        t0, tl, k = time.time(), 0.0, 0
        if mode == "ctx":
            rng.shuffle(items)
            groups = [items[i:i + acc] for i in range(0, len(items), acc)]
        else:
            bylen = defaultdict(list)
            for it in items:
                bylen[len(data[it[0]]["ids"][it[1]])].append(it)
            groups = []
            for g in bylen.values():
                rng.shuffle(g)
                groups += [g[i:i + B] for i in range(0, len(g), B)]
            rng.shuffle(groups)
        for group in groups:
            loss_g = 0.0
            if mode == "ctx":
                for d, s, e in group:
                    x = data[d]
                    ids, spans, pos, dom = chunkinput(x, s, e, K, dev)
                    y = torch.tensor([li[l] for l in x["labels"][s:e]], device=dev)
                    with autocast(dev, pr):
                        g, em, v = model(ids, spans, pos, dom)
                    em = em.float()
                    loss = F.cross_entropy(em, y, weight=cw)
                    if model.crf is not None:
                        loss = loss + model.crf.nll(em, y) / len(y)
                    if model.shift is not None and len(y) > 1:
                        z = model.shift(torch.cat([g[1:], g[:-1]], -1).float()).squeeze(-1)
                        loss = loss + float(r.shift) * F.binary_cross_entropy_with_logits(z, (y[1:] != y[:-1]).float())
                    if teacher is not None:
                        tp = torch.from_numpy(teacher["probs"][d][s:e]).to(dev).float()
                        kl = F.kl_div(F.log_softmax(em / T, -1), (tp.clamp_min(1e-9) ** (1 / T)) /
                                      (tp.clamp_min(1e-9) ** (1 / T)).sum(-1, keepdim=True), reduction="batchmean") * T * T
                        loss = alpha * loss + (1 - alpha) * kl
                        if proj is not None:
                            tv = torch.from_numpy(teacher["vecs"][d][s:e]).to(dev).float()
                            loss = loss + beta * F.mse_loss(F.normalize(proj(v), dim=-1), F.normalize(tv, dim=-1))
                    (loss / len(group)).backward()
                    loss_g += float(loss.detach()) / len(group)
            else:
                ids = torch.tensor([data[d]["ids"][i] for d, i, _ in group], device=dev)
                dom = torch.tensor([DOMAINS.index(data[d]["domain"]) for d, _, _ in group], device=dev)
                y = torch.tensor([li[data[d]["labels"][i]] for d, i, _ in group], device=dev)
                with autocast(dev, pr):
                    em, v = model.sentonly(ids, dom)
                loss = F.cross_entropy(em.float(), y, weight=cw)
                loss.backward()
                loss_g = float(loss.detach())
            if ada is not None:
                (float(r.get("orth", 0.1)) * ada.penalty()).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(r.get("clip", 1.0)))
            if ada is not None:
                ada.update(step)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tl, k = tl + loss_g, k + 1
        msg = f"epoch {ep + 1} | loss {tl / max(1, k):.4f}"
        if ada is not None:
            rk = ada.ranks()
            msg += f" | adalora mean rank {sum(rk) / max(1, len(rk)):.2f}"
        if not va:
            print(f"{msg} | all docs (fixed {r.epochs} epochs) | {time.time() - t0:.0f}s", flush=True)
            if ep + 1 == int(r.epochs):
                save(run, model, cfg, labels, info, tot, trn, ep + 1, None)
            continue
        em = emissions(model, data, va, cfg, dev)
        m, dm = score(em, model, data, va, labels)
        print(f"{msg} | dev macro-F1 {m['f1']:.4f} (CL {dm['CL']:.4f}, IT {dm['IT']:.4f}) P {m['p']:.4f} R {m['r']:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)
        log.write(json.dumps({"epoch": ep + 1, "loss": tl / max(1, k), "dev": m["f1"], "CL": dm["CL"], "IT": dm["IT"]}) + "\n")
        log.flush()
        if m["f1"] > best:
            best, bad = m["f1"], 0
            save(run, model, cfg, labels, info, tot, trn, ep + 1, {"f1": m["f1"], "p": m["p"], "r": m["r"], **dm,
                                                                    "per": {l: v[2] for l, v in m["per"].items()}})
        else:
            bad += 1
            if bad >= int(r.get("patience", 4)):
                print(f"early stop after epoch {ep + 1}")
                break
    log.close()
    print(f"best dev macro-F1 {best:.4f}" if va else f"trained on {len(tr)} labelled docs for {r.epochs} epochs")


def save(run, model, cfg, labels, info, tot, trn, ep, dev):
    torch.save({"model": model.state_dict(), "cfg": plain(cfg), "labels": labels, "epoch": ep, "dev": dev,
                "info": {k: v for k, v in info.items() if k != "mcfg"}, "params": tot, "trainable": trn}, run / "best.pt")
    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": cfg.rr.method, "mode": cfg.rr.get("mode", "ctx"),
                                  "epoch": ep, "dev": dev, "params": tot, "trainable": trn,
                                  **{k: v for k, v in info.items() if k != "mcfg"}}, 1)


# ----------------------------------------------------------------------------- prediction

@torch.no_grad()
def emissions(model, data, docs, cfg, dev, vecs=False):
    """{doc: (S, K) log-softmax emissions} (+ sentence vectors); each sentence from its most central chunk."""
    r = cfg.rr
    was = model.training
    model.eval()
    pr, L, K = r.get("precision", "bf16"), int(r.window), int(r.get("posbuckets", 10))
    out, vout = {}, {}
    for d in docs:
        x = data[d]
        n = len(x["ids"])
        em = np.zeros((n, model.cls.out_features), dtype=np.float32)
        vv = None
        if model.mode == "ctx":
            cen = np.full(n, -1.0)
            for s, e in chunks([len(i) for i in x["ids"]], L, float(r.get("overlap", 0.5))):
                ids, spans, pos, dom = chunkinput(x, s, e, K, dev)
                with autocast(dev, pr):
                    _, o, v = model(ids, spans, pos, dom)
                o = F.log_softmax(o.float(), -1).cpu().numpy()
                v = v.float().cpu().numpy()
                if vv is None:
                    vv = np.zeros((n, v.shape[1]), dtype=np.float32)
                for j, i in enumerate(range(s, e)):
                    # distance to the nearest chunk edge that is not also a document edge
                    c = min(i - s if s > 0 else n, e - 1 - i if e < n else n)
                    if c > cen[i]:
                        cen[i], em[i], vv[i] = c, o[j], v[j]
        else:
            byl = defaultdict(list)
            for i, t in enumerate(x["ids"]):
                byl[len(t)].append(i)
            for idx in byl.values():
                for a in range(0, len(idx), 64):
                    sel = idx[a:a + 64]
                    ids = torch.tensor([x["ids"][i] for i in sel], device=dev)
                    dom = torch.full((len(sel),), DOMAINS.index(x["domain"]), device=dev)
                    with autocast(dev, pr):
                        o, v = model.sentonly(ids, dom)
                    o = F.log_softmax(o.float(), -1).cpu().numpy()
                    v = v.float().cpu().numpy()
                    if vv is None:
                        vv = np.zeros((n, v.shape[1]), dtype=np.float32)
                    em[sel], vv[sel] = o, v
        out[d] = em
        vout[d] = vv
    model.train(was)
    return (out, vout) if vecs else out


def transitions(model):
    if model.crf is None:
        return None
    return {k: getattr(model.crf, k).detach().float().cpu().numpy() for k in ("trans", "start", "end")}


def decode(em, crf, bias=None, tscale=1.0):
    e = em if bias is None else em + bias
    if crf is None:
        return [int(i) for i in e.argmax(1)]
    return viterbi(e, tscale * crf["trans"], tscale * crf["start"], tscale * crf["end"])


def score(em, model, data, docs, labels, crf="auto", bias=None):
    crf = transitions(model) if crf == "auto" else crf
    preds = {d: decode(em[d], crf, bias) for d in docs}
    g, p = flat(docs, preds, labels, data)
    dm = {}
    for dom in DOMAINS:
        dd = [d for d in docs if data[d]["domain"] == dom]
        if dd:
            dm[dom] = macro(*flat(dd, preds, labels, data), labels)["f1"]
        else:
            dm[dom] = float("nan")
    return macro(g, p, labels), dm


def writesub(cfg, labels, data, preds, meta):
    test = [d for d, x in data.items() if x["split"] == "test"]
    out = {d: [labels[i] for i in preds[d]] for d in test}
    for d in test:
        assert len(out[d]) == len(data[d]["sents"]), d
        assert all(l in labels for l in out[d])
    sd = subdir(cfg)
    writejson(sd / "prediction.json", out, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, **meta,
                                 "test_docs": len(out), "test_sentences": sum(len(v) for v in out.values()),
                                 "label_counts": Counter(l for v in out.values() for l in v)}, 1)
    print(f"ok: {len(out)} test docs, {sum(len(v) for v in out.values())} sentences -> {sd / 'submission.zip'}")


def officialdev(cfg, run, labels, data, va, preds):
    """Run the official evaluate_phase3.py on the dev predictions (checks format + our macro-F1 implementation)."""
    import subprocess
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase3.py"))
    if not ev.exists():
        return
    for dom in DOMAINS:
        writejson(run / f"dev_truth_{dom}.json", {d: data[d]["labels"] for d in va if data[d]["domain"] == dom})
    writejson(run / "dev_pred.json", {d: [labels[i] for i in preds[d]] for d in va})
    writejson(run / "labels.json", labels)
    r = subprocess.run([sys.executable, str(ev), "--truth-cl", str(run / "dev_truth_CL.json"), "--truth-it",
                        str(run / "dev_truth_IT.json"), "--prediction", str(run / "dev_pred.json"), "--labels",
                        str(run / "labels.json")], capture_output=True, text=True)
    print("official evaluator on dev: " + " | ".join(x.strip() for x in (r.stdout + r.stderr).splitlines() if x.strip()))


def predict(cfg):
    """Emissions for train/dev/test (saved for ensembling and distillation) + the single-model submission."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = path(f"runs/rr/{cfg.name}")
    model, obj = loadrr(run, dev)
    labels, data = loadall(cfg)
    encode(gettok(loadcfg(cfg.rr.backbone.config)), data, int(cfg.rr.get("maxsent", 128)))
    em, vec = emissions(model, data, list(data), cfg, dev, vecs=True)
    crf = transitions(model)
    torch.save({"em": em, "vecs": vec, "crf": crf, "labels": labels, "params": obj["params"],
                "trainable": obj["trainable"], "phase1": obj["info"]["phase1"], "method": cfg.rr.method}, run / "emissions.pt")
    va = [d for d, x in data.items() if x["split"] == "dev" and x["labels"] is not None]
    if va:
        m, dm = score(em, model, data, va, labels)
        print(f"dev macro-F1 {m['f1']:.4f} (CL {dm['CL']:.4f}, IT {dm['IT']:.4f})" +
              ("" if not cfg.rr.get("alldocs") else "  [trained on dev too: optimistic]"))
        print("per label: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in m["per"].items()))
        officialdev(cfg, run, labels, data, va, {d: decode(em[d], crf) for d in va})
    test = [d for d, x in data.items() if x["split"] == "test"]
    writesub(cfg, labels, data, {d: decode(em[d], crf) for d in test},
             {"method": cfg.rr.method, "mode": cfg.rr.get("mode", "ctx"), "phase1": obj["info"]["phase1"],
              "params": obj["params"], "trainable": obj["trainable"], "epoch": obj["epoch"], "dev": obj["dev"]})


# ----------------------------------------------------------------------------- ensemble, calibration, teacher

def loademis(runs):
    out = []
    for r in runs:
        p = path(r) / "emissions.pt"
        if not p.exists():
            print(f"skip {r}: no emissions.pt (run predict first)")
            continue
        out.append((r, torch.load(p, map_location="cpu", weights_only=False)))
    return out


def combine(members, docs, weights=None):
    w = weights or [1.0] * len(members)
    em = {d: np.log(sum(wi * np.exp(m["em"][d]) for wi, (_, m) in zip(w, members)) / sum(w) + 1e-9) for d in docs}
    crfs = [m["crf"] for _, m in members if m["crf"] is not None]
    crf = None if len(crfs) < len(members) else {k: np.mean([c[k] for c in crfs], 0) for k in ("trans", "start", "end")}
    return em, crf


def ensemble(cfg):
    """Mean of member probabilities; CRF transitions averaged (when all members have one); per-label decision biases
    (Lipton et al. 2014: thresholds for macro-F1) and the transition scale chosen on the official dev set."""
    e = cfg.ens
    labels, data = loadall(cfg)
    members = loademis(e.runs)
    assert members, "no members"
    va = [d for d, x in data.items() if x["split"] == "dev" and x["labels"] is not None]
    test = [d for d, x in data.items() if x["split"] == "test"]
    K = len(labels)
    gold = [l for d in va for l in data[d]["labels"]]
    w = [1.0] * len(members)
    if e.get("biasfrom") and len(readjson(path(e.biasfrom)).get("weights", [])) == len(members):
        w = readjson(path(e.biasfrom))["weights"]
    elif e.get("tuneweights"):
        # greedy coordinate search over member weights on dev (no biases yet)
        def wf1(ws):
            em_, crf_ = combine(members, va, ws)
            return macro(gold, [labels[i] for d in va for i in decode(em_[d], crf_)], labels)["f1"]
        best = wf1(w)
        for _ in range(2):
            for j in range(len(members)):
                for v in e.get("wgrid", [0.0, 0.5, 2.0, 3.0]):
                    trial = list(w)
                    trial[j] = v
                    if sum(trial) == 0:
                        continue
                    f = wf1(trial)
                    if f > best + 1e-4:
                        best, w = f, trial
        print(f"member weights {dict(zip([Path(r).name for r, _ in members], w))}: dev {best:.4f}")
    em, crf = combine(members, list(data), w)

    def f1(bias, ts):
        preds = {d: decode(em[d], crf, bias, ts) for d in va}
        return macro(gold, [labels[i] for d in va for i in preds[d]], labels)

    if e.get("biasfrom"):
        mm = readjson(path(e.biasfrom))
        bias, ts = np.array([mm["bias"][l] for l in labels], dtype=np.float32), float(mm["tscale"])
        print(f"biases and transition scale from {e.biasfrom}")
    else:
        bias, ts = np.zeros(K, dtype=np.float32), 1.0
        base = f1(bias, ts)["f1"]
        print(f"ensemble of {len(members)}: dev macro-F1 {base:.4f}" + ("" if crf is not None else " (no CRF: argmax)"))
        best = base
        if crf is not None:
            for v in e.get("tgrid", [0.0, 0.5, 1.5, 2.0]):
                f = f1(bias, v)["f1"]
                if f > best + 1e-4:
                    best, ts = f, v
            print(f"  transition scale {ts}: {best:.4f}")
        for _ in range(int(e.get("passes", 2))):
            for k in range(K):
                for v in sorted(e.get("grid", [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]), key=abs):
                    trial = bias.copy()
                    trial[k] = v
                    f = f1(trial, ts)["f1"]
                    if f > best + 1e-4:
                        best, bias = f, trial
            print(f"  biases {dict(zip(labels, np.round(bias, 2).tolist()))}: {best:.4f}")
    final = f1(bias, ts)
    print("dev per label: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in final["per"].items()))
    frozen = {}
    for _, m in members:
        if m["method"] == "frozen":
            frozen.update(m.get("backbones", {}))
        elif m["method"] not in ("full", "kd", "partial"):
            frozen[m["phase1"]] = m["params"] - m["trainable"]
    size = sum(m["params"] if m["method"] in ("full", "kd", "partial") else m["trainable"] for _, m in members) + \
        sum(frozen.values())
    writesub(cfg, labels, data, {d: decode(em[d], crf, bias, ts) for d in test},
             {"method": "ensemble (mean probabilities, averaged CRF) + dev-tuned per-label biases",
              "members": [r for r, _ in members], "weights": list(w), "bias": dict(zip(labels, bias.tolist())), "tscale": ts,
              "dev_f1": final["f1"], "dev_per_label": {l: v[2] for l, v in final["per"].items()},
              "params_deployed": size})


def teach(cfg):
    """Distillation targets: the ensemble's per-sentence probabilities (and the first member's sentence vectors) on
    every document, saved for the student."""
    e = cfg.ens
    labels, data = loadall(cfg)
    members = loademis(e.runs)
    em, _ = combine(members, list(data))
    probs = {d: np.exp(v).astype(np.float32) for d, v in em.items()}
    vecs = members[0][1]["vecs"]
    out = path(cfg.teach.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"probs": probs, "vecs": vecs, "vdim": int(next(iter(vecs.values())).shape[1]), "labels": labels,
                "members": [r for r, _ in members]}, out)
    print(f"ok: teacher of {len(members)} members -> {out}")


# ----------------------------------------------------------------------------- frozen features + document BiLSTM-CRF

@torch.no_grad()
def feats(cfg):
    """Frozen Phase-1 sentence features: mean token state of each sentence at several depths (final norm applied),
    concatenated. Causal backbones run causally as pretrained, each sentence taken from the chunk that gives it the most
    left context; the bidirectional LegalSeq encoder uses the most central chunk."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, data = loadall(cfg)
    q = cfg.feats
    out = path(q.out)
    out.mkdir(parents=True, exist_ok=True)
    for b in q.backbones:
        f = out / f"{b.name}.pt"
        if f.exists():
            print(f"skip {b.name}: {f} exists")
            continue
        if not path(b.run).exists():
            print(f"skip {b.name}: {b.run} not found")
            continue
        rr_ = wrap({**plain(cfg.rr), "backbone": {"config": b.config, "run": b.run, "ckpt": b.ckpt}, "method": "full",
                    "unmask": "none", "layers": None, "peftlayers": None, "mix": False})
        t, info = ner.make(wrap({**cfg, "ner": rr_}), dev, labels)
        bb = t.backbone.eval()
        seq2seq = t.family == "seq2seq"
        tok = gettok(loadcfg(b.config))
        encode(tok, data, int(q.get("maxsent", 128)))
        L, nl = int(b.get("window", 1024)), len(bb.enc) if seq2seq else len(bb.blocks)
        depth = sorted({max(1, round(fr * nl)) for fr in q.get("depths", [0.5, 0.75, 1.0])})
        nrm = bb.encnorm if seq2seq else bb.norm
        res = {}
        t0 = time.time()
        for d, x in data.items():
            n = len(x["ids"])
            fv, best = None, np.full(n, -1.0)
            for s, e in chunks([len(i) for i in x["ids"]], L, 0.5):
                ids, spans, _, _ = chunkinput(x, s, e, 10, dev)
                with autocast(dev, q.get("precision", "bf16")):
                    hs = bb.encode(ids, torch.ones_like(ids, dtype=torch.bool), hiddens=True)[1] if seq2seq \
                        else bb(ids, last=True, hiddens=True)["hiddens"]
                lens = torch.tensor([e_ - s_ for s_, e_ in spans], device=dev)
                seg = torch.repeat_interleave(torch.arange(len(spans), device=dev), lens)
                parts = []
                for k in depth:
                    h = nrm(hs[k])[0].float()
                    parts.append(torch.zeros(len(spans), h.size(-1), device=dev).index_add_(0, seg, h) / lens[:, None])
                v = torch.cat(parts, -1).cpu().numpy()
                if fv is None:
                    fv = np.zeros((n, v.shape[1]), dtype=np.float32)
                for j, i in enumerate(range(s, e)):
                    c = min(i - s if s > 0 else n, e - 1 - i if e < n else n) if seq2seq else (i - s)
                    if c > best[i]:
                        best[i], fv[i] = c, v[j]
            res[d] = fv.astype(np.float16)
        params = sum(p.numel() for p in bb.parameters())
        torch.save({"feats": res, "dim": int(next(iter(res.values())).shape[1]), "depths": depth, "params": params,
                    "phase1": f"{b.run}:{b.ckpt}", "model": info["phase1_model"]}, f)
        print(f"ok {b.name} ({info['phase1_model']}, {params / 1e6:.1f}M, layers {depth} of {nl}): "
              f"{sum(len(v) for v in res.values())} sentences x {next(iter(res.values())).shape[1]} -> {f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del t, bb
        torch.cuda.empty_cache()


class DocTagger(nn.Module):
    """Whole-document hierarchical tagger over frozen sentence features (cf. Bhattacharya et al. 2019 Hier-BiLSTM-CRF;
    HSLN): feature dropout -> projection -> + position bucket + domain -> BiLSTM -> emissions -> CRF; shift head."""

    def __init__(self, fin, k, q):
        super().__init__()
        p, h = int(q.get("proj", 512)), int(q.get("hid", 256))
        self.fdrop = nn.Dropout(float(q.get("fdrop", 0.5)))
        self.proj = nn.Sequential(nn.Linear(fin, p), nn.LayerNorm(p), nn.GELU())
        self.npos = int(q.get("posbuckets", 20))
        self.pos = nn.Embedding(self.npos, p)
        self.dom = nn.Embedding(len(DOMAINS), p)
        self.lstm = nn.LSTM(p, h, num_layers=int(q.get("layers", 2)), batch_first=True, bidirectional=True,
                            dropout=float(q.get("sdrop", 0.3)))
        self.drop = nn.Dropout(float(q.get("sdrop", 0.3)))
        self.cls = nn.Linear(2 * h, k)
        self.crf = CRF(k) if q.get("crf", True) else None
        self.shift = nn.Linear(4 * h, 1) if float(q.get("shift", 0.3)) else None
        V = int(q.get("vocab", 0))
        self.bag = nn.EmbeddingBag(V, p, mode="sum") if V else None
        if self.bag is not None:
            nn.init.normal_(self.bag.weight, 0.0, 0.02)
        self.wdrop = float(q.get("wdrop", 0.2))

    def forward(self, x, dom, sp=None):
        n = x.size(0)
        pos = torch.tensor([posbucket(i, n, self.npos) for i in range(n)], device=x.device)
        z = self.proj(self.fdrop(x)) + self.pos(pos) + self.dom(torch.full((n,), dom, device=x.device))
        if self.bag is not None and sp is not None:
            idx, w, off = sp
            if self.training and self.wdrop:      # word dropout on the sparse TF-IDF channel
                w = w * (torch.rand_like(w) > self.wdrop).float() / (1 - self.wdrop)
            z = z + self.bag(idx, off, per_sample_weights=w)
        g = self.lstm(self.drop(z)[None])[0][0]
        return g, self.cls(self.drop(g))


def seq(cfg):
    """Train the document-level BiLSTM-CRF on concatenated frozen features for several seeds; average the seeds'
    probabilities (and CRF transitions); save emissions in the same format as `predict` + a submission."""
    q, r = cfg.seq, cfg.rr
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, data = loadall(cfg)
    li = {l: i for i, l in enumerate(labels)}
    have = [n for n in q.feats if (path(cfg.feats.out) / f"{n}.pt").exists()]
    if len(have) < len(q.feats):
        print(f"missing feature files (skipped): {sorted(set(q.feats) - set(have))}")
    assert have, "no feature files: run `rr.py phase3/configs/feats.yaml feats` first"
    q.feats = have
    fs = [torch.load(path(cfg.feats.out) / f"{n}.pt", map_location="cpu", weights_only=False) for n in have]
    X = {d: np.concatenate([f["feats"][d].astype(np.float32) for f in fs], 1) for d in data}
    trs = ("train", "dev") if q.get("alldocs") else ("train",)
    tr = [d for d, x in data.items() if x["split"] in trs and x["labels"] is not None]
    va = [] if q.get("alldocs") else [d for d, x in data.items() if x["split"] == "dev" and x["labels"] is not None]
    allx = np.concatenate([X[d] for d in tr])
    mu, sd = allx.mean(0), allx.std(0) + 1e-5
    X = {d: torch.tensor((v - mu) / sd, dtype=torch.float32) for d, v in X.items()}
    Y = {d: torch.tensor([li[l] for l in data[d]["labels"]]) for d in tr + va}
    cw = classweights(data, tr, labels, float(q.get("cw", 0.5))).to(dev)
    fin = next(iter(X.values())).shape[1]
    run = path(f"runs/rr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | frozen features {list(q.feats)} ({fin} dims) | docs train {len(tr)} dev {len(va)} | "
          f"seeds {list(q.seeds)}")
    seeds_em, crfs, epochs, devs = [], [], [], []
    for seed in q.seeds:
        torch.manual_seed(int(seed))
        rng = random.Random(int(seed))
        m = DocTagger(fin, len(labels), q).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=float(q.get("lr", 1e-3)), weight_decay=float(q.get("wd", 0.01)))
        acc = int(q.get("accum", 4))
        best, bad, bstate, bep = -1.0, 0, None, 0
        E = int(q.get("epochs", 80))
        for ep in range(E):
            m.train()
            order = list(tr)
            rng.shuffle(order)
            for gi in range(0, len(order), acc):
                for d in order[gi:gi + acc]:
                    x, y = X[d].to(dev), Y[d].to(dev)
                    g, em = m(x, DOMAINS.index(data[d]["domain"]))
                    loss = F.cross_entropy(em, y, weight=cw)
                    if m.crf is not None:
                        loss = loss + m.crf.nll(em, y) / len(y)
                    if m.shift is not None and len(y) > 1:
                        z = m.shift(torch.cat([g[1:], g[:-1]], -1)).squeeze(-1)
                        loss = loss + float(q.get("shift", 0.3)) * F.binary_cross_entropy_with_logits(z, (y[1:] != y[:-1]).float())
                    (loss / acc).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if not va:
                continue
            em = docem(m, X, data, va, dev)
            crf = None if m.crf is None else {k: getattr(m.crf, k).detach().cpu().numpy() for k in ("trans", "start", "end")}
            f = macro([l for d in va for l in data[d]["labels"]], [labels[i] for d in va for i in decode(em[d], crf)], labels)["f1"]
            if f > best:
                best, bad, bep = f, 0, ep + 1
                bstate = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= int(q.get("patience", 12)):
                    break
        if va:
            m.load_state_dict(bstate)
            print(f"  seed {seed}: best dev macro-F1 {best:.4f} at epoch {bep}", flush=True)
            devs.append(best)
            epochs.append(bep)
        seeds_em.append(docem(m, X, data, list(data), dev))
        if m.crf is not None:
            crfs.append({k: getattr(m.crf, k).detach().cpu().numpy() for k in ("trans", "start", "end")})
    em = {d: np.log(np.mean([np.exp(s_[d]) for s_ in seeds_em], 0) + 1e-9).astype(np.float32) for d in data}
    crf = {k: np.mean([c[k] for c in crfs], 0) for k in ("trans", "start", "end")} if crfs else None
    head = sum(p.numel() for p in m.parameters())
    bbs = {f["phase1"]: f["params"] for f in fs}
    torch.save({"em": em, "vecs": None, "crf": crf, "labels": labels, "params": head * len(q.seeds) + sum(bbs.values()),
                "trainable": head * len(q.seeds), "phase1": ",".join(bbs), "backbones": bbs, "method": "frozen"},
               run / "emissions.pt")
    info = {"seeds": list(q.seeds), "seed_dev": devs, "seed_epochs": epochs, "feats": list(q.feats)}
    if va:
        preds = {d: decode(em[d], crf) for d in va}
        mm = macro([l for d in va for l in data[d]["labels"]], [labels[i] for d in va for i in preds[d]], labels)
        dm = {dom: macro(*flat([d for d in va if data[d]["domain"] == dom], preds, labels, data), labels)["f1"] for dom in DOMAINS}
        print(f"seed-averaged dev macro-F1 {mm['f1']:.4f} (CL {dm['CL']:.4f}, IT {dm['IT']:.4f}) | epochs {epochs}")
        print("per label: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in mm["per"].items()))
        officialdev(cfg, run, labels, data, va, preds)
        info.update({"dev": {"f1": mm["f1"], **dm, "per": {l: v[2] for l, v in mm["per"].items()}},
                     "epoch": int(np.median(epochs))})
    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": "frozen", "mode": "doc",
                                  "params": head * len(q.seeds) + sum(bbs.values()), "trainable": head * len(q.seeds),
                                  "phase1_model": "+".join(f["model"] for f in fs), **info}, 1)
    test = [d for d, x in data.items() if x["split"] == "test"]
    writesub(cfg, labels, data, {d: decode(em[d], crf) for d in test},
             {"method": "frozen Phase-1 features + document BiLSTM-CRF (seed bag)", **info,
              "params_deployed": head * len(q.seeds) + sum(bbs.values())})


@torch.no_grad()
def docem(m, X, data, docs, dev, sp=None):
    m.eval()
    out = {d: F.log_softmax(m(X[d].to(dev), DOMAINS.index(data[d]["domain"]), spto(sp, d, dev))[1], -1).cpu().numpy()
           for d in docs}
    m.train()
    return out


# ----------------------------------------------------------------------------- cross-fitted document tagger

def cvfolds(data, K):
    """Domain-stratified K-fold assignment of all labelled docs (train + dev); identical in every mode."""
    lab = sorted(d for d, x in data.items() if x["labels"] is not None)
    rng = random.Random(0)
    fold = {}
    for dom in DOMAINS:
        dd = [d for d in lab if data[d]["domain"] == dom]
        rng.shuffle(dd)
        for i, d in enumerate(dd):
            fold[d] = i % K
    return fold


def lexfeats(cfg, data, dim):
    """Lexical view: TF-IDF of word uni+bigrams over every official sentence (unsupervised, no labels), reduced to
    `dim` dimensions by truncated SVD (LSA, Deerwester et al. 1990). Cached in the features directory."""
    f = path(cfg.feats.out) / f"lex{dim}.pt"
    if f.exists():
        return torch.load(f, map_location="cpu", weights_only=False)["feats"]
    import re as _re
    docs = sorted(data)
    sents = [s for d in docs for s in data[d]["sents"]]
    toks = []
    for s_ in sents:
        w = _re.findall(r"\w+", s_.lower())
        toks.append(w + [a_ + " " + b_ for a_, b_ in zip(w, w[1:])])
    df = Counter(t for ts in toks for t in set(ts))
    vocab = {t: i for i, (t, c) in enumerate(sorted(((t, c) for t, c in df.items() if c >= 3), key=lambda x: -x[1])[:50000])}
    n = len(sents)
    idf = np.array([0.0] * len(vocab), dtype=np.float32)
    for t, i in vocab.items():
        idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0           # smooth idf
    rows, cols, vals = [], [], []
    for r_, ts in enumerate(toks):
        c = Counter(t for t in ts if t in vocab)
        if not c:
            continue
        v = np.array([(1 + math.log(k_)) * idf[vocab[t]] for t, k_ in c.items()], dtype=np.float32)   # sublinear tf
        v /= np.linalg.norm(v) + 1e-12
        rows += [r_] * len(c)
        cols += [vocab[t] for t in c]
        vals += v.tolist()
    A = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float32), (n, len(vocab))).coalesce()
    # randomised truncated SVD (Halko, Martinsson, Tropp 2011): Z = U_k S_k
    g = torch.Generator().manual_seed(0)
    Q = torch.linalg.qr(torch.sparse.mm(A, torch.randn(len(vocab), dim + 16, generator=g)))[0]
    At = A.t().coalesce()
    for _ in range(3):
        Q = torch.linalg.qr(torch.sparse.mm(A, torch.linalg.qr(torch.sparse.mm(At, Q))[0]))[0]
    Bt = torch.sparse.mm(At, Q)                                   # (V, k) = (Q^T A)^T
    Ub, S, _ = torch.linalg.svd(Bt.T, full_matrices=False)
    Z = (Q @ Ub[:, :dim] * S[:dim]).numpy().astype(np.float32)
    print(f"lexical LSA: {n} sentences, {len(vocab)} terms (df >= 3, uni+bigrams) -> {dim} dims")
    out, k = {}, 0
    for d in docs:
        n = len(data[d]["sents"])
        out[d] = Z[k:k + n]
        k += n
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": out, "dim": dim}, f)
    print(f"cached -> {f}")
    return out


def sparsefeats(data, mindf=2):
    """Full sparse lexical channel: per sentence, L2-normalised sublinear TF-IDF over word unigrams + bigrams
    (df >= mindf over all official sentences, no labels), as EmbeddingBag inputs (indices, weights, offsets) per doc."""
    import re as _re
    docs = sorted(data)
    toks = {}
    for d in docs:
        toks[d] = []
        for s_ in data[d]["sents"]:
            w = _re.findall(r"\w+", s_.lower())
            toks[d].append(w + [a_ + " " + b_ for a_, b_ in zip(w, w[1:])])
    n = sum(len(v) for v in toks.values())
    df = Counter(t for d in docs for ts in toks[d] for t in set(ts))
    vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= mindf))}
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab}
    out = {}
    for d in docs:
        idx, val, off = [], [], []
        for ts in toks[d]:
            off.append(len(idx))
            c = Counter(t for t in ts if t in vocab)
            v = np.array([(1 + math.log(k_)) * idf[t] for t, k_ in c.items()], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12) if len(v) else 1.0
            idx += [vocab[t] for t in c]
            val += v.tolist()
        out[d] = (torch.tensor(idx, dtype=torch.long), torch.tensor(val, dtype=torch.float32), torch.tensor(off, dtype=torch.long))
    return out, len(vocab)


def spto(sp, d, dev):
    if sp is None:
        return None
    return tuple(t.to(dev) for t in sp[d])


def fitdoc(X, Y, data, docs, q, seed, dev, cw, epochs, sp=None):
    torch.manual_seed(int(seed))
    rng = random.Random(int(seed))
    m = DocTagger(next(iter(X.values())).shape[1], cw.numel(), q).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=float(q.get("lr", 1e-3)), weight_decay=float(q.get("wd", 0.01)))
    acc = int(q.get("accum", 4))
    for ep in range(epochs):
        m.train()
        order = list(docs)
        rng.shuffle(order)
        for gi in range(0, len(order), acc):
            for d in order[gi:gi + acc]:
                x, y = X[d].to(dev), Y[d].to(dev)
                g, em = m(x, DOMAINS.index(data[d]["domain"]), spto(sp, d, dev))
                loss = F.cross_entropy(em, y, weight=cw)
                if m.crf is not None:
                    loss = loss + m.crf.nll(em, y) / len(y)
                if m.shift is not None and len(y) > 1:
                    z = m.shift(torch.cat([g[1:], g[:-1]], -1)).squeeze(-1)
                    loss = loss + float(q.get("shift", 0.3)) * F.binary_cross_entropy_with_logits(z, (y[1:] != y[:-1]).float())
                (loss / acc).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    return m


def calibrate(em, crf, data, docs, labels, q, perdomain):
    """Transition scale + per-label (optionally per-domain) decision biases maximising macro-F1 on `docs`."""
    K = len(labels)
    gold = [l for d in docs for l in data[d]["labels"]]
    keys = list(DOMAINS) if perdomain else ["all"]
    bias = {k: np.zeros(K, dtype=np.float32) for k in keys}

    def f1(bs, ts):
        p = []
        for d in docs:
            p += [labels[i] for i in decode(em[d], crf, bs[data[d]["domain"] if perdomain else "all"], ts)]
        return macro(gold, p, labels)

    ts = 1.0
    best = f1(bias, ts)["f1"]
    hist = [("raw", best)]
    if crf is not None:
        for v in q.get("tgrid", [0.0, 0.25, 0.5, 0.75, 1.5]):
            f = f1(bias, v)["f1"]
            if f > best + 1e-4:
                best, ts = f, v
        hist.append((f"tscale {ts}", best))
    for _ in range(int(q.get("passes", 2))):
        for key in keys:
            for k in range(K):
                for v in sorted(q.get("grid", [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]), key=abs):
                    trial = {kk: vv.copy() for kk, vv in bias.items()}
                    trial[key][k] = v
                    f = f1(trial, ts)["f1"]
                    if f > best + 1e-4:
                        best, bias = f, trial
    hist.append(("bias", best))
    return bias, ts, f1(bias, ts), hist


def cv(cfg):
    """Cross-fitted document tagger: K folds over ALL labelled documents (train + dev), several seeds per fold.
    Every labelled document gets an out-of-fold prediction; the CRF transition scale and per-label biases are tuned
    on all of them (rare labels absent from dev are covered by train); test = average of all fold x seed models."""
    q = cfg.seq
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, data = loadall(cfg)
    li = {l: i for i, l in enumerate(labels)}
    have = [n for n in q.feats if (path(cfg.feats.out) / f"{n}.pt").exists()]
    if len(have) < len(q.feats):
        print(f"missing feature files (skipped): {sorted(set(q.feats) - set(have))}")
    fs = [torch.load(path(cfg.feats.out) / f"{n}.pt", map_location="cpu", weights_only=False) for n in have]
    views = [{d: f["feats"][d].astype(np.float32) for d in data} for f in fs]
    if int(q.get("pca", 0)):
        # unsupervised per-view PCA (all sentences, no labels): each frozen view -> `pca` dims, so the views have equal
        # weight and the input is ~8x narrower for a training set of only ~70 documents
        k, order = int(q.pca), sorted(data)
        for vi, v in enumerate(views):
            M = torch.tensor(np.concatenate([v[d] for d in order]))
            mu_ = M.mean(0)
            _, _, V = torch.pca_lowrank(M - mu_, q=min(k, M.shape[1]), center=False, niter=4)
            Z = ((M - mu_) @ V).numpy()
            out, j = {}, 0
            for d in order:
                out[d] = Z[j:j + len(v[d])]
                j += len(v[d])
            views[vi] = out
        print(f"per-view PCA: {len(views)} views -> {k} dims each")
    if int(q.get("lex", 0)):
        views.append(lexfeats(cfg, data, int(q.lex)))
    lab = sorted(d for d, x in data.items() if x["labels"] is not None)
    test = [d for d, x in data.items() if x["split"] == "test"]
    sp = None
    if q.get("sparse"):
        sp, V = sparsefeats(data, int(q.get("mindf", 2)))
        q.vocab = V
        print(f"sparse lexical channel: {V} word uni+bigram features (df >= {int(q.get('mindf', 2))})")
    X = {d: np.concatenate([v[d] for v in views], 1) for d in data}
    allx = np.concatenate([X[d] for d in lab])
    mu, sd = allx.mean(0), allx.std(0) + 1e-5
    X = {d: torch.tensor((v - mu) / sd, dtype=torch.float32) for d, v in X.items()}
    Y = {d: torch.tensor([li[l] for l in data[d]["labels"]]) for d in lab}
    K = int(q.get("folds", 5))
    fold = cvfolds(data, K)
    E = int(q.get("cvepochs", 20))
    run = path(f"runs/rr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    cnt = Counter(l for d in lab for l in data[d]["labels"])
    print(f"{cfg.name} | views {have + (['lex'] if int(q.get('lex', 0)) else [])} ({X[lab[0]].shape[1]} dims) | "
          f"{len(lab)} labelled docs (train + dev), {K} folds x seeds {list(q.seeds)}, {E} epochs | "
          f"rare labels: " + ", ".join(f"{l} {cnt[l]}" for l in labels if cnt[l] < 100))
    oof = {d: [] for d in lab}
    tst = {d: [] for d in test}
    crfs = []
    for k in range(K):
        trd = [d for d in lab if fold[d] != k]
        hod = [d for d in lab if fold[d] == k]
        cw = classweights(data, trd, labels, float(q.get("cw", 0.5))).to(dev)
        for seed in q.seeds:
            m = fitdoc(X, Y, data, trd, q, 100 * k + int(seed), dev, cw, E, sp)
            for d, v in docem(m, X, data, hod, dev, sp).items():
                oof[d].append(v)
            for d, v in docem(m, X, data, test, dev, sp).items():
                tst[d].append(v)
            if m.crf is not None:
                crfs.append({kk: getattr(m.crf, kk).detach().cpu().numpy() for kk in ("trans", "start", "end")})
        hs = macro([l for d in hod for l in data[d]["labels"]],
                   [labels[i] for d in hod for i in np.mean([np.exp(v) for v in oof[d]], 0).argmax(1)], labels)["f1"]
        print(f"  fold {k + 1}/{K}: {len(trd)} train docs, {len(hod)} held out, held-out argmax macro-F1 {hs:.4f}", flush=True)
    em = {d: np.log(np.mean([np.exp(v) for v in (oof[d] if d in oof else tst[d])], 0) + 1e-9).astype(np.float32)
          for d in lab + test}
    crf = {kk: np.mean([c[kk] for c in crfs], 0) for kk in ("trans", "start", "end")} if crfs else None
    perdomain = bool(q.get("perdomain", False))
    bias, ts, fin, hist = calibrate(em, crf, data, lab, labels, q, perdomain)
    va = [d for d in lab if data[d]["split"] == "dev"]
    devf = macro([l for d in va for l in data[d]["labels"]],
                 [labels[i] for d in va for i in decode(em[d], crf, bias[data[d]["domain"] if perdomain else "all"], ts)], labels)["f1"]
    print("OOF macro-F1 (all labelled docs): " + " -> ".join(f"{a} {b:.4f}" for a, b in hist) + f" | dev part {devf:.4f}")
    print("OOF per label: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in fin["per"].items()))
    print("biases: " + "; ".join(f"{kk}: " + ", ".join(f"{l} {b:+.2f}" for l, b in zip(labels, bb) if b) for kk, bb in bias.items()))
    head = sum(p.numel() for p in m.parameters())
    nmod = K * len(q.seeds)
    bbs = {f["phase1"]: f["params"] for f in fs}
    torch.save({"em": em, "vecs": None, "crf": crf, "labels": labels, "params": head * nmod + sum(bbs.values()),
                "trainable": head * nmod, "phase1": ",".join(bbs), "backbones": bbs, "method": "frozen"}, run / "emissions.pt")
    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": "frozen", "mode": "doc-cv",
                                  "dev": {"f1": devf, "oof_f1": fin["f1"]}, "epoch": E, "params": head * nmod + sum(bbs.values()),
                                  "trainable": head * nmod, "phase1_model": "+".join(f["model"] for f in fs)}, 1)
    writesub(cfg, labels, data, {d: decode(em[d], crf, bias[data[d]["domain"] if perdomain else "all"], ts) for d in test},
             {"method": "cross-fitted frozen-feature document BiLSTM-CRF, OOF-calibrated", "views": have,
              "lex": int(q.get("lex", 0)), "folds": K, "seeds": list(q.seeds), "epochs": E, "tscale": ts,
              "bias": {kk: dict(zip(labels, bb.tolist())) for kk, bb in bias.items()}, "oof_progression": hist,
              "oof_f1": fin["f1"], "dev_part_f1": devf, "oof_per_label": {l: v[2] for l, v in fin["per"].items()},
              "params_deployed": head * nmod + sum(bbs.values())})


def cvjoin(cfg):
    """Assemble K fold runs of a fine-tuned model into one cross-fitted member: each labelled doc takes the emissions of
    the fold model that did not train on it, test docs the mean of all fold models; OOF-calibrated submission."""
    q = cfg.cvjoin
    labels, data = loadall(cfg)
    K = len(q.runs)
    fold = cvfolds(data, K)
    lab = sorted(fold)
    test = [d for d, x in data.items() if x["split"] == "test"]
    ms = [torch.load(path(r) / "emissions.pt", map_location="cpu", weights_only=False) for r in q.runs]
    em = {d: ms[fold[d]]["em"][d] for d in lab}
    em.update({d: np.log(np.mean([np.exp(m["em"][d]) for m in ms], 0) + 1e-9).astype(np.float32) for d in test})
    crfs = [m["crf"] for m in ms if m["crf"] is not None]
    crf = {k: np.mean([c[k] for c in crfs], 0) for k in ("trans", "start", "end")} if len(crfs) == K else None
    for k in range(K):
        hod = [d for d in lab if fold[d] == k]
        f = macro([l for d in hod for l in data[d]["labels"]], [labels[i] for d in hod for i in decode(em[d], crf)], labels)["f1"]
        print(f"  fold {k + 1}/{K} ({q.runs[k]}): held-out macro-F1 {f:.4f}")
    bias, ts, fin, hist = calibrate(em, crf, data, lab, labels, cfg.get("seq", {}), False)
    print("OOF macro-F1 (all labelled docs): " + " -> ".join(f"{a} {b:.4f}" for a, b in hist))
    print("OOF per label: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in fin["per"].items()))
    run = path(f"runs/rr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    m0 = ms[0]
    torch.save({"em": em, "vecs": None, "crf": crf, "labels": labels, "params": m0["params"] + (K - 1) * m0["trainable"],
                "trainable": K * m0["trainable"], "phase1": m0["phase1"], "method": m0["method"]}, run / "emissions.pt")
    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": m0["method"], "mode": "ctx-cv",
                                  "dev": {"oof_f1": fin["f1"]}, "epoch": 0, "params": m0["params"], "trainable": K * m0["trainable"]}, 1)
    writesub(cfg, labels, data, {d: decode(em[d], crf, bias["all"], ts) for d in test},
             {"method": f"cross-fitted {m0['method']} ({K} folds), OOF-calibrated", "folds": list(q.runs), "tscale": ts,
              "bias": dict(zip(labels, bias["all"].tolist())), "oof_progression": hist, "oof_f1": fin["f1"],
              "oof_per_label": {l: v[2] for l, v in fin["per"].items()}})


def cvens(cfg):
    """Ensemble of cross-fitted members (each with honest out-of-fold emissions on every labelled doc): member weights,
    CRF transition scale and per-label biases are all tuned on the OOF predictions of ALL labelled docs (train + dev)."""
    e = cfg.ens
    labels, data = loadall(cfg)
    members = loademis(e.runs)
    lab = sorted(d for d, x in data.items() if x["labels"] is not None)
    test = [d for d, x in data.items() if x["split"] == "test"]
    gold = [l for d in lab for l in data[d]["labels"]]

    def wf1(ws):
        em_, crf_ = combine(members, lab, ws)
        return macro(gold, [labels[i] for d in lab for i in decode(em_[d], crf_)], labels)["f1"]

    w = [1.0] * len(members)
    best = wf1(w)
    print(f"{len(members)} cross-fitted members, equal weights: OOF {best:.4f}")
    for _ in range(2):
        for j in range(len(members)):
            for v in e.get("wgrid", [0.0, 0.5, 2.0, 3.0]):
                trial = list(w)
                trial[j] = v
                if sum(trial) == 0:
                    continue
                f = wf1(trial)
                if f > best + 1e-4:
                    best, w = f, trial
    print(f"member weights {dict(zip([Path(r).name for r, _ in members], w))}: OOF {best:.4f}")
    em, crf = combine(members, lab + test, w)
    bias, ts, fin, hist = calibrate(em, crf, data, lab, labels, e, bool(e.get("perdomain", False)))
    print("OOF macro-F1 (all labelled docs): " + " -> ".join(f"{a} {b:.4f}" for a, b in hist))
    print("OOF per label: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in fin["per"].items()))
    pd_ = bool(e.get("perdomain", False))
    writesub(cfg, labels, data, {d: decode(em[d], crf, bias[data[d]["domain"] if pd_ else "all"], ts) for d in test},
             {"method": "OOF-calibrated ensemble of cross-fitted members", "members": [r for r, _ in members],
              "weights": list(w), "tscale": ts, "bias": {k: dict(zip(labels, b.tolist())) for k, b in bias.items()},
              "oof_progression": hist, "oof_f1": fin["f1"], "oof_per_label": {l: v[2] for l, v in fin["per"].items()}})


# ----------------------------------------------------------------------------- data download and check

def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    r = cfg.rr
    have = list_repo_files(r.repo, repo_type="dataset")
    print("remote Phase-3 files:", *[f for f in have if f.startswith("Phase_3")], sep="\n  ")
    for k, f in r.files.items():
        if f not in have:
            hits = [x for x in have if x.startswith("Phase_3") and Path(x).name == Path(f).name]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix rr.files.{k}")
            f = hits[0]
        hf_hub_download(r.repo, f, repo_type="dataset", local_dir=str(path(r.root)))
        print(f"ok {f}")

def check(cfg):
    labels, data = loadall(cfg)
    print(f"labels ({len(labels)}): {labels}")
    tok = gettok(loadcfg(cfg.rr.backbone.config))
    encode(tok, data, 10 ** 9)
    for dom in DOMAINS:
        for split in ("train", "dev", "test"):
            dd = [x for x in data.values() if x["domain"] == dom and x["split"] == split]
            if not dd:
                continue
            ns = [len(x["sents"]) for x in dd]
            nt = [sum(len(i) for i in x["ids"]) for x in dd]
            sl = [len(i) for x in dd for i in x["ids"]]
            print(f"{dom} {split}: {len(dd)} docs, sentences {sum(ns)} (per doc {min(ns)}-{max(ns)}, mean {np.mean(ns):.0f}), "
                  f"tokens/doc mean {np.mean(nt):.0f} max {max(nt)}, tokens/sentence median {np.median(sl):.0f} "
                  f"p95 {np.percentile(sl, 95):.0f} max {max(sl)}")
            if dd[0]["labels"] is not None:
                bad = [x for x in dd if len(x["labels"]) != len(x["sents"])]
                assert not bad, f"{len(bad)} docs with label/sentence count mismatch"
                c = Counter(l for x in dd for l in x["labels"])
                unk = set(c) - set(labels)
                assert not unk, f"unknown labels {unk}"
                print("   " + ", ".join(f"{l} {c[l]}" for l in labels))
    tr = [x for x in data.values() if x["split"] == "train" and x["labels"]]
    pos = defaultdict(list)
    for x in tr:
        for i, l in enumerate(x["labels"]):
            pos[l].append(i / max(1, len(x["labels"])))
    print("mean relative position per label: " + ", ".join(f"{l} {np.mean(v):.2f}" for l, v in pos.items()))
    same = sum(a == b for x in tr for a, b in zip(x["labels"], x["labels"][1:]))
    print(f"label continuity: {same / sum(len(x['labels']) - 1 for x in tr):.1%} of adjacent sentence pairs share a label")


def main():
    cfg, modes = args()
    {"download": download, "check": check, "feats": feats, "seq": seq, "cv": cv, "cvjoin": cvjoin, "cvens": cvens, "train": train, "predict": predict, "ensemble": ensemble, "teach": teach}[modes[0] if modes else "check"](cfg)


if __name__ == "__main__":
    main()
