"""Phase 2: legal NER (character-level spans) on top of the Phase-1 models.

python ner.py phase2/configs/n1_seq_full.yaml download   # Phase_2_LNER files from the official dataset (once)
python ner.py phase2/configs/n1_seq_full.yaml check      # formats, offset convention, alignment ceiling per label
python ner.py phase2/configs/n1_seq_full.yaml train      # fine-tune; select the epoch by held-out strict macro-F1
python ner.py phase2/configs/n1_seq_full.yaml predict    # fold3 -> submissions/<id>_<name>/{prediction.json,submission.zip,meta.json}
python ner.py phase2/configs/n6_kd.yaml best             # print the best token-head run (KD teacher)

Text is never normalised: offsets refer to the raw document strings. Tokens are Assignment-1 BPE pieces of
whitespace-separated words, each carrying its exact character span. Heads: BIO token tagging decoded with a
BIO-constrained Viterbi, or a biaffine span scorer (Yu et al. 2020) decoded by greedy non-overlap selection.
Predicted spans are trimmed of punctuation that the training annotations exclude (learned per label).
"""
import bisect
import json
import math
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint import ckptpath
from config import args, loadcfg, path, plain, subdir, wrap
from data import writejson
from generate import autocast
from models import build
from peft import LoRA, LoRMA, LoRMAPi, SVDLinear, addadapters, addbitfit, addia3, count, layers, qlora, wraplinears
from tokenizer import gettok

PUNCT = set(",.;:()[]{}\"'`-/")


# ----------------------------------------------------------------------------- data

def readjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def loadtexts(p):
    d = readjson(p)
    if isinstance(d, list):
        d = {str(r.get("id", r.get("doc_id"))): r for r in d}
    return {str(k): (v if isinstance(v, str) else v.get("text")) for k, v in d.items()}


def entity(e):
    if isinstance(e, dict):
        a = e.get("start", e.get("start_char", e.get("begin")))
        b = e.get("end", e.get("end_char"))
        l = e.get("label", e.get("type", e.get("tag")))
        t = e.get("text", e.get("entity_text"))
    else:
        a, b, l = e[0], e[1], e[2]
        t = e[3] if len(e) > 3 else None
    return int(a), int(b), str(l), t


def loadgold(p):
    d = readjson(p)
    if isinstance(d, list):
        d = {str(r.get("id", r.get("doc_id"))): r for r in d}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = v.get("entities", v.get("annotations", v.get("spans", [])))
        out[str(k)] = [entity(e) for e in v]
    return out


def files(cfg):
    n = cfg.ner
    return {k: path(n.root) / v for k, v in n.files.items()}


def loadall(cfg):
    f = files(cfg)
    labels = [x.strip() for x in open(f["labels"], encoding="utf-8") if x.strip()]
    texts, gold = {}, {}
    for k in ("fold1", "fold2"):
        texts.update(loadtexts(f[f"{k}_text"]))
        gold.update(loadgold(f[f"{k}_gold"]))
    test = loadtexts(f["test"])
    gold = {d: gold.get(d, []) for d in texts}
    return labels, texts, gold, test


def endconvention(texts, gold):
    """1 if the gold end offsets are inclusive (text[a:b+1] == entity text), else 0."""
    ex = inc = tot = 0
    for d, ents in gold.items():
        t = texts[d]
        for a, b, _, s in ents:
            if s is None:
                continue
            tot += 1
            ex += t[a:b] == s
            inc += t[a:b + 1] == s
    return int(inc > ex), {"with_text": tot, "exclusive_match": ex, "inclusive_match": inc}


def exclusive(gold, inc):
    return {d: [(a, b + inc, l, s) for a, b, l, s in ents] for d, ents in gold.items()}


def foldids(cfg):
    f = files(cfg)
    return {k: set(loadtexts(f[f"{k}_text"])) for k in ("fold1", "fold2")}


def valsplit(ids, n, seed):
    rng = random.Random(seed)
    ids = sorted(ids)
    rng.shuffle(ids)
    return set(ids[:n])


# ----------------------------------------------------------------------------- tokenisation with char spans

def tokenize(tok, text):
    """BPE ids and the [start, end) character span of each piece (the word-start marker maps to no characters)."""
    ids, spans = [], []
    for m in re.finditer(r"\S+", text):
        w, ws = m.group(), m.start()
        b2c = []
        for ci, ch in enumerate(w):
            b2c.extend([ci] * len(ch.encode("utf-8")))
        b2c.append(len(w))
        pos = -3
        for pid in tok.word(w):
            nb = len(tok.itos[pid])
            a, b = max(pos, 0), max(pos + nb, 0)
            spans.append((ws + b2c[a], ws + b2c[b]))
            ids.append(pid)
            pos += nb
        assert pos == len(b2c) - 1, f"byte accounting failed for {w!r}"
    return ids, spans


def tagset(labels):
    return ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]


def tokspans(spans, ents, labels):
    """Gold entities as inclusive token ranges (ts, te, k): pieces overlapping the character span."""
    st = np.array([s for s, _ in spans] or [0])
    en = np.array([e for _, e in spans] or [0])
    out = []
    for a, b, l, _ in ents:
        idx = np.nonzero((en > a) & (st < b) & (en > st))[0]
        if len(idx) and l in labels:
            out.append((int(idx[0]), int(idx[-1]), labels.index(l)))
    return out


def biotags(n, tspans):
    tags = np.zeros(n, dtype=np.int64)
    for ts, te, k in sorted(tspans, key=lambda x: (x[0], -x[1])):
        tags[ts] = 1 + 2 * k
        tags[ts + 1:te + 1] = 2 + 2 * k
    return tags


def charspan(spans, ts, te):
    cs = [spans[i] for i in range(ts, te + 1) if spans[i][1] > spans[i][0]]
    return (cs[0][0], cs[-1][1]) if cs else None


def fromtags(tags, spans, labels):
    out, i, n = [], 0, len(tags)
    while i < n:
        t = int(tags[i])
        if t and t % 2 == 1:
            k = (t - 1) // 2
            j = i + 1
            while j < n and int(tags[j]) == 2 + 2 * k:
                j += 1
            c = charspan(spans, i, j - 1)
            if c:
                out.append((c[0], c[1], labels[k]))
            i = j
        else:
            i += 1
    return out


def learntrim(texts, gold):
    """Punctuation the annotators leave outside spans, per label and side (learned from the training gold)."""
    inc, exc = Counter(), Counter()
    for d, ents in gold.items():
        t = texts[d]
        for a, b, l, _ in ents:
            if b > a and t[b - 1] in PUNCT:
                inc[(l, "end", t[b - 1])] += 1
            if b < len(t) and t[b] in PUNCT:
                exc[(l, "end", t[b])] += 1
            if b > a and t[a] in PUNCT:
                inc[(l, "start", t[a])] += 1
            if a > 0 and t[a - 1] in PUNCT:
                exc[(l, "start", t[a - 1])] += 1
    return {key for key in set(inc) | set(exc) if exc[key] > inc[key]}


def trim(text, a, b, l, strip):
    while b > a and (text[b - 1].isspace() or (l, "end", text[b - 1]) in strip):
        b -= 1
    while a < b and (text[a].isspace() or (l, "start", text[a]) in strip):
        a += 1
    return a, b


def windows(n, L, stride):
    if n <= L:
        return [(0, n)] if n else []
    starts = list(range(0, n - L + 1, stride))
    if starts[-1] != n - L:
        starts.append(n - L)
    return [(s, s + L) for s in starts]


def viterbi(lp):
    """Best BIO-consistent tag path: I-k only after B-k or I-k."""
    T, S = lp.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    allowed = np.ones((S, S), dtype=bool)
    for j in range(2, S, 2):
        allowed[:, j] = False
        allowed[j - 1, j] = allowed[j, j] = True
    trans = np.where(allowed, 0.0, -1e9)
    start = np.array([0.0 if (j == 0 or j % 2 == 1) else -1e9 for j in range(S)])
    dp = start + lp[0]
    back = np.zeros((T, S), dtype=np.int64)
    for t in range(1, T):
        sc = dp[:, None] + trans
        back[t] = sc.argmax(0)
        dp = sc.max(0) + lp[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def macro(gold, pred, labels):
    """Strict per-label P/R/F1 and macro averages, exactly as evaluate_phase2.py."""
    st = {l: [0, 0, 0] for l in labels}
    for d in gold:
        g, p = set(gold[d]), set(pred.get(d, []))
        for l in labels:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            st[l][0] += len(gs & ps)
            st[l][1] += len(ps - gs)
            st[l][2] += len(gs - ps)
    per = {}
    for l, (tp, fp, fn) in st.items():
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[l] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    k = len(labels)
    return {"f1": sum(v[2] for v in per.values()) / k, "p": sum(v[0] for v in per.values()) / k,
            "r": sum(v[1] for v in per.values()) / k, "per": per}


# ----------------------------------------------------------------------------- model

class Biaffine(nn.Module):
    """s(i, j, c) = [h_i^s; 1]^T U_c [h_j^e; 1]  (Yu, Bohnet, Poesio, ACL 2020); class 0 = no entity."""

    def __init__(self, dim, ncls, h=256, drop=0.1):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop))
        self.e = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop))
        self.U = nn.Parameter(torch.empty(h + 1, ncls, h + 1))
        nn.init.xavier_uniform_(self.U.view(h + 1, -1))

    def forward(self, x):
        one = x.new_ones(*x.shape[:2], 1)
        hs = torch.cat([self.s(x), one], -1).float()
        he = torch.cat([self.e(x), one], -1).float()
        return torch.einsum("bsi,ick,bek->bsec", hs, self.U.float(), he)


class Tagger(nn.Module):
    def __init__(self, backbone, family, dim, nlabels, head="token", drop=0.1, prompt=0, mix=0):
        super().__init__()
        self.backbone, self.family, self.head = backbone, family, head
        self.mixw = self.gamma = None
        if mix:
            # learned scalar mix of all layer outputs (ELMo, Peters et al. 2018): gamma * sum_l softmax(w)_l norm(h_l)
            assert family == "causal" and not prompt
            self.mixw = nn.Parameter(torch.zeros(mix))
            self.gamma = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(drop)
        if head == "token":
            self.cls = nn.Linear(dim, 1 + 2 * nlabels)
        else:
            self.span = Biaffine(dim, 1 + nlabels, drop=drop)
        self.prompt = None
        if prompt:
            # prompt tuning (Lester et al. 2021): soft prompt initialised from embeddings of real vocabulary tokens
            assert family == "causal", "prompt tuning is implemented for the decoder backbones"
            g = torch.Generator().manual_seed(0)
            rows = torch.randint(260, backbone.emb.weight.size(0), (prompt,), generator=g)
            self.prompt = nn.Parameter(backbone.emb.weight.detach()[rows].clone())

    def hidden(self, ids):
        if self.family == "seq2seq":
            return self.backbone.encode(ids, torch.ones_like(ids, dtype=torch.bool))[0]
        if self.mixw is not None:
            hs = self.backbone(ids, last=True, hiddens=True)["hiddens"][1:]
            w = torch.softmax(self.mixw.float(), 0)
            return self.gamma * sum(w[i] * self.backbone.norm(h) for i, h in enumerate(hs))
        if self.prompt is None:
            return self.backbone(ids, last=True)["hidden"]
        bb, P = self.backbone, self.prompt.size(0)
        x = torch.cat([self.prompt[None].expand(ids.size(0), -1, -1), bb.emb(ids)], 1)
        ex = bb.extras(P + ids.size(1), 0, ids.device, x.dtype)
        ex["pos"] = 0
        for blk in bb.blocks:
            x = blk(x, ex, None)
        return bb.norm(x)[:, P:]

    def forward(self, ids):
        h = self.drop(self.hidden(ids))
        return self.cls(h).float() if self.head == "token" else self.span(h)


def unmask(backbone, spec):
    """Remove the causal mask (Dukic & Snajder 2024) in all or the top-K attention blocks."""
    blocks = [b for b in getattr(backbone, "blocks", []) if hasattr(b, "causal")]
    if spec in (None, "none") or not blocks:
        return 0
    sel = blocks if spec == "all" else blocks[-int(str(spec).split(":")[1]):]
    for b in sel:
        b.causal = False
    return len(sel)


def make(cfg, dev, labels, mcfg=None):
    """Phase-1 backbone + adaptation method + head, as configured. With `mcfg` (an exported model config) the
    architecture is built without reading the Phase-1 checkpoint; the caller then loads the full tagger weights."""
    n = cfg.ner
    b = n.backbone
    if mcfg is None:
        obj = torch.load(ckptpath(path(b.run), b.ckpt), map_location="cpu", weights_only=False)
        mcfg = wrap(obj["cfg"]["model"])
        bb = build(mcfg)
        bb.load_state_dict(obj["model"])
        del obj
    else:
        mcfg = wrap(mcfg)
        bb = build(mcfg)
    family = getattr(bb, "family", "causal")
    if family == "seq2seq":
        bb.dec = nn.ModuleList()
    if n.get("layers"):
        bb.blocks = nn.ModuleList(list(bb.blocks)[:int(n.layers)])
    unmasked = unmask(bb, n.get("unmask", "all"))
    meth = n.method
    for p in bb.parameters():
        p.requires_grad = meth in ("full", "kd")
    # layer-targeted PEFT: adapt only the top K layers (default: every layer)
    where = layers(bb)[-int(n.peftlayers):] if n.get("peftlayers") else [bb]
    if meth == "lora":
        for mod in where:
            wraplinears(mod, LoRA, tuple(n.targets), r=int(n.rank), alpha=float(n.alpha), drop=float(n.get("loradrop", 0.0)))
    elif meth == "lorma":
        for mod in where:
            wraplinears(mod, LoRMA, tuple(n.targets), r=int(n.rank), alpha=float(n.get("alpha", n.rank)))
    elif meth == "rslora":
        for mod in where:
            wraplinears(mod, LoRA, tuple(n.targets), r=int(n.rank), alpha=float(n.alpha), rs=True,
                        drop=float(n.get("loradrop", 0.0)))
    elif meth == "qlora":
        for mod in where:
            qlora(mod, tuple(n.targets), r=int(n.rank), alpha=float(n.alpha), drop=float(n.get("loradrop", 0.0)))
    elif meth == "adalora":
        for mod in where:
            wraplinears(mod, SVDLinear, tuple(n.targets), r=int(n.get("rank0", 12)), alpha=float(n.alpha))
    elif meth == "lormapi":
        wraplinears(bb, LoRMAPi, tuple(n.get("pitargets", ["q", "kv", "o"])), r=int(n.rank))
    elif meth == "ia3":
        addia3(bb)
    elif meth == "prompt":
        pass
    elif meth == "adapter":
        addadapters(bb, int(mcfg.dim), int(n.bottleneck))
    elif meth == "partial":
        for blk in layers(bb)[-int(n.partial):]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in (bb.encnorm if family == "seq2seq" else bb.norm).parameters():
            p.requires_grad = True
    elif meth == "bitfit":
        addbitfit(bb, tuple(n.targets))
    elif meth == "unfreeze":
        pass                      # gradual unfreezing: the training loop opens layers from the top (cj.py)
    elif meth not in ("full", "kd"):
        raise ValueError(meth)
    bb.ckpt = bool(n.get("ckpt", False))
    t = Tagger(bb, family, int(mcfg.dim), len(labels), n.get("head", "token"), float(n.get("drop", 0.1)),
               int(n.get("prompt", 0)) if meth == "prompt" else 0, len(bb.blocks) if n.get("mix") else 0)
    return t.to(dev), {"family": family, "unmasked_blocks": unmasked, "phase1": f"{b.run}:{b.ckpt}",
                       "phase1_model": mcfg.name, "mcfg": plain(mcfg)}


def loadtagger(run, dev):
    obj = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = wrap(obj["cfg"])
    t, _ = make(cfg, dev, obj["labels"])
    t.load_state_dict(obj["model"])
    return t.eval(), obj


# ----------------------------------------------------------------------------- examples

def prepare(tok, texts, gold, labels):
    docs = {}
    for d, text in texts.items():
        ids, spans = tokenize(tok, text)
        ts = tokspans(spans, gold.get(d, []), labels)
        docs[d] = {"text": text, "ids": ids, "spans": spans, "tspans": ts, "tags": biotags(len(ids), ts)}
    return docs


def examples(docs, L, stride):
    out = []
    for d, x in docs.items():
        for s, e in windows(len(x["ids"]), L, stride):
            out.append((d, s, e))
    return out


def batches(items, B, rng):
    """Mini-batches of equal-length windows (no padding, so unmasked attention never sees pads)."""
    bylen = defaultdict(list)
    for it in items:
        bylen[it[2] - it[1]].append(it)
    out = []
    for group in bylen.values():
        rng.shuffle(group)
        out += [group[i:i + B] for i in range(0, len(group), B)]
    rng.shuffle(out)
    return out


def tensors(docs, batch, head, W, dev):
    ids = torch.tensor([docs[d]["ids"][s:e] for d, s, e in batch], device=dev)
    if head == "token":
        y = torch.tensor(np.stack([docs[d]["tags"][s:e] for d, s, e in batch]), device=dev)
        return ids, y
    T = ids.size(1)
    y = torch.zeros(len(batch), T, T, dtype=torch.long)
    for r, (d, s, e) in enumerate(batch):
        for ts, te, k in docs[d]["tspans"]:
            if s <= ts and te < e and te - ts < W:
                y[r, ts - s, te - s] = k + 1
    return ids, y.to(dev)


def band(T, W, dev):
    i = torch.arange(T, device=dev)
    dlt = i[None] - i[:, None]
    return (dlt >= 0) & (dlt < W)


def weights(docs, items, head, nlabels, W, gamma):
    if head == "token":
        c = np.bincount(np.concatenate([docs[d]["tags"] for d in {it[0] for it in items}]),
                        minlength=1 + 2 * nlabels).astype(np.float64)
    else:
        c = np.zeros(1 + nlabels)
        for d, s, e in items:
            T = e - s
            c[0] += T * min(W, T)
            for ts, te, k in docs[d]["tspans"]:
                if s <= ts and te < e and te - ts < W:
                    c[k + 1] += 1
    c = np.maximum(c, 1.0)
    w = (c.max() / c) ** gamma
    return torch.tensor(np.minimum(w, 50.0), dtype=torch.float32)


def loss_fn(model, ids, y, head, W, cw, smooth=0.0, rdrop=0.0):
    out = model(ids)
    if head == "token":
        ce = lambda o: F.cross_entropy(o.reshape(-1, o.size(-1)), y.reshape(-1), weight=cw.to(o.device),
                                       label_smoothing=smooth)
        if not rdrop:
            return ce(out), out
        # R-Drop (Liang et al. 2021): second dropout pass, symmetric KL between the two predictive distributions
        out2 = model(ids)
        a, b = F.log_softmax(out.float(), -1), F.log_softmax(out2.float(), -1)
        m = (y != -100).float()
        k = 0.5 * ((a.exp() * (a - b)).sum(-1) + (b.exp() * (b - a)).sum(-1))
        return 0.5 * (ce(out) + ce(out2)) + rdrop * (k * m).sum() / m.sum().clamp(min=1), out
    m = band(ids.size(1), W, ids.device)
    return F.cross_entropy(out[:, m].reshape(-1, out.size(-1)), y[:, m].reshape(-1), weight=cw.to(out.device)), out


# ----------------------------------------------------------------------------- prediction

@torch.no_grad()
def tokenlogp(model, ids, nlabels, L, stride, B, pr, dev):
    """Per-token tag log-probs for one document; each token is read from the window where it is most central."""
    N = len(ids)
    S = 1 + 2 * nlabels
    lp = np.zeros((N, S), dtype=np.float32)
    cen = np.full(N, -1, dtype=np.int64)
    wins = windows(N, L, stride)
    for i in range(0, len(wins), B):
        chunk = wins[i:i + B]
        with autocast(dev, pr):
            o = model(torch.tensor([ids[s:e] for s, e in chunk], device=dev))
        o = F.log_softmax(o.float(), -1).cpu().numpy()
        for r, (s, e) in enumerate(chunk):
            pos = np.arange(s, e)
            c = np.minimum(pos - s, e - 1 - pos)
            better = c > cen[s:e]
            lp[s:e][better] = o[r][better]
            cen[s:e][better] = c[better]
    return lp


def finish(ents, text, strip):
    res = set()
    for a, b, l in ents:
        a, b = trim(text, a, b, l, strip)
        if b > a:
            res.add((int(a), int(b), l))
    return sorted(res)


@torch.no_grad()
def predictdocs(model, docs, labels, strip, cfg, dev):
    n = cfg.ner
    model.eval()
    L, stride, B, W = int(n.window), int(n.stride), int(n.batch), int(n.get("maxwidth", 64))
    pr = n.get("precision", "bf16")
    out = {}
    for d, x in docs.items():
        ids, spans, text = x["ids"], x["spans"], x["text"]
        N = len(ids)
        wins = windows(N, L, stride)
        ents = []
        if not wins:
            out[d] = []
            continue
        if model.head == "token":
            ents = fromtags(viterbi(tokenlogp(model, ids, len(labels), L, stride, B, pr, dev)), spans, labels)
        else:
            cand = {}
            for i in range(0, len(wins), B):
                chunk = wins[i:i + B]
                with autocast(dev, pr):
                    o = model(torch.tensor([ids[s:e] for s, e in chunk], device=dev))
                p = o.float().softmax(-1)
                T = o.size(1)
                m = band(T, W, dev)
                best, k = p[..., 1:].max(-1)
                keep = (m[None] & (best > p[..., 0])).nonzero()
                if len(keep):
                    r_, a_, b_ = keep.unbind(1)
                    sc, kk = best[r_, a_, b_].cpu().tolist(), k[r_, a_, b_].cpu().tolist()
                    for r, a, b, v, c in zip(r_.cpu().tolist(), a_.cpu().tolist(), b_.cpu().tolist(), sc, kk):
                        key = (chunk[r][0] + a, chunk[r][0] + b, int(c))
                        cand[key] = max(cand.get(key, 0.0), float(v))
            used = np.zeros(N, dtype=bool)
            for (ts, te, kk), _ in sorted(cand.items(), key=lambda kv: -kv[1]):
                if not used[ts:te + 1].any():
                    used[ts:te + 1] = True
                    c = charspan(spans, ts, te)
                    if c:
                        ents.append((c[0], c[1], labels[kk]))
        out[d] = finish(ents, text, strip)
    return out


def ceiling(docs, gold, labels, strip, head, W):
    """Fraction of gold spans recoverable exactly after token alignment, decoding and trimming."""
    hit, tot = Counter(), Counter()
    for d, x in docs.items():
        g = {(a, b, l) for a, b, l, _ in gold[d]}
        if head == "token":
            rec = fromtags(x["tags"], x["spans"], labels)
        else:
            rec = [(*charspan(x["spans"], ts, te), labels[k]) for ts, te, k in x["tspans"]
                   if te - ts < W and charspan(x["spans"], ts, te)]
        rec = {(*trim(x["text"], a, b, l, strip), l) for a, b, l in rec}
        for e in g:
            tot[e[2]] += 1
            hit[e[2]] += e in rec
    return {l: (hit[l] / tot[l] if tot[l] else None, tot[l]) for l in labels}


# ----------------------------------------------------------------------------- ensembling and consistency

def propagate(text, ents, labels_on, minlen=4, fold=False):
    """Document-level label consistency: copy a label to other exact occurrences of a predicted surface string
    (word-bounded, not overlapping existing spans) when that string carries a single label in the document.
    fold=True matches case-insensitively (cause-title names are in capitals, body mentions are not)."""
    by = defaultdict(set)
    for a, b, l in ents:
        by[text[a:b].casefold() if fold else text[a:b]].add(l)
    used = np.zeros(len(text) + 1, dtype=bool)
    for a, b, _ in ents:
        used[a:b] = True
    out = set(ents)
    for sfc, ls in by.items():
        if len(ls) != 1 or len(sfc) < minlen:
            continue
        l = next(iter(ls))
        if l not in labels_on:
            continue
        for m in re.finditer(re.escape(sfc), text, re.I if fold else 0):
            a, b = m.span()
            if used[a:b].any():
                continue
            if (a > 0 and text[a - 1].isalnum()) or (b < len(text) and text[b].isalnum()):
                continue
            out.add((a, b, l))
            used[a:b] = True
    return sorted(out)


def vote(text, ents, labels_on, minlen=3):
    """Two-stage document-level majority label (Krishnan & Manning 2006): mentions of the same name (case-folded
    surface) predicted with different labels all get the majority label, when that majority label is in labels_on."""
    by = defaultdict(Counter)
    for a, b, l in ents:
        if b - a >= minlen:
            by[text[a:b].casefold()][l] += 1
    out = []
    for a, b, l in ents:
        c = by.get(text[a:b].casefold())
        if c and len(c) > 1:
            (m, k), = c.most_common(1)
            if m in labels_on and m != l and k > c[l] and list(c.values()).count(k) == 1:
                l = m
        out.append((a, b, l))
    return sorted(set(out))


PERSON = ("APP", "RESP", "WIT", "A.COUNSEL", "R.COUNSEL", "JUDGE")
PWAFTER = re.compile(r"\s*[\(\[]?\s*P\.?\s*W\.?\s*[-–]?\s*\d+")
PWBEFORE = re.compile(r"P\.?\s*W\.?\s*[-–]?\s*\d+\s*[-–:,]?\s*(?:Dr\.?\s*|Smt\.?\s*|Shri\.?\s*)?$")


def witcue(text, ents):
    """Prosecution-witness cue: a person span right before "(PW-5)" / "PW 5" or right after "PW-5 -" is a witness."""
    out = []
    for a, b, l in ents:
        if l in PERSON and l != "WIT" and (PWAFTER.match(text, b, min(len(text), b + 16))
                                          or PWBEFORE.search(text[max(0, a - 24):a])):
            l = "WIT"
        out.append((a, b, l))
    return sorted(set(out))


VERSUS = re.compile(r"(?<![A-Za-z])(?:versus|vs\.?|v\.)(?![A-Za-z])", re.I)
APPW = re.compile(r"appellants?|petitioners?|plaintiffs?|complainants?", re.I)
RESPW = re.compile(r"respondents?|defendants?|opposite part(?:y|ies)", re.I)
ENDHDR = re.compile(r"respondents?|defendants?|J\s?U\s?D\s?G\s?M\s?E\s?N\s?T|DATE OF|O\s?R\s?D\s?E\s?R|WITH\s", re.I)
PARTYLAB = ("APP", "RESP", "WIT", "A.COUNSEL", "R.COUNSEL", "JUDGE", "AUTH", "COURT")


def header(text, ents, span=300):
    """Cause-title parsing (preamble party extraction, Kalamkar et al. 2022): around each "Versus" that has a party
    role word on either side ("..Appellant(s) Versus .. Respondent(s)", "PETITIONER: .. Vs. RESPONDENT: .."),
    predicted names before it become APP and names after it (up to the respondent marker) become RESP."""
    zones = []
    for m in VERSUS.finditer(text):
        a, b = m.span()
        before, after = text[max(0, a - span):a], text[b:b + span]
        if not (APPW.search(before) or RESPW.search(after)):
            continue
        lab = re.match(r"\s*(?:RESPONDENTS?|DEFENDANTS?)\s*:", text[b:b + 30], re.I)
        if lab:            # "Vs. RESPONDENT: NAME" -- the label opens the respondent zone, it does not close it
            b += lab.end()
        e = ENDHDR.search(text, b, min(len(text), b + span))
        z0 = max(0, a - span)
        p = list(re.finditer(r"PETITIONER\s*:|APPELLANT\s*:|NO\.?\s*[\d/().\s-]+(?:OF\s+\d{4})?", text[z0:a], re.I))
        if p:
            z0 += p[-1].end()
        zones.append((z0, a, "APP"))
        zones.append((b, e.start() if e else min(len(text), b + span), "RESP"))
    if not zones:
        return ents
    out = []
    for a, b, l in ents:
        if l in PARTYLAB:
            for z0, z1, zl in zones:
                if z0 <= a and b <= z1:
                    l = zl
                    break
        out.append((a, b, l))
    return sorted(set(out))


def counselfix(text, ents, window=150):
    """Counsel side from the role word that follows ("Mr. X, learned counsel for the respondent"), else the nearest
    one before the name."""
    out = []
    for a, b, l in ents:
        if l in ("A.COUNSEL", "R.COUNSEL"):
            seg = text[b:min(len(text), b + window)]
            ma, mr = APPW.search(seg), RESPW.search(seg)
            if ma or mr:
                l = "A.COUNSEL" if ma and (not mr or ma.start() < mr.start()) else "R.COUNSEL"
            else:
                seg = text[max(0, a - window):a]
                la, lr = list(APPW.finditer(seg)), list(RESPW.finditer(seg))
                if la or lr:
                    l = "A.COUNSEL" if la and (not lr or la[-1].start() > lr[-1].start()) else "R.COUNSEL"
        out.append((a, b, l))
    return sorted(set(out))


ROLE = re.compile(r"\b(appellants?|petitioners?|plaintiffs?|respondents?|defendants?)\b", re.I)


def rolefix(text, ents, window=80):
    """Re-label APP/RESP by the nearest explicit role word: first in the `window` characters after the name
    (header style "NAME ... Appellant(s)"), otherwise the closest one before it ("the respondent NAME")."""
    out = []
    for a, b, l in ents:
        if l in ("APP", "RESP"):
            m = ROLE.search(text, b, min(len(text), b + window))
            if m is None:
                prev = list(ROLE.finditer(text, max(0, a - window), a))
                m = prev[-1] if prev else None
            if m is not None:
                l = "RESP" if m.group(1).lower().startswith(("respondent", "defendant")) else "APP"
        out.append((a, b, l))
    return sorted(set(out))


def wordspans(text):
    ws = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return ws, [w[0] for w in ws]


def wordrange(ws, starts, a, b):
    j0 = bisect.bisect_right(starts, a) - 1
    if j0 < 0 or ws[j0][1] <= a:
        j0 += 1
    j1 = bisect.bisect_right(starts, b - 1) - 1
    return j0, j1


def boundstats(texts, gold):
    """Per (label, lower-cased word): how often it occurs inside a gold span vs just outside it, for each side.
    ein/eout: non-first words inside / the word right after a span; sin/sout: non-last words inside / the word before."""
    st = {k: Counter() for k in ("ein", "eout", "sin", "sout")}
    for d, ents in gold.items():
        t = texts[d]
        ws, starts = wordspans(t)
        for a, b, l, _ in ents:
            j0, j1 = wordrange(ws, starts, a, b)
            if j0 > j1:
                continue
            for j in range(j0 + 1, j1 + 1):
                st["ein"][(l, t[ws[j][0]:ws[j][1]].lower())] += 1
            for j in range(j0, j1):
                st["sin"][(l, t[ws[j][0]:ws[j][1]].lower())] += 1
            if j1 + 1 < len(ws):
                st["eout"][(l, t[ws[j1 + 1][0]:ws[j1 + 1][1]].lower())] += 1
            if j0 > 0:
                st["sout"][(l, t[ws[j0 - 1][0]:ws[j0 - 1][1]].lower())] += 1
    return st


def repair(text, ents, st, labels_on, strip, thr=0.75, minc=3, maxext=6):
    """Learned boundary repair: extend a span over neighbouring words that the training gold almost always keeps
    inside spans of that label (or drop edge words it almost always leaves outside); never creates overlaps."""
    ws, starts = wordspans(text)
    cur = sorted(ents)
    out = []

    def ratio(kin, kout, l, j):
        w = (l, text[ws[j][0]:ws[j][1]].lower())
        i_, o_ = st[kin][w], st[kout][w]
        return (i_ / (i_ + o_) if i_ + o_ else 0.5), i_ + o_

    for idx, (a, b, l) in enumerate(cur):
        if l not in labels_on:
            out.append((a, b, l))
            continue
        j0, j1 = wordrange(ws, starts, a, b)
        if j0 > j1:
            out.append((a, b, l))
            continue
        o0, o1 = j0, j1
        k = 0
        while j1 + 1 < len(ws) and k < maxext:
            r, c = ratio("ein", "eout", l, j1 + 1)
            if c >= minc and r >= thr:
                j1, k = j1 + 1, k + 1
            else:
                break
        while k == 0 and j1 > j0:
            r, c = ratio("ein", "eout", l, j1)
            if c >= minc and r <= 1 - thr:
                j1 -= 1
            else:
                break
        k = 0
        while j0 > 0 and k < maxext:
            r, c = ratio("sin", "sout", l, j0 - 1)
            if c >= minc and r >= thr:
                j0, k = j0 - 1, k + 1
            else:
                break
        while k == 0 and j0 < j1:
            r, c = ratio("sin", "sout", l, j0)
            if c >= minc and r <= 1 - thr:
                j0 += 1
            else:
                break
        na = a if j0 == o0 else ws[j0][0]
        nb = b if j1 == o1 else ws[j1][1]
        na, nb = trim(text, na, nb, l, strip)
        clash = any(x < nb and na < y for k2, (x, y, _) in enumerate(cur) if k2 != idx)
        out.append((a, b, l) if clash or nb <= na else (na, nb, l))
    return sorted(set(out))


@torch.no_grad()
def ensemble(cfg):
    """Average per-token tag probabilities of several token-head runs (same tokenizer), Viterbi, trim; choose the
    subset and the consistency labels on the shared held-out docs; write the test submission."""
    from itertools import combinations
    e = cfg.ens
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, labels, texts, gold, test, inc, conv = setup(cfg)
    val = valsplit(list(texts), int(cfg.ner.valdocs), cfg.seed)
    vdocs = prepare(tok, {d: texts[d] for d in val}, {}, labels)
    tdocs = prepare(tok, test, {}, labels)
    vgold = {d: [(a, b, l) for a, b, l, _ in gold[d]] for d in val}
    lps, meta, strip = {}, {}, None
    runs = []
    for r in e.runs:
        if not (path(r) / "best.json").exists():
            print(f"skip {r}: not trained")
            continue
        bj = json.load(open(path(r) / "best.json"))
        if bj.get("val_f1") is None or bj["head"] != "token":
            print(f"skip {r}: all-docs run or not a token head")
            continue
        runs.append(r)
    for r in runs:
        m, obj = loadtagger(path(r), dev)
        n = wrap(obj["cfg"]).ner
        args_ = (len(labels), int(n.window), int(n.stride), int(n.batch), n.get("precision", "bf16"), dev)
        lps[r] = ({d: tokenlogp(m, x["ids"], *args_) for d, x in vdocs.items()},
                  {d: tokenlogp(m, x["ids"], *args_) for d, x in tdocs.items()})
        meta[r] = {"params": obj["params"], "val_f1": obj["val"]["f1"]}
        strip = strip or set(map(tuple, obj["strip"]))
        print(f"{r}: single-model held-out F1 {obj['val']['f1']:.4f}, params {obj['params'] / 1e6:.1f}M", flush=True)
        del m
        torch.cuda.empty_cache()

    def decode(rs, which, docs):
        out = {}
        for d, x in docs.items():
            p = np.mean([np.exp(lps[r][which][d]) for r in rs], 0)
            out[d] = finish(fromtags(viterbi(np.log(p + 1e-9)), x["spans"], labels), x["text"], strip)
        return out

    cap = float(e.get("maxparams", 3e8))
    mg = float(e.get("mingain", 0.005))
    trd = [d for d in texts if d not in val]
    stats, bound_on = None, []
    if e.get("fixed"):
        rs = tuple(runs)
        size = sum(meta[r]["params"] for r in rs)
        pv = decode(rs, 0, vdocs)
        base = final = macro(vgold, pv, labels)
        role, on, bound_on = bool(e.get("rolefix", False)), list(e.get("consistency", [])), list(e.get("boundary_labels", []))
        print(f"fixed members: {' + '.join(Path(r).name for r in rs)} ({size / 1e6:.1f}M{', OVER CAP' if size > cap else ''}) | "
              f"held-out F1 {base['f1']:.4f} (members may have trained on these docs; optimistic) | role fix {role} | "
              f"boundary {bound_on} | consistency {on}")
        if bound_on:
            stats = boundstats({d: texts[d] for d in trd}, {d: gold[d] for d in trd})
    else:
        rows = []
        for k in range(1, min(len(runs), int(e.get("maxmodels", 3))) + 1):
            for rs in combinations(runs, k):
                size = sum(meta[r]["params"] for r in rs)
                f1 = macro(vgold, decode(rs, 0, vdocs), labels)["f1"]
                rows.append((f1, rs, size))
                print(f"  {' + '.join(Path(r).name for r in rs):60s} params {size / 1e6:6.1f}M  held-out F1 {f1:.4f}"
                      f"{'' if size <= cap else '  (over cap)'}")
        f1, rs, size = max(r for r in rows if r[2] <= cap)
        pv = decode(rs, 0, vdocs)
        base = macro(vgold, pv, labels)
        role = bool(e.get("rolefix", True))
        if role:
            alt = {d: rolefix(texts[d], pv[d]) for d in val}
            ma = macro(vgold, alt, labels)
            role = ma["f1"] > base["f1"] + mg
            print(f"  role-cue fix: held-out F1 {base['f1']:.4f} -> {ma['f1']:.4f} (APP {base['per']['APP'][2]:.3f}->"
                  f"{ma['per']['APP'][2]:.3f}, RESP {base['per']['RESP'][2]:.3f}->{ma['per']['RESP'][2]:.3f})"
                  f"{'  -> on' if role else '  -> off'}")
            if role:
                pv, base = alt, ma
        if e.get("boundary"):
            stats = boundstats({d: texts[d] for d in trd}, {d: gold[d] for d in trd})
            for l in labels:
                alt = {d: repair(texts[d], pv[d], stats, {l}, strip) for d in val}
                g = macro(vgold, alt, labels)["per"][l][2] - base["per"][l][2]
                if g > mg:
                    bound_on.append(l)
                print(f"  boundary repair {l:10s} per-label F1 change {g:+.4f}{'  -> on' if l in bound_on else ''}")
            if bound_on:
                pv = {d: repair(texts[d], pv[d], stats, set(bound_on), strip) for d in val}
                base = macro(vgold, pv, labels)
        on = []
        for l in labels:
            alt = {d: propagate(texts[d], pv[d], {l}) for d in val}
            gain = macro(vgold, alt, labels)["per"][l][2] - base["per"][l][2]
            if gain > mg:
                on.append(l)
            print(f"  consistency {l:10s} per-label F1 change {gain:+.4f}{'  -> on' if l in on else ''}")
        final = macro(vgold, {d: propagate(texts[d], pv[d], set(on)) for d in val}, labels) if on else base
        print(f"chosen: {' + '.join(Path(r).name for r in rs)} ({size / 1e6:.1f}M) | role fix {role} | boundary "
              f"{bound_on} | held-out F1 {base['f1']:.4f} -> {final['f1']:.4f} with consistency on {on}")
    pt = decode(rs, 1, tdocs)
    if role:
        pt = {d: rolefix(test[d], pt[d]) for d in test}
    if bound_on:
        pt = {d: repair(test[d], pt[d], stats, set(bound_on), strip) for d in test}
    out = {d: [[a, b - inc, l] for a, b, l in propagate(test[d], pt[d], set(on))] for d in test}
    errs = validate(out, test, labels)
    if errs:
        raise SystemExit("prediction invalid: " + "; ".join(errs[:10]))
    sd = subdir(cfg)
    writejson(sd / "prediction.json", out, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": "ensemble (mean token probabilities)",
                                 "members": list(rs), "params_total": size, "consistency_labels": on, "rolefix": role,
                                 "boundary_labels": bound_on, "fixed": bool(e.get("fixed")),
                                 "val_f1_ensemble": base["f1"], "val_f1_final": final["f1"],
                                 "members_val_f1": {r: meta[r]["val_f1"] for r in rs},
                                 "entities": sum(len(v) for v in out.values())}, 1)
    print(f"ok: {len(out)} docs, {sum(len(v) for v in out.values())} entities -> {sd / 'submission.zip'}")


def analyze(cfg):
    """Error types on the held-out docs of a run: exact, boundary (same label, overlapping), label confusion
    (same span), boundary+label, missed, spurious; plus the most frequent confusions."""
    run = path(f"runs/ner/{cfg.name}")
    truth, pred = readjson(run / "val_truth.json"), readjson(run / "val_pred.json")
    kinds, conf = Counter(), Counter()
    for d in truth:
        g = {tuple(x) for x in truth[d]}
        p = {tuple(x) for x in pred.get(d, [])}
        for x in g:
            if x in p:
                kinds[("exact", x[2])] += 1
                continue
            ov = [y for y in p if y[0] < x[1] and x[0] < y[1]]
            if any(y[2] == x[2] for y in ov):
                kinds[("boundary", x[2])] += 1
            elif any((y[0], y[1]) == (x[0], x[1]) for y in ov):
                kinds[("label", x[2])] += 1
                conf[(x[2], next(y[2] for y in ov if (y[0], y[1]) == (x[0], x[1])))] += 1
            elif ov:
                kinds[("boundary+label", x[2])] += 1
                conf[(x[2], ov[0][2])] += 1
            else:
                kinds[("missed", x[2])] += 1
        for y in p:
            if y not in g and not any(x[0] < y[1] and y[0] < x[1] for x in g):
                kinds[("spurious", y[2])] += 1
    labs = sorted({l for _, l in kinds})
    cols = ["exact", "boundary", "label", "boundary+label", "missed", "spurious"]
    print(f"{'label':10s} " + " ".join(f"{c:>15s}" for c in cols))
    for l in labs:
        print(f"{l:10s} " + " ".join(f"{kinds[(c, l)]:15d}" for c in cols))
    print("most frequent confusions (gold -> predicted):", conf.most_common(8))


# ----------------------------------------------------------------------------- training

def setup(cfg):
    tok = gettok(loadcfg(cfg.ner.backbone.config))
    labels, texts, gold, test = loadall(cfg)
    inc, conv = endconvention(texts, gold)
    g = exclusive(gold, inc)
    return tok, labels, texts, g, test, inc, conv


def train(cfg):
    n = cfg.ner
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, labels, texts, gold, test, inc, conv = setup(cfg)
    if n.get("valfold"):
        fi = foldids(cfg)
        val, keep = fi[n.valfold], fi[n.trainfold]
    else:
        val = valsplit(list(texts), int(n.valdocs), cfg.seed) if int(n.valdocs) else set()
        keep = set(texts)
    tr = {d: t for d, t in texts.items() if d not in val and d in keep}
    strip = learntrim({d: texts[d] for d in tr}, {d: gold[d] for d in tr})
    docs = prepare(tok, texts, gold, labels)
    vdocs = {d: docs[d] for d in val}
    head, W = n.get("head", "token"), int(n.get("maxwidth", 64))
    items = examples({d: docs[d] for d in tr}, int(n.window), int(n.stride))
    cw = weights(docs, items, head, len(labels), W, float(n.spancw if head != "token" else n.cw))
    pitems, npe = [], 0
    if n.get("pseudo"):      # self-training: teacher-labelled unlabelled windows (low-confidence tokens = -100)
        assert head == "token", "pseudo-labels are token tags"
        P = torch.load(path(n.pseudo), map_location="cpu", weights_only=False)
        for i in range(P["ids"].size(0)):
            docs[f"__p{i}"] = {"ids": P["ids"][i].tolist(), "tags": P["tags"][i].numpy()}
            pitems.append((f"__p{i}", 0, P["ids"].size(1)))
        npe = min(len(pitems), int(float(n.get("pseudoratio", 0.5)) * len(items)))
        print(f"self-training: {len(pitems)} pseudo-labelled windows from {P['teachers']}, {npe} per epoch, "
              f"{float((P['tags'] >= 0).float().mean()):.1%} tokens kept (conf >= {P['tau']})")
    model, info = make(cfg, dev, labels)
    teacher = None
    if n.method == "kd":
        teacher, tobj = loadtagger(n.kd.teacher, dev)
        assert teacher.head == "token" and model.head == "token", "KD needs token heads on teacher and student"
        for p in teacher.parameters():
            p.requires_grad = False
    unl = unlabeled(cfg, int(n.window)) if n.method == "kd" and int(n.kd.get("unlabeled", 0)) else []
    tot, trn = count(model)
    hp = [p for nm, p in model.named_parameters() if p.requires_grad and not nm.startswith("backbone.")]
    bp = [p for nm, p in model.named_parameters() if p.requires_grad and nm.startswith("backbone.")]
    groups = [g for g in ({"params": bp, "lr": float(n.lr)}, {"params": hp, "lr": float(n.headlr)}) if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=float(n.wd))
    nb = math.ceil((len(items) + npe) / int(n.batch))
    steps = nb * int(n.epochs)
    warm = int(float(n.warmup) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    run = path(f"runs/ner/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | {info['phase1_model']} from {info['phase1']} | method {n.method} | head {head} | "
          f"unmasked blocks {info['unmasked_blocks']} | params {tot / 1e6:.1f}M, trainable {trn / 1e6:.2f}M "
          f"({100 * trn / tot:.2f}%)")
    print(f"docs train {len(tr)} val {len(val)} test {len(test)} | windows {len(items)} x {n.window} | "
          f"gold ends {'inclusive' if inc else 'exclusive'} {conv} | strip rules {len(strip)}")
    vgold = {d: [(a, b, l) for a, b, l, _ in gold[d]] for d in val}
    best, bad, log = -1.0, 0, open(run / "metrics.jsonl", "a")
    pr = n.get("precision", "bf16")
    kdT = float(n.kd.get("T", 2.0)) if teacher is not None else 1.0
    alpha = float(n.kd.get("alpha", 0.5)) if teacher is not None else 1.0
    B = int(n.batch)
    for ep in range(int(n.epochs)):
        model.train()
        t0, tl, k = time.time(), 0.0, 0
        for bi, batch in enumerate(batches(items + rng.sample(pitems, npe), int(n.batch), rng)):
            ids, y = tensors(docs, batch, head, W, dev)
            with autocast(dev, pr):
                loss, out = loss_fn(model, ids, y, head, W, cw, float(n.get("smooth", 0.0)), float(n.get("rdrop", 0.0)))
                if teacher is not None:
                    with torch.no_grad():
                        tlog = teacher(ids).float()
                    loss = alpha * loss + (1 - alpha) * kl(out, tlog, kdT)
                    if unl:
                        u = torch.tensor([unl[((ep * nb + bi) * B + r) % len(unl)] for r in range(B)], device=dev)
                        with torch.no_grad():
                            tu = teacher(u).float()
                        loss = loss + (1 - alpha) * kl(model(u), tu, kdT)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(n.clip))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tl, k = tl + float(loss.detach()), k + 1
        if not vdocs:
            print(f"epoch {ep + 1} | loss {tl / max(1, k):.4f} | no held-out docs (fixed {n.epochs} epochs) | "
                  f"{time.time() - t0:.0f}s", flush=True)
            log.write(json.dumps({"epoch": ep + 1, "loss": tl / max(1, k)}) + "\n")
            log.flush()
            if ep + 1 == int(n.epochs):
                m = {"f1": None, "p": None, "r": None, "per": {}}
                torch.save({"model": model.state_dict(), "cfg": plain(cfg), "labels": labels, "strip": sorted(strip),
                            "inc": inc, "epoch": ep + 1, "val": m, "info": info, "params": tot, "trainable": trn},
                           run / "best.pt")
                writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": n.method, "head": head,
                                              "epoch": ep + 1, "val_f1": None, "alldocs": True,
                                              "params": tot, "trainable": trn, **info}, 1)
            continue
        pred = predictdocs(model, vdocs, labels, strip, cfg, dev)
        m = macro(vgold, pred, labels)
        print(f"epoch {ep + 1} | loss {tl / max(1, k):.4f} | val strict macro-F1 {m['f1']:.4f} P {m['p']:.4f} "
              f"R {m['r']:.4f} | {time.time() - t0:.0f}s", flush=True)
        log.write(json.dumps({"epoch": ep + 1, "loss": tl / max(1, k), "val": m}) + "\n")
        log.flush()
        if m["f1"] > best:
            best, bad = m["f1"], 0
            torch.save({"model": model.state_dict(), "cfg": plain(cfg), "labels": labels, "strip": sorted(strip),
                        "inc": inc, "epoch": ep + 1, "val": m, "info": info, "params": tot, "trainable": trn},
                       run / "best.pt")
            writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": n.method, "head": head,
                                          "epoch": ep + 1, "val_f1": m["f1"], "val_p": m["p"], "val_r": m["r"],
                                          "per_label": {l: v[2] for l, v in m["per"].items()},
                                          "params": tot, "trainable": trn, **info}, 1)
            writejson(run / "val_truth.json", {d: [[a, b - inc, l] for a, b, l in sorted(set(v))] for d, v in vgold.items()})
            writejson(run / "val_pred.json", {d: [[a, b - inc, l] for a, b, l in v] for d, v in pred.items()})
        else:
            bad += 1
            if bad >= int(n.get("patience", 4)):
                print(f"early stop after epoch {ep + 1}")
                break
    log.close()
    if not vdocs:
        print(f"trained on all {len(tr)} docs for {n.epochs} epochs; saved final model")
        return
    print(f"best val strict macro-F1 {best:.4f}")
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase2.py"))
    if ev.exists():
        r = subprocess.run([sys.executable, str(ev), "--truth", str(run / "val_truth.json"), "--prediction",
                            str(run / "val_pred.json"), "--labels", str(files(cfg)["labels"])], capture_output=True, text=True)
        print("official evaluator on the held-out docs:\n" + r.stdout[-600:] + r.stderr[-300:])


def kl(student, teacher, T):
    return F.kl_div(F.log_softmax(student.float() / T, -1), F.softmax(teacher / T, -1), reduction="batchmean") \
        * T * T / max(1, student.size(1))


def unlabeled(cfg, L):
    """Transfer set for distillation: windows of official Phase-1 Supreme Court text (no labels needed)."""
    from cache import Stream, cacheroot
    p1 = loadcfg(cfg.ner.backbone.config)
    st = Stream(cacheroot(p1) / "train" / "supremecourt")
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.ner.kd.unlabeled)
    return [st.get(int(i), L).tolist() for i in rng.integers(0, st.n - L - 1, n)]


# ----------------------------------------------------------------------------- submission

def predict(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = path(f"runs/ner/{cfg.name}")
    model, obj = loadtagger(run, dev)
    tok, labels, texts, gold, test, inc, conv = setup(cfg)
    docs = prepare(tok, test, {}, labels)
    pred = predictdocs(model, docs, labels, set(map(tuple, obj["strip"])), cfg, dev)
    out = {d: [[a, b - obj["inc"], l] for a, b, l in pred[d]] for d in test}
    errs = validate(out, test, labels)
    if errs:
        raise SystemExit("prediction invalid: " + "; ".join(errs[:10]))
    sd = subdir(cfg)
    writejson(sd / "prediction.json", out, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    with zipfile.ZipFile(sd / "submission.zip") as z:
        assert z.namelist() == ["prediction.json"]
    counts = Counter(l for v in out.values() for _, _, l in v)
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": cfg.ner.method, "head": cfg.ner.get("head", "token"),
                                 "phase1": obj["info"], "params": obj["params"], "trainable": obj["trainable"],
                                 "epoch": obj["epoch"], "val_f1": obj["val"]["f1"], "alldocs": obj["val"]["f1"] is None,
                                 "teacher": cfg.ner.kd.get("teacher") if cfg.ner.method == "kd" else None,
                                 "entities": sum(counts.values()),
                                 "per_label_counts": dict(counts), "gold_end_inclusive": obj["inc"]}, 1)
    print(f"ok: {len(out)} docs, {sum(counts.values())} entities -> {sd / 'submission.zip'}")


def validate(out, test, labels):
    errs = []
    if set(out) != set(test):
        errs.append(f"id mismatch: missing {len(set(test) - set(out))}, extra {len(set(out) - set(test))}")
    for d, v in out.items():
        seen = set()
        for s in v:
            a, b, l = s
            if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(test[d])):
                errs.append(f"{d}: bad offsets {s}")
            if l not in labels:
                errs.append(f"{d}: bad label {l}")
            if (a, b, l) in seen:
                errs.append(f"{d}: duplicate {s}")
            seen.add((a, b, l))
    return errs


# ----------------------------------------------------------------------------- out-of-fold calibrated system

def gazetteer(texts, gold, minc=2, purity=0.8, minlen=4):
    """Surface strings annotated with one label at least `minc` times whose word-bounded occurrences in the
    training text carry that label at least `purity` of the time (frequent, unambiguous entity strings)."""
    lab = defaultdict(Counter)
    for d, ents in gold.items():
        t = texts[d]
        for a, b, l, _ in ents:
            if b - a >= minlen:
                lab[t[a:b]][l] += 1
    big = "\n".join(texts[d] for d in gold)
    gaz = {}
    for sfc, c in lab.items():
        l, k = c.most_common(1)[0]
        if k < minc:
            continue
        occ = sum(1 for m in re.finditer(re.escape(sfc), big)
                  if not (m.start() > 0 and big[m.start() - 1].isalnum()) and not (m.end() < len(big) and big[m.end()].isalnum()))
        if occ and k / occ >= purity:
            gaz[sfc] = l
    return gaz


def applygaz(text, ents, gaz, labels_on):
    used = np.zeros(len(text) + 1, dtype=bool)
    for a, b, _ in ents:
        used[a:b] = True
    out = set(ents)
    for sfc, l in sorted(gaz.items(), key=lambda kv: -len(kv[0])):
        if l not in labels_on:
            continue
        for m in re.finditer(re.escape(sfc), text):
            a, b = m.span()
            if used[a:b].any() or (a > 0 and text[a - 1].isalnum()) or (b < len(text) and text[b].isalnum()):
                continue
            out.add((a, b, l))
            used[a:b] = True
    return sorted(out)


_OOF = {}


def _dec(job):
    d, which, bias = job
    lp, spans, text, strip, labels = _OOF[which][d]
    x = lp.copy()
    for k, b in enumerate(bias):
        if b:
            x[:, 1 + 2 * k] += b
            x[:, 2 + 2 * k] += b
    tags = viterbi(x)
    out, i, n = {}, 0, len(tags)
    while i < n:     # as fromtags + finish, keeping each span's confidence (mean ensemble prob of its tags)
        t = int(tags[i])
        if t and t % 2 == 1:
            k = (t - 1) // 2
            j = i + 1
            while j < n and int(tags[j]) == 2 + 2 * k:
                j += 1
            c = charspan(spans, i, j - 1)
            if c:
                a, b = trim(text, c[0], c[1], labels[k], strip)
                if b > a:
                    conf = float(np.exp(lp[np.arange(i, j), tags[i:j]]).mean())
                    out[(int(a), int(b), labels[k])] = max(conf, out.get((int(a), int(b), labels[k]), 0.0))
            i = j
        else:
            i += 1
    return d, out


def sharedsize(meta, rs):
    """Deployment size: PEFT members on the same frozen Phase-1 backbone share it; full / KD / partial count fully."""
    tot, frozen = 0, {}
    for r in rs:
        m = meta[r]
        if m["method"] in ("full", "kd", "partial"):
            tot += m["params"]
        else:
            tot += m["trainable"]
            frozen[m["phase1"]] = max(frozen.get(m["phase1"], 0), m["params"] - m["trainable"])
    return tot + sum(frozen.values())


@torch.no_grad()
def oof(cfg):
    """2-fold out-of-fold system. Every train doc is predicted by the ensemble trained on the other fold; per-label
    decision biases and each post-processing step (role cue, learned boundary repair, gazetteer, consistency) are
    chosen on these 70 honest predictions, with every learned resource (trim rules, boundary statistics, gazetteer)
    built from the other fold only. The test ensemble (all-docs refits or the fold models) then gets the same
    decisions with resources built from all 70 docs."""
    import multiprocessing as mp
    o = cfg.oof
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, labels, texts, gold, test, inc, conv = setup(cfg)
    fi = foldids(cfg)
    other = {"fold1": "fold2", "fold2": "fold1"}
    docs = prepare(tok, texts, gold, labels)
    tdocs = prepare(tok, test, {}, labels)
    grp = {f: {d: texts[d] for d in fi[f]} for f in fi}
    ggrp = {f: {d: gold[d] for d in fi[f]} for f in fi}
    strip = {f: learntrim(grp[f], ggrp[f]) for f in fi}
    strip["all"] = learntrim(texts, gold)
    meta, acc = {}, {"oof": defaultdict(list), "test": defaultdict(list)}

    def score(model, n, dd):
        args_ = (len(labels), int(n.window), int(n.stride), int(n.batch), n.get("precision", "bf16"), dev)
        return {d: np.exp(tokenlogp(model, x["ids"], *args_)) for d, x in dd.items()}

    for f in ("fold1", "fold2"):
        target = {d: docs[d] for d in fi[other[f]]}
        for r in o[f]:
            m, obj = loadtagger(path(r), dev)
            for d, p in score(m, wrap(obj["cfg"]).ner, target).items():
                acc["oof"][d].append(p)
            print(f"oof member {r} (trained on {f}) -> predicted {len(target)} docs of {other[f]}", flush=True)
            del m
            torch.cuda.empty_cache()
    for r in o.test:
        m, obj = loadtagger(path(r), dev)
        meta[r] = {"params": obj["params"], "trainable": obj["trainable"], "phase1": obj["info"]["phase1"],
                   "method": obj["cfg"]["ner"]["method"]}
        for d, p in score(m, wrap(obj["cfg"]).ner, tdocs).items():
            acc["test"][d].append(p)
        print(f"test member {r}", flush=True)
        del m
        torch.cuda.empty_cache()
    size = sharedsize(meta, list(o.test))
    fold_of = {d: f for f in fi for d in fi[f]}
    _OOF["oof"] = {d: (np.log(np.mean(acc["oof"][d], 0) + 1e-9), docs[d]["spans"], texts[d], strip[other[fold_of[d]]], labels)
                   for d in texts}
    _OOF["test"] = {d: (np.log(np.mean(acc["test"][d], 0) + 1e-9), tdocs[d]["spans"], test[d], strip["all"], labels)
                    for d in test}
    gall = {d: [(a, b, l) for a, b, l, _ in gold[d]] for d in texts}
    pool = mp.get_context("fork").Pool(int(o.get("workers", 16)))

    def decodec(which, bias):
        return dict(pool.map(_dec, [(d, which, tuple(bias)) for d in _OOF[which]]))

    def filt(dc, thr):
        return {d: sorted(e for e, c in v.items() if c >= thr.get(e[2], 0.0)) for d, v in dc.items()}

    def decode(which, bias):
        return filt(decodec(which, bias), {})

    bias, thr = [0.0] * len(labels), {}
    cur = decode("oof", bias)
    hist = [("ensemble", macro(gall, cur, labels)["f1"])]
    print(f"OOF macro-F1 ({len(texts)} docs), ensemble: {hist[-1][1]:.4f}", flush=True)
    if o.get("bias", True):
        best = hist[-1][1]
        for _ in range(int(o.get("passes", 1))):
            for k, l in enumerate(labels):
                for v in sorted(o.get("grid", [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]), key=abs):   # ties keep the gentlest bias
                    trial = bias.copy()
                    trial[k] = float(v)
                    f1 = macro(gall, decode("oof", trial), labels)["f1"]
                    if f1 > best + 1e-4:
                        best, bias = f1, trial
                print(f"  bias {l:10s} {bias[k]:+.1f}  OOF macro-F1 {best:.4f}", flush=True)
        cur = decode("oof", bias)
        hist.append(("bias", macro(gall, cur, labels)["f1"]))
    if o.get("conf", False):
        # per-label span-confidence threshold (drop low-confidence spans of labels with spurious hits)
        oc = decodec("oof", bias)
        best = macro(gall, filt(oc, thr), labels)["f1"]
        for l in labels:
            for v in sorted(o.get("confgrid", [0.4, 0.5, 0.6, 0.7, 0.8])):   # ties keep the lowest threshold
                trial = {**thr, l: float(v)}
                f1 = macro(gall, filt(oc, trial), labels)["f1"]
                if f1 > best + 1e-4:
                    best, thr = f1, trial
            print(f"  conf  {l:10s} >= {thr.get(l, 0.0):.1f}  OOF macro-F1 {best:.4f}", flush=True)
        cur = filt(oc, thr)
        hist.append(("conf", macro(gall, cur, labels)["f1"]))
    mg = float(o.get("mingain", 0.003))
    steps = list(o.get("steps", ["rolefix", "boundary", "gazetteer", "consistency"]))
    bstats = {f: boundstats(grp[f], ggrp[f]) for f in fi}
    gaz = {f: gazetteer(grp[f], ggrp[f]) for f in fi}
    chosen = {}

    def apply(step, d, ents, on, src):
        t = texts[d] if src != "all" else test[d]
        if step == "rolefix":
            return rolefix(t, ents)
        if step == "boundary":
            return repair(t, ents, src_stats[src], on, strip[src])
        if step == "gazetteer":
            return applygaz(t, ents, src_gaz[src], on)
        if step == "witcue":
            return witcue(t, ents)
        if step == "header":
            return header(t, ents)
        if step == "counselfix":
            return counselfix(t, ents)
        if step == "vote":
            return vote(t, ents, on)
        if step == "consistency_ci":
            return propagate(t, ents, on, fold=True)
        return propagate(t, ents, on)

    src_stats = dict(bstats)
    src_gaz = dict(gaz)
    for step in steps:
        base = macro(gall, cur, labels)
        if step in ("rolefix", "witcue", "header", "counselfix"):
            alt = {d: apply(step, d, cur[d], None, other[fold_of[d]]) for d in texts}
            ma = macro(gall, alt, labels)
            on = ma["f1"] > base["f1"] + mg
            print(f"  {step}: OOF {base['f1']:.4f} -> {ma['f1']:.4f}{'  -> on' if on else '  -> off'}", flush=True)
            if on:
                cur = alt
            chosen[step] = bool(on)
        else:
            on = []
            for l in labels:
                alt = {d: apply(step, d, cur[d], {l}, other[fold_of[d]]) for d in texts}
                g = macro(gall, alt, labels)["per"][l][2] - base["per"][l][2]
                if g > mg:
                    on.append(l)
            if on:
                cur = {d: apply(step, d, cur[d], set(on), other[fold_of[d]]) for d in texts}
            print(f"  {step}: labels {on} | OOF {base['f1']:.4f} -> {macro(gall, cur, labels)['f1']:.4f}", flush=True)
            chosen[step] = on
        hist.append((step, macro(gall, cur, labels)["f1"]))
    final = macro(gall, cur, labels)
    print("OOF per-label F1: " + ", ".join(f"{l} {v[2]:.3f}" for l, v in final["per"].items()))
    src_stats["all"] = boundstats(texts, gold)
    src_gaz["all"] = gazetteer(texts, gold)
    pt = filt(decodec("test", bias), thr)
    pool.close()
    for step in steps:
        on = chosen.get(step)
        if not on:
            continue
        pt = {d: apply(step, d, pt[d], set(on) if isinstance(on, list) else None, "all") for d in test}
    out = {d: [[a, b - inc, l] for a, b, l in pt[d]] for d in test}
    errs = validate(out, test, labels)
    if errs:
        raise SystemExit("prediction invalid: " + "; ".join(errs[:10]))
    sd = subdir(cfg)
    writejson(sd / "prediction.json", out, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, "method": "OOF-calibrated ensemble + validated rules",
                                 "oof_members": {f: list(o[f]) for f in fi}, "test_members": list(o.test),
                                 "params_deployed": size, "bias": dict(zip(labels, bias)), "conf": thr, "steps": chosen,
                                 "oof_progression": hist, "oof_per_label": {l: v[2] for l, v in final["per"].items()},
                                 "entities": sum(len(v) for v in out.values())}, 1)
    print(f"progression: " + " -> ".join(f"{k} {v:.4f}" for k, v in hist))
    print(f"ok: {len(out)} docs, {sum(len(v) for v in out.values())} entities, deployed params {size / 1e6:.1f}M "
          f"-> {sd / 'submission.zip'}")


@torch.no_grad()
def pseudo(cfg):
    """Self-training data (Noisy Student, Xie et al. 2020; BOND, Liang et al. 2020): a teacher ensemble (mean token
    probabilities, optional OOF-tuned per-label biases, constrained Viterbi) tags unlabelled official Phase-1 Supreme
    Court windows; tokens whose ensemble confidence is below tau are ignored (-100) in the student loss."""
    q = cfg.pseudo
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = [x.strip() for x in open(files(cfg)["labels"], encoding="utf-8") if x.strip()]
    L, B = int(cfg.ner.window), int(q.get("batch", 16))
    cfg.ner.kd = wrap({"unlabeled": int(q.n)})
    cfg.seed = int(q.get("seed", cfg.seed))
    unl = torch.tensor(unlabeled(cfg, L))
    bias = np.zeros(1 + 2 * len(labels), dtype=np.float32)
    if q.get("biasfrom"):
        bm = json.load(open(path(q.biasfrom)))["bias"]
        for k, l in enumerate(labels):
            bias[1 + 2 * k] = bias[2 + 2 * k] = float(bm.get(l, 0.0))
    prob = torch.zeros(unl.size(0), L, 1 + 2 * len(labels))
    for r in q.teachers:
        m, obj = loadtagger(path(r), dev)
        pr = obj["cfg"]["ner"].get("precision", "bf16")
        for i in range(0, unl.size(0), B):
            with autocast(dev, pr):
                o = m(unl[i:i + B].to(dev))
            prob[i:i + B] += F.softmax(o.float(), -1).cpu()
        print(f"teacher {r}: {unl.size(0)} windows", flush=True)
        del m
        torch.cuda.empty_cache()
    prob /= len(q.teachers)
    lp = (torch.log(prob + 1e-9) + torch.from_numpy(bias)).numpy()
    tags = torch.from_numpy(np.stack([viterbi(x) for x in lp]))
    conf = prob.gather(-1, tags.unsqueeze(-1)).squeeze(-1)
    tau = float(q.get("tau", 0.9))
    tags[conf < tau] = -100
    ent = int(((tags > 0) & (tags % 2 == 1)).sum())
    out = path(q.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": unl.to(torch.int32), "tags": tags, "teachers": list(q.teachers), "tau": tau,
                "bias": bias.tolist()}, out)
    print(f"ok: {unl.size(0)} x {L} windows, {float((tags >= 0).float().mean()):.1%} tokens kept, {ent} entity "
          f"starts -> {out}")


# ----------------------------------------------------------------------------- export and inference benchmark

def export(cfg):
    """Self-contained inference package for a run: fp16 tagger weights + architecture config (no Phase-1 checkpoint
    needed to rebuild), labels, trim rules and offset convention -> runs/ner/<name>/export.pt."""
    run = path(f"runs/ner/{cfg.name}")
    obj = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    t, info = make(wrap(obj["cfg"]), torch.device("cpu"), obj["labels"])
    t.load_state_dict(obj["model"])
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in t.state_dict().items()}
    torch.save({"model": sd, "mcfg": info["mcfg"], "cfg": obj["cfg"], "labels": obj["labels"], "strip": obj["strip"],
                "inc": obj["inc"], "params": obj["params"], "trainable": obj["trainable"], "epoch": obj["epoch"],
                "val": obj["val"], "info": {k: v for k, v in info.items() if k != "mcfg"}}, run / "export.pt")
    print(f"exported {run / 'export.pt'} ({(run / 'export.pt').stat().st_size / 2**20:.0f} MiB)")


def loadexport(p, dev):
    obj = torch.load(p, map_location="cpu", weights_only=False)
    t, _ = make(wrap(obj["cfg"]), dev, obj["labels"], mcfg=obj["mcfg"])
    t.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in obj["model"].items()})
    return t.eval(), obj


@torch.no_grad()
def bench(cfg):
    """Inference cost on the 35 test docs: load time, forward-only and end-to-end time, tokens/s, peak GPU memory;
    also checks that the predictions match the submitted ones. Writes bench/<name>_<gpu>.json."""
    assert torch.cuda.is_available(), "bench needs a GPU"
    dev = torch.device("cuda")
    run = path(f"runs/ner/{cfg.name}")
    src = run / "export.pt"
    t0 = time.time()
    if src.exists():
        model, obj = loadexport(src, dev)
    else:
        model, obj = loadtagger(run, dev)
    torch.cuda.synchronize()
    load_s = time.time() - t0
    tok, labels, texts, gold, test, inc, conv = setup(cfg)
    docs = prepare(tok, test, {}, labels)
    n = wrap(obj["cfg"]).ner
    pr = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    run_cfg = wrap(plain(obj["cfg"]))
    run_cfg.ner.precision = pr
    L, stride, B = int(n.window), int(n.stride), int(n.batch)
    first = next(iter(docs))
    predictdocs(model, {first: docs[first]}, labels, set(map(tuple, obj["strip"])), run_cfg, dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ntok = nwin = 0
    t1 = time.time()
    for x in docs.values():
        wins = windows(len(x["ids"]), L, stride)
        nwin += len(wins)
        for i in range(0, len(wins), B):
            chunk = wins[i:i + B]
            ntok += sum(e - s for s, e in chunk)
            with autocast(dev, pr):
                model(torch.tensor([x["ids"][s:e] for s, e in chunk], device=dev))
    torch.cuda.synchronize()
    fwd_s = time.time() - t1
    t2 = time.time()
    pred = predictdocs(model, docs, labels, set(map(tuple, obj["strip"])), run_cfg, dev)
    torch.cuda.synchronize()
    total_s = time.time() - t2
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    out = {d: [[a, b - obj["inc"], l] for a, b, l in pred[d]] for d in test}
    agree = None
    sub = subdir(cfg) / "prediction.json"
    if sub.exists():
        ref = readjson(sub)
        a_ = {(d, *map(str, x)) for d, v in out.items() for x in v}
        b_ = {(d, *map(str, x)) for d, v in ref.items() for x in v}
        agree = len(a_ & b_) / max(1, len(a_ | b_))
    gpu = torch.cuda.get_device_name(0)
    res = {"id": cfg.id, "name": cfg.name, "method": n.method, "head": n.get("head", "token"),
           "phase1_model": obj.get("info", {}).get("phase1_model"), "gpu": gpu, "precision": pr,
           "torch": torch.__version__, "params": obj["params"], "trainable": obj["trainable"],
           "weights_mib": sum(v.numel() for v in model.state_dict().values()) * 2 / 2 ** 20,
           "load_s": round(load_s, 2), "docs": len(docs), "tokens": ntok, "windows": nwin, "window": L, "batch": B,
           "fwd_s": round(fwd_s, 3), "total_s": round(total_s, 3), "fwd_tok_per_s": round(ntok / fwd_s),
           "total_tok_per_s": round(ntok / total_s), "peak_mem_mib": round(peak), "entities": sum(len(v) for v in out.values()),
           "agreement_with_submission": None if agree is None else round(agree, 4)}
    slug = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")
    writejson(path(f"bench/{cfg.name}_{slug}.json"), res, 1)
    print(json.dumps(res))


def benchtable(cfg):
    rows = [readjson(p) for p in sorted(path("bench").glob("*.json"))]
    print("| id | name | model | method | head | GPU | prec | params | trainable | weights MiB | peak MiB | fwd tok/s | end-to-end tok/s | total s | agree |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["gpu"], r["id"])):
        print(f"| {r['id']} | {r['name']} | {r['phase1_model']} | {r['method']} | {r['head']} | {r['gpu']} | {r['precision']} | "
              f"{r['params'] / 1e6:.1f}M | {r['trainable'] / 1e6:.2f}M | {r['weights_mib']:.0f} | {r['peak_mem_mib']} | "
              f"{r['fwd_tok_per_s']} | {r['total_tok_per_s']} | {r['total_s']} | {r['agreement_with_submission']} |")


# ----------------------------------------------------------------------------- utilities

def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    n = cfg.ner
    have = list_repo_files(n.repo, repo_type="dataset")
    print("remote Phase-2 files:", *[f for f in have if f.startswith("Phase_2")], sep="\n  ")
    for k, f in n.files.items():
        if f not in have:
            hits = [x for x in have if x.startswith("Phase_2") and Path(x).name == Path(f).name]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix ner.files.{k}")
            f = hits[0]
        hf_hub_download(n.repo, f, repo_type="dataset", local_dir=str(path(n.root)))
        print(f"ok {f}")


def check(cfg):
    tok, labels, texts, gold, test, inc, conv = setup(cfg)
    f = files(cfg)
    print(f"labels ({len(labels)}): {labels}")
    print(f"train docs {len(texts)} | test docs {len(test)} | gold end offsets {'INCLUSIVE' if inc else 'exclusive'} {conv}")
    raw = loadgold(f["fold1_gold"])
    k0 = next(iter(raw))
    print(f"gold example ({k0}): {raw[k0][:3]}")
    cnt = Counter(l for v in gold.values() for _, _, l, _ in v)
    print("entities per label:", dict(cnt.most_common()))
    over = 0
    for v in gold.values():
        v = sorted(v, key=lambda e: (e[0], e[1]))
        over += sum(1 for x, y in zip(v, v[1:]) if y[0] < x[1])
    bad = sum(1 for d, v in gold.items() for a, b, *_ in v if not 0 <= a < b <= len(texts[d]))
    print(f"overlapping gold pairs {over} | out-of-range spans {bad}")
    docs = prepare(tok, texts, gold, labels)
    lens = sorted(len(x["ids"]) for x in docs.values())
    print(f"tokens per doc: min {lens[0]} median {lens[len(lens) // 2]} max {lens[-1]}")
    strip = learntrim(texts, gold)
    print(f"learned strip rules: {sorted(strip)[:20]}{' ...' if len(strip) > 20 else ''}")
    W = int(cfg.ner.get("maxwidth", 64))
    for head in ("token", "biaffine"):
        c = ceiling(docs, gold, labels, strip, head, W)
        allh = sum((v[0] or 0) * v[1] for v in c.values()) / max(1, sum(v[1] for v in c.values()))
        print(f"alignment ceiling ({head}): overall {allh:.4f} | " + ", ".join(f"{l} {v[0]:.3f}" for l, v in c.items() if v[0] is not None))
    widths = Counter(min(te - ts + 1, 999) // 16 * 16 for x in docs.values() for ts, te, _ in x["tspans"])
    print(f"entity widths in tokens (bucket of 16): {dict(sorted(widths.items()))}")


def best(cfg):
    rows = []
    for j in sorted(path("runs/ner").glob("*/best.json")):
        b = json.load(open(j))
        if b["head"] == "token" and b["method"] != "kd" and b.get("val_f1") is not None:
            rows.append((b["val_f1"], str(j.parent)))
    if not rows:
        raise SystemExit("no token-head run with best.json")
    print(max(rows)[1])


def main():
    cfg, modes = args()
    {"download": download, "check": check, "train": train, "predict": predict, "best": best, "ensemble": ensemble,
     "analyze": analyze, "export": export, "bench": bench, "benchtable": benchtable, "oof": oof, "pseudo": pseudo}[modes[0] if modes else "check"](cfg)


if __name__ == "__main__":
    main()
