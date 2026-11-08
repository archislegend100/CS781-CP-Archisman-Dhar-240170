"""Phase 7, round 2: re-ranking on top of the trained retrievers (no new training).

  1. emb   : save query / candidate embeddings of every trained dense member (dev + test) for candidate-graph methods.
  2. rerank: (a) TF-IDF cosine over word uni+bi(+tri)grams as a second lexical view;
             (b) hubness correction (CSLS, Conneau et al. ICLR 2018; inverted softmax, Smith et al. ICLR 2017): candidates
                 that score high for every query of the split (long, generic judgments) are penalised by their mean
                 top-k score over the split's queries -- transductive, uses no labels;
             (c) weighted fusion of all z-normalised views (greedy weights on dev);
             (d) candidate-graph pseudo-relevance feedback (Rocchio 1971; score-level PRF over a candidate-candidate
                 similarity graph built from dense embeddings + TF-IDF): s'(q,c) = s(q,c) + beta * mean_{c' in top-m(q)}
                 G(c', c);
             every choice is tuned on dev; a 5-fold split of the dev queries reports how much of the gain is honest.

python pcr2.py phase7/configs/p9_rerank.yaml emb | rerank
"""
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import pcr
from config import args, path, wrap
from data import writejson


def emb(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pcr.loaddata(cfg)
    for r in cfg.rr.runs:
        run = path(r)
        out = run / "emb.pt"
        if out.exists() or not (run / "best.pt").exists():
            print(f"skip {r}")
            continue
        m, obj = pcr.loadenc(run, dev)
        c = wrap(obj["cfg"]).pcr
        E = {s: {"Q": pcr.embed(m, data[s]["qtok"], c, dev).cpu(), "C": pcr.embed(m, data[s]["ctok"], c, dev).cpu()} for s in ("dev", "test")}
        torch.save(E, out)
        print(f"ok {r}: dev {tuple(E['dev']['C'].shape)} test {tuple(E['test']['C'].shape)}", flush=True)
        del m
        torch.cuda.empty_cache()


def tfidf(qtext, ctext, n=2, mindf=2, maxf=300000):
    """L2-normalised sublinear TF-IDF over word 1..n-grams of a split's queries + pool; returns (Q x C, C x C) cosine."""
    def grams(t):
        w = pcr.words(t)
        g = list(w)
        for k in range(2, n + 1):
            g += ["_".join(w[i:i + k]) for i in range(len(w) - k + 1)]
        return Counter(g)
    docs = [grams(t) for t in list(qtext) + list(ctext)]
    df = Counter(t for d in docs for t in d)
    keep = [t for t, c in df.most_common() if c >= mindf][:maxf]
    voc = {t: i for i, t in enumerate(keep)}
    N = len(docs)
    idf = np.array([math.log((1 + N) / (1 + df[t])) + 1 for t in keep], dtype=np.float32)
    rows, cols, vals = [], [], []
    for r, d in enumerate(docs):
        items = [(voc[t], (1 + math.log(c)) * idf[voc[t]]) for t, c in d.items() if t in voc]
        if not items:
            continue
        v = np.array([x[1] for x in items], dtype=np.float32)
        v /= np.linalg.norm(v)
        rows += [r] * len(items)
        cols += [x[0] for x in items]
        vals += v.tolist()
    nq = len(qtext)
    def mat(lo, hi):
        sel = [(r - lo, c_, v_) for r, c_, v_ in zip(rows, cols, vals) if lo <= r < hi]
        idx = torch.tensor([[x[0] for x in sel], [x[1] for x in sel]], dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.tensor([x[2] for x in sel], dtype=torch.float32), (hi - lo, len(keep))).coalesce()
    Xq, Xc = mat(0, nq), mat(nq, N)
    Xct = Xc.t().coalesce()
    S = torch.sparse.mm(Xq, Xct).to_dense().numpy()
    G = torch.sparse.mm(Xc, Xct).to_dense().numpy()
    return S.astype(np.float32), G.astype(np.float32)


def zr(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


def hub(S, lam, k=10):
    """CSLS-style hubness correction on row-z-scored scores: subtract lam x the candidate's mean top-k score over queries."""
    Z = zr(S)
    if lam == 0:
        return Z
    kk = min(k, Z.shape[0])
    h = np.sort(Z, 0)[-kk:].mean(0, keepdims=True)
    return zr(Z - lam * h)


def prf(S, G, m, beta):
    """Score-level pseudo-relevance feedback over a candidate graph G (C x C, z-scored rows)."""
    if beta == 0:
        return S
    top = np.argsort(-S, 1)[:, :m]
    fb = G[top].mean(1)                              # (Q, C)
    return zr(S) + beta * zr(fb)


def evald(S, d, idx=None):
    qs = range(len(d["qids"])) if idx is None else idx
    ranked = {d["qids"][i]: pcr.topk(S[i], d["cids"], d["qids"][i]) for i in qs}
    return pcr.f1at7(ranked, {d["qids"][i]: d["rel"][i] for i in qs})


def pipeline(views, Gs, d, params):
    lam, w, m, beta, gw = params
    F = sum(wk * hub(v, lam) for wk, v in zip(w, views))
    G = sum(g_w * zr(g) for g_w, g in zip(gw, Gs)) if Gs else None
    return prf(F, G, m, beta) if G is not None else F


def search(views, Gs, d, idx, wgrid, hubs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Coordinate search on the queries idx: hubness lambda -> view weights -> PRF (m, beta, graph mix)."""
    ev = lambda p: evald(pipeline(views, Gs, d, p), d, idx)["f1"]
    nv = len(views)
    p = [0.0, [1.0] * nv, 5, 0.0, [1.0] * len(Gs)]
    best = ev(p)
    for lam in hubs:
        q = list(p); q[0] = lam
        f = ev(q)
        if f > best + 1e-4:
            best, p = f, q
    for _ in range(2):
        for j in range(nv):
            for v in wgrid:
                q = list(p); q[1] = list(p[1]); q[1][j] = v
                if sum(q[1]) == 0:
                    continue
                f = ev(q)
                if f > best + 1e-4:
                    best, p = f, q
    if Gs:
        for m in (3, 5, 7, 10):
            for beta in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75):
                q = list(p); q[2], q[3] = m, beta
                f = ev(q)
                if f > best + 1e-4:
                    best, p = f, q
        if p[3] > 0:
            for j in range(len(Gs)):
                for v in (0.0, 0.5, 2.0):
                    q = list(p); q[4] = list(p[4]); q[4][j] = v
                    if sum(q[4]) == 0:
                        continue
                    f = ev(q)
                    if f > best + 1e-4:
                        best, p = f, q
    return p, best


def rerank(cfg):
    e = cfg.rr
    data = pcr.loaddata(cfg)
    B25 = torch.load(path(cfg.pcr.datadir) / "bm25.pt", map_location="cpu", weights_only=False)
    names, V, Gsrc = ["bm25"], {s: [B25[s]] for s in ("dev", "test")}, {s: [] for s in ("dev", "test")}
    gnames = []
    tf = path(cfg.pcr.datadir) / f"tfidf{int(e.get('ngram', 2))}.pt"
    if not tf.exists():
        T = {s: tfidf(data[s]["qtext"], data[s]["ctext"], int(e.get("ngram", 2))) for s in ("dev", "test")}
        torch.save(T, tf)
    T = torch.load(tf, map_location="cpu", weights_only=False)
    names.append("tfidf")
    gnames.append("tfidf")
    for s in ("dev", "test"):
        V[s].append(T[s][0])
        Gsrc[s].append(T[s][1])
    for r in e.runs:
        p = path(r)
        if not (p / "scores.pt").exists():
            print(f"skip {r}")
            continue
        sc = torch.load(p / "scores.pt", map_location="cpu", weights_only=False)
        names.append(p.name)
        for s in ("dev", "test"):
            V[s].append(sc[s])
        if (p / "emb.pt").exists() and p.name in e.get("graph", []):
            E = torch.load(p / "emb.pt", map_location="cpu", weights_only=False)
            gnames.append(p.name)
            for s in ("dev", "test"):
                Gsrc[s].append((E[s]["C"] @ E[s]["C"].T).numpy())
    d = data["dev"]
    nq = len(d["qids"])
    print(f"views {names} | graphs {gnames} | dev queries {nq}")
    print("singles (dev F1@7): " + ", ".join(f"{n} {evald(zr(v), d)['f1']:.4f}" for n, v in zip(names, V['dev'])))
    wgrid = list(e.get("wgrid", [0.0, 0.5, 1.5, 2.0, 3.0]))
    # honest estimate: 5-fold over dev queries (choose on 4/5, score the held-out 1/5)
    idx = list(range(nq))
    random.Random(0).shuffle(idx)
    folds = [idx[k::5] for k in range(5)]
    hits = {"plain": [], "full": []}
    for k in range(5):
        tr = [i for f_ in folds[:k] + folds[k + 1:] for i in f_]
        pp, _ = search(V["dev"][:1] + V["dev"][2:], [], d, tr, wgrid, hubs=(0.0,))   # round-1 style fusion only
        base = [0.0, [pp[1][0], 0.0] + pp[1][1:], 5, 0.0, [1.0] * len(gnames)]
        pf, _ = search(V["dev"], Gsrc["dev"], d, tr, wgrid)
        hits["plain"].append((evald(pipeline(V["dev"], Gsrc["dev"], d, base), d, folds[k])))
        hits["full"].append((evald(pipeline(V["dev"], Gsrc["dev"], d, pf), d, folds[k])))
    cvf = lambda L: np.mean([x["f1"] for x in L])
    print(f"5-fold dev estimate: plain fusion {cvf(hits['plain']):.4f} -> + TF-IDF / hubness / PRF {cvf(hits['full']):.4f}", flush=True)
    p, best = search(V["dev"], Gsrc["dev"], d, None, wgrid)
    lam, w, m, beta, gw = p
    full = evald(pipeline(V["dev"], Gsrc["dev"], d, p), d)
    print(f"chosen on all dev: hubness lambda {lam} | weights {dict(zip(names, w))} | PRF m {m} beta {beta} graphs {dict(zip(gnames, gw))} "
          f"| dev F1@7 {full['f1']:.4f} P {full['p']:.4f} R {full['r']:.4f}")
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    pcr.officialdev(cfg, run, d, pipeline(V["dev"], Gsrc["dev"], d, p))
    St = pipeline(V["test"], Gsrc["test"], data["test"], p)
    pcr.writesub(cfg, data["test"], St, {"method": "fusion + TF-IDF + hubness correction + candidate-graph PRF", "views": names,
                                         "graphs": gnames, "hub_lambda": lam, "weights": w, "prf_m": m, "prf_beta": beta,
                                         "graph_weights": gw, "dev_f1": full["f1"], "dev_p": full["p"], "dev_r": full["r"],
                                         "cv5_plain": cvf(hits["plain"]), "cv5_full": cvf(hits["full"])})


def main():
    cfg, modes = args()
    {"emb": emb, "rerank": rerank}[modes[0] if modes else "rerank"](cfg)


if __name__ == "__main__":
    main()
