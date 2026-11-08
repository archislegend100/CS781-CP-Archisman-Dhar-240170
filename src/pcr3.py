"""Phase 7, round 3: cross-fitted learning-to-rank over all retrieval signals.

Per (query, candidate) features: z-scored score of every view (BM25, TF-IDF, each dense member), its hubness-corrected
version, its *reverse rank* (rank of the query among all queries of the pool for that candidate: mutual-nearest-neighbour
evidence against hub candidates), candidate-graph PRF feedback per graph, the candidate's hubness, and the round-2 (P9)
score with its rank. A small MLP scorer is trained with a listwise multi-positive softmax loss (ListNet, Cao et al.
ICML 2007; Burges 2010 for LTR) over each query's top-`depth` candidates. Because the dense members were trained on the
train split (their train scores are over-confident), the ranker is trained on DEV with 5-fold cross-fitting over queries:
every dev query is scored by models that never saw it (honest F1@7), test = mean of the fold models' scores.

python pcr3.py phase7/configs/p12_ltr.yaml ltr
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import pcr
import pcr2
from config import args, path
from data import writejson


def views(cfg, e, data):
    B25 = torch.load(path(cfg.pcr.datadir) / "bm25.pt", map_location="cpu", weights_only=False)
    T = torch.load(path(cfg.pcr.datadir) / f"tfidf{int(e.get('ngram', 2))}.pt", map_location="cpu", weights_only=False)
    names, V, G, gn = ["bm25", "tfidf"], {s: [B25[s], T[s][0]] for s in ("dev", "test")}, {s: [T[s][1]] for s in ("dev", "test")}, ["tfidf"]
    for r in e.runs:
        p = path(r)
        if not (p / "scores.pt").exists():
            print(f"skip {r}")
            continue
        sc = torch.load(p / "scores.pt", map_location="cpu", weights_only=False)
        names.append(p.name)
        for s in ("dev", "test"):
            V[s].append(sc[s])
        if p.name in e.get("graph", []) and (p / "emb.pt").exists():
            E = torch.load(p / "emb.pt", map_location="cpu", weights_only=False)
            gn.append(p.name)
            for s in ("dev", "test"):
                G[s].append((E[s]["C"] @ E[s]["C"].T).numpy())
    return names, V, G, gn


def colrank(Z):
    """For each (q, c): rank of q among all queries for candidate c, scaled to [0, 1] (0 = c's best query)."""
    r = np.argsort(np.argsort(-Z, 0), 0)
    return (r / max(1, Z.shape[0] - 1)).astype(np.float32)


def features(Vs, Gs, base, depth):
    """Feature tensor (Q, depth, F) over each query's top-`depth` candidates by `base`, and their indices."""
    top = np.argsort(-base, 1)[:, :depth]
    F = []
    for v in Vs:
        Z = pcr2.zr(v)
        H = pcr2.hub(v, 0.5)
        F += [Z, H, colrank(Z)]
    for g in Gs:
        F.append(pcr2.zr(pcr2.prf(base, pcr2.zr(g), 5, 1.0) - pcr2.zr(base)))
    Zb = pcr2.zr(base)
    F.append(Zb)
    F.append(colrank(Zb))
    hubv = np.sort(Zb, 0)[-10:].mean(0, keepdims=True).repeat(Zb.shape[0], 0)
    F.append(hubv)
    X = np.stack([f[np.arange(len(top))[:, None], top] for f in F], -1).astype(np.float32)
    rk = np.broadcast_to(np.log1p(np.arange(depth))[None, :, None], (len(top), depth, 1)).astype(np.float32)
    return np.concatenate([X, rk], -1), top


class Ranker(nn.Module):
    def __init__(self, nf, h):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit(X, Y, epochs, seed, h, wd):
    torch.manual_seed(seed)
    m = Ranker(X.shape[-1], h)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd)
    X, Y = torch.tensor(X), torch.tensor(Y)
    keep = Y.sum(1) > 0
    X, Y = X[keep], Y[keep]
    tgt = Y / Y.sum(1, keepdim=True)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(X))
        for a in range(0, len(X), 32):
            b = perm[a:a + 32]
            loss = -(tgt[b] * torch.log_softmax(m(X[b]), -1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def score(models, X):
    with torch.no_grad():
        return np.mean([m(torch.tensor(X)).numpy() for m in models], 0)


def ranked(S_top, top, d):
    out = {}
    for i, q in enumerate(d["qids"]):
        order = top[i][np.argsort(-S_top[i])]
        out[q] = [d["cids"][j] for j in order if d["cids"][j] != q][:pcr.K]
    return out


def ltr(cfg):
    e = cfg.rr
    data = pcr.loaddata(cfg)
    names, V, G, gn = views(cfg, e, data)
    meta = json.load(open(path(e.p9meta)))
    assert meta["views"] == names and meta["graphs"] == gn, f"P9 views {meta['views']} / {meta['graphs']} != {names} / {gn}"
    p9 = [meta["hub_lambda"], meta["weights"], meta["prf_m"], meta["prf_beta"], meta["graph_weights"]]
    d, t = data["dev"], data["test"]
    base = {s: pcr2.pipeline(V[s], G[s], data[s], p9) for s in ("dev", "test")}
    depth = int(e.get("depth", 100))
    Xd, topd = features(V["dev"], G["dev"], base["dev"], depth)
    Xt, topt = features(V["test"], G["test"], base["test"], depth)
    Yd = np.array([[1.0 if d["cids"][j] in set(r) else 0.0 for j in row] for row, r in zip(topd, d["rel"])], dtype=np.float32)
    rel = dict(zip(d["qids"], d["rel"]))
    p9dev = pcr.f1at7({q: pcr.topk(base["dev"][i], d["cids"], q) for i, q in enumerate(d["qids"])}, rel)
    print(f"{len(names)} views, {len(gn)} graphs, {Xd.shape[-1]} features, depth {depth} | recall of the top-{depth}: "
          f"{Yd.sum() / sum(len(r) for r in d['rel']):.3f} | P9 dev F1@7 {p9dev['f1']:.4f}", flush=True)
    nq = len(d["qids"])
    idx = list(range(nq))
    random.Random(0).shuffle(idx)
    folds = [sorted(idx[k::5]) for k in range(5)]
    seeds = list(e.get("seeds", [1, 2, 3]))
    best = None
    for ep in e.get("epochs", [20, 40, 80]):
        for h in e.get("hid", [32, 64]):
            for wd in e.get("wd", [1e-3, 1e-2]):
                oof = np.zeros((nq, depth), dtype=np.float32)
                for k in range(5):
                    tr = [i for f_ in folds[:k] + folds[k + 1:] for i in f_]
                    ms = [fit(Xd[tr], Yd[tr], ep, s, h, wd) for s in seeds]
                    oof[folds[k]] = score(ms, Xd[folds[k]])
                f = pcr.f1at7(ranked(oof, topd, d), rel)
                print(f"  epochs {ep} hidden {h} wd {wd}: cross-fitted dev F1@7 {f['f1']:.4f} P {f['p']:.4f} R {f['r']:.4f}", flush=True)
                if best is None or f["f1"] > best[0]["f1"]:
                    best = (f, ep, h, wd, oof)
    f, ep, h, wd, oof = best
    # blend with P9 (choose on the cross-fitted scores)
    bz = pcr2.zr(np.take_along_axis(base["dev"], topd, 1))
    lz = pcr2.zr(oof)
    bl = max(((pcr.f1at7(ranked(lz + a * bz, topd, d), rel)["f1"], a) for a in (0.0, 0.25, 0.5, 1.0, 2.0)))
    print(f"chosen: epochs {ep} hidden {h} wd {wd} | cross-fitted dev F1@7 {f['f1']:.4f} (P9 {p9dev['f1']:.4f}) | blend with P9 "
          f"a={bl[1]}: {bl[0]:.4f}", flush=True)
    a = bl[1]
    ms = [fit(Xd[tr], Yd[tr], ep, s, h, wd) for k in range(5) for tr in [[i for f_ in folds[:k] + folds[k + 1:] for i in f_]] for s in seeds]
    St = pcr2.zr(score(ms, Xt)) + a * pcr2.zr(np.take_along_axis(base["test"], topt, 1))
    Sfull = np.full(base["test"].shape, -1e9, dtype=np.float32)
    np.put_along_axis(Sfull, topt, St, 1)
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    Sd = np.full(base["dev"].shape, -1e9, dtype=np.float32)
    np.put_along_axis(Sd, topd, lz + a * bz, 1)
    pcr.officialdev(cfg, run, d, Sd)
    pcr.writesub(cfg, t, Sfull, {"method": "cross-fitted listwise LTR over retrieval signals", "views": names, "graphs": gn,
                                 "features": int(Xd.shape[-1]), "depth": depth, "epochs": ep, "hidden": h, "wd": wd,
                                 "blend_p9": a, "dev_f1_crossfit": bl[0], "dev_f1_p9": p9dev["f1"]})


def main():
    cfg, modes = args()
    {"ltr": ltr}[modes[0] if modes else "ltr"](cfg)


if __name__ == "__main__":
    main()
