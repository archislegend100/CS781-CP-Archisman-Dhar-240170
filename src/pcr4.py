"""Phase 7, round 4: new relevance signals for the cross-fitted ranker.

  * chunk-level late interaction (MaxSim): every 512-token chunk of query and candidate is embedded separately by a
    trained Phase-1 retriever; s(q,c) = 1/2 [mean_i max_j cos(q_i, c_j) + mean_j max_i cos(c_j, q_i)] -- a single shared
    legal issue or fact pattern in one part of a long judgment is no longer averaged away (ColBERT, Khattab & Zaharia
    SIGIR 2020, at chunk granularity; BERT-MaxP, Dai & Callan SIGIR 2019; segment-level legal queries, "Segment First,
    Retrieve Better", 2025);
  * query-graph relevance propagation: all queries of a split share one candidate pool, and similar cases tend to cite
    the same precedents, so p(q,c) = sum over q's k nearest queries q' of w(q,q') z(q',c) (collaborative filtering /
    graph-based ensembles over the case graph as in IL-PCSR, EMNLP 2025; transductive, no labels); neighbours by query
    embeddings and by agreement of the queries' ranking vectors;
  * TF-IDF uni+bi+trigrams as an extra lexical view (rare phrase "events", U-CREAT 2023).
These join round-3's features in the 5-fold cross-fitted listwise ranker (bag of 5 seeds).

python pcr4.py phase7/configs/p15_ltr3.yaml chunkemb | ltr
"""
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

import pcr
import pcr2
import pcr3
from config import args, path, wrap


@torch.no_grad()
def chunkemb(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pcr.loaddata(cfg)
    M = int(cfg.rr.get("maxchunks", 16))
    for r in cfg.rr.chunkruns:
        run = path(r)
        if (run / "chunks.pt").exists() or not (run / "best.pt").exists():
            print(f"skip {r}")
            continue
        m, obj = pcr.loadenc(run, dev)
        c = wrap(obj["cfg"]).pcr
        W = int(c.window)
        out = {}
        for s in ("dev", "test"):
            out[s] = {}
            for side in ("q", "c"):
                toks = data[s][f"{side}tok"]
                jobs = [(i, pcr.chunk(t, W, s_)) for i, t in enumerate(toks) for s_ in pcr.starts(len(t), W, M)]
                E = []
                for a in range(0, len(jobs), 32):
                    x = torch.tensor(np.stack([j[1] for j in jobs[a:a + 32]]).astype(np.int64), device=dev)
                    with pcr.autocast(dev, c.get("precision", "bf16")):
                        E.append(m(x).float().half().cpu())
                out[s][side] = (torch.cat(E), torch.tensor([j[0] for j in jobs]))
        torch.save(out, run / "chunks.pt")
        print(f"ok {r}: test chunks q {len(out['test']['q'][1])} c {len(out['test']['c'][1])}", flush=True)
        del m
        torch.cuda.empty_cache()


@torch.no_grad()
def maxsim(qe, qi, ce, ci, nq, nc, dev):
    qe, ce = qe.to(dev).float(), ce.to(dev).float()
    qi, ci = qi.to(dev), ci.to(dev)
    S = torch.zeros(nq, nc, device=dev)
    for q in range(nq):
        Q = qe[qi == q]                                        # (kq, D)
        sim = Q @ ce.T                                         # (kq, M)
        mq = torch.full((Q.size(0), nc), -2.0, device=dev).scatter_reduce(1, ci.expand(Q.size(0), -1), sim, "amax")
        a = mq.mean(0)                                         # query chunks -> best chunk of each candidate
        cm = sim.max(0).values                                 # candidate chunk -> best query chunk
        b = torch.zeros(nc, device=dev).index_add_(0, ci, cm) / torch.bincount(ci, minlength=nc).clamp(min=1)
        S[q] = (a + b) / 2
    return S.cpu().numpy()


@torch.no_grad()
def mscores(m, d, c, dev, M):
    """Symmetric chunk MaxSim scores (Q x C) of a split with an in-memory model."""
    W = int(c.window)
    side = {}
    was = m.training
    m.eval()
    for sd in ("q", "c"):
        toks = d[f"{sd}tok"]
        jobs = [(i, pcr.chunk(t, W, s_)) for i, t in enumerate(toks) for s_ in pcr.starts(len(t), W, M)]
        E = []
        for a in range(0, len(jobs), 32):
            x = torch.tensor(np.stack([j[1] for j in jobs[a:a + 32]]).astype(np.int64), device=dev)
            with pcr.autocast(dev, c.get("precision", "bf16")):
                E.append(m(x).float())
        side[sd] = (torch.cat(E), torch.tensor([j[0] for j in jobs], device=dev))
    m.train(was)
    return maxsim(*side["q"], *side["c"], len(d["qids"]), len(d["cids"]), dev)


def qprop(Z, A, k):
    """Propagate z-scored candidate scores over each query's k nearest other queries (similarity A, Q x Q)."""
    A = A.copy()
    np.fill_diagonal(A, -np.inf)
    nb = np.argsort(-A, 1)[:, :k]
    w = np.take_along_axis(A, nb, 1)
    w = np.exp((w - w.max(1, keepdims=True)) / max(1e-6, w.std()))
    w /= w.sum(1, keepdims=True)
    return pcr2.zr((w[:, :, None] * Z[nb]).sum(1))


def ltr(cfg):
    e = cfg.rr
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pcr.loaddata(cfg)
    names, V, G, gn = pcr3.views(cfg, e, data)
    meta = json.load(open(path(e.p9meta)))
    assert meta["views"] == names and meta["graphs"] == gn, "base re-ranker views differ from this config's views"
    p9 = [meta["hub_lambda"], meta["weights"], meta["prf_m"], meta["prf_beta"], meta["graph_weights"]]
    base = {s: pcr2.pipeline(V[s], G[s], data[s], p9) for s in ("dev", "test")}
    extra, enames = {s: [] for s in ("dev", "test")}, []
    t3 = path(cfg.pcr.datadir) / "tfidf3.pt"
    if t3.exists():
        T3 = torch.load(t3, map_location="cpu", weights_only=False)
        enames.append("tfidf3")
        for s in ("dev", "test"):
            extra[s].append(T3[s][0])
    for r in e.get("extraruns", []):
        p = path(r) / "scores.pt"
        if not p.exists():
            print(f"skip extra {r}")
            continue
        sc = torch.load(p, map_location="cpu", weights_only=False)
        enames.append("extra:" + path(r).name)
        for s in ("dev", "test"):
            extra[s].append(sc[s])
    for r in e.chunkruns:
        p = path(r) / "chunks.pt"
        if not p.exists():
            print(f"skip maxsim {r}")
            continue
        C = torch.load(p, map_location="cpu", weights_only=False)
        enames.append("maxsim:" + path(r).name)
        for s in ("dev", "test"):
            nq, nc = len(data[s]["qids"]), len(data[s]["cids"])
            extra[s].append(maxsim(*C[s]["q"], *C[s]["c"], nq, nc, dev))
    # query graphs: dense query embeddings of the graph members + agreement of base ranking vectors
    qsims = {s: [] for s in ("dev", "test")}
    for r in e.runs:
        p = path(r) / "emb.pt"
        if path(r).name in e.get("graph", []) and p.exists():
            E = torch.load(p, map_location="cpu", weights_only=False)
            for s in ("dev", "test"):
                qsims[s].append((E[s]["Q"] @ E[s]["Q"].T).numpy())
    for s in ("dev", "test"):
        Zb = pcr2.zr(base[s])
        qsims[s].append(np.corrcoef(Zb))
    k = int(e.get("qk", 10))
    d, t = data["dev"], data["test"]
    rel = dict(zip(d["qids"], d["rel"]))
    for nm, X in zip(enames, extra["dev"]):
        print(f"  new view {nm}: dev F1@7 alone {pcr.f1at7({q: pcr.topk(X[i], d['cids'], q) for i, q in enumerate(d['qids'])}, rel)['f1']:.4f}")
    for j in range(len(qsims["dev"])):
        P = qprop(pcr2.zr(base["dev"]), qsims["dev"][j], k)
        f = pcr.f1at7({q: pcr.topk(P[i], d["cids"], q) for i, q in enumerate(d["qids"])}, rel)["f1"]
        print(f"  query-graph propagation #{j}: dev F1@7 alone {f:.4f}")

    def feats(s, depth):
        X, top = pcr3.features(V[s] + extra[s], G[s], base[s], depth)
        P = [qprop(pcr2.zr(base[s]), A, k) for A in qsims[s]] + [qprop(pcr2.zr(v), qsims[s][-1], k) for v in extra[s]]
        P += [qprop(qprop(pcr2.zr(base[s]), qsims[s][-1], k), qsims[s][-1], k)]          # 2-hop propagation
        Pf = np.stack([p_[np.arange(len(top))[:, None], top] for p_ in P], -1).astype(np.float32)
        return np.concatenate([X, Pf], -1), top

    depth = int(e.get("depth", 100))
    Xd, topd = feats("dev", depth)
    Xt, topt = feats("test", depth)
    Yd = np.array([[1.0 if d["cids"][j] in set(r) else 0.0 for j in row] for row, r in zip(topd, d["rel"])], dtype=np.float32)
    base_f1 = pcr.f1at7({q: pcr.topk(base["dev"][i], d["cids"], q) for i, q in enumerate(d["qids"])}, rel)["f1"]
    print(f"{Xd.shape[-1]} features (new views {enames}, {len(qsims['dev'])} query graphs) | base re-ranker dev F1@7 {base_f1:.4f}", flush=True)
    nq = len(d["qids"])
    idx = list(range(nq))
    random.Random(0).shuffle(idx)
    folds = [sorted(idx[f_::5]) for f_ in range(5)]
    seeds = list(e.get("seeds", [1, 2, 3, 4, 5]))
    best = None
    for ep in e.get("epochs", [20, 40, 80]):
        for h in e.get("hid", [32, 64]):
            for wd in e.get("wd", [1e-3, 1e-2]):
                oof = np.zeros((nq, depth), dtype=np.float32)
                for f_ in range(5):
                    tr = [i for g in folds[:f_] + folds[f_ + 1:] for i in g]
                    oof[folds[f_]] = pcr3.score([pcr3.fit(Xd[tr], Yd[tr], ep, sd, h, wd) for sd in seeds], Xd[folds[f_]])
                m_ = pcr.f1at7(pcr3.ranked(oof, topd, d), rel)
                print(f"  epochs {ep} hidden {h} wd {wd}: cross-fitted dev F1@7 {m_['f1']:.4f}", flush=True)
                if best is None or m_["f1"] > best[0]["f1"]:
                    best = (m_, ep, h, wd, oof)
    m_, ep, h, wd, oof = best
    bz = pcr2.zr(np.take_along_axis(base["dev"], topd, 1))
    lz = pcr2.zr(oof)
    bl = max((pcr.f1at7(pcr3.ranked(lz + a * bz, topd, d), rel)["f1"], a) for a in (0.0, 0.25, 0.5, 1.0))
    print(f"chosen: epochs {ep} hidden {h} wd {wd} | cross-fitted dev F1@7 {m_['f1']:.4f} (base {base_f1:.4f}) | blend a={bl[1]}: {bl[0]:.4f}", flush=True)
    a = bl[1]
    ms = [pcr3.fit(Xd[tr], Yd[tr], ep, sd, h, wd) for f_ in range(5) for tr in [[i for g in folds[:f_] + folds[f_ + 1:] for i in g]] for sd in seeds]
    St = pcr2.zr(pcr3.score(ms, Xt)) + a * pcr2.zr(np.take_along_axis(base["test"], topt, 1))
    Sfull = np.full(base["test"].shape, -1e9, dtype=np.float32)
    np.put_along_axis(Sfull, topt, St, 1)
    Sd = np.full(base["dev"].shape, -1e9, dtype=np.float32)
    np.put_along_axis(Sd, topd, lz + a * bz, 1)
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    pcr.officialdev(cfg, run, d, Sd)
    np.save(run / "test_scores.npy", Sfull)
    np.save(run / "dev_scores.npy", Sd)
    pcr.writesub(cfg, t, Sfull, {"method": "cross-fitted LTR + chunk MaxSim + query-graph propagation", "new_views": enames,
                                 "query_graphs": len(qsims["dev"]), "features": int(Xd.shape[-1]), "epochs": ep, "hidden": h,
                                 "wd": wd, "blend": a, "dev_f1_crossfit": bl[0], "dev_f1_base": base_f1})


def ens(cfg):
    """Ensemble of cross-fitted rankers. Each member's dev scores are out-of-fold (honest), so the fusion is chosen on dev:
    reciprocal-rank fusion (Cormack et al. SIGIR 2009) or z-score averaging, member weights by greedy search."""
    e = cfg.rr
    data = pcr.loaddata(cfg)
    d, t = data["dev"], data["test"]
    rel = dict(zip(d["qids"], d["rel"]))
    names, M = [], []
    for r in e.ensruns:
        p = path(r)
        if (p / "dev_scores.npy").exists():
            names.append(p.name)
            M.append({s: np.load(p / f"{s}_scores.npy") for s in ("dev", "test")})
        else:
            print(f"skip {r}")

    def tr(S, kind):
        if kind == "rrf":
            rk = np.argsort(np.argsort(-S, 1), 1)
            return 1.0 / (60.0 + rk)
        Z = np.where(S > -1e8, S, np.nan)
        Z = (Z - np.nanmean(Z, 1, keepdims=True)) / (np.nanstd(Z, 1, keepdims=True) + 1e-6)
        return np.nan_to_num(Z, nan=-10.0)

    f1 = lambda S: pcr.f1at7({q: pcr.topk(S[i], d["cids"], q) for i, q in enumerate(d["qids"])}, rel)["f1"]
    print("members (cross-fitted dev F1@7): " + ", ".join(f"{n} {f1(m['dev']):.4f}" for n, m in zip(names, M)))
    best = None
    for kind in ("rrf", "z"):
        T = [tr(m["dev"], kind) for m in M]
        w = [1.0] * len(M)
        cur = f1(sum(wi * x for wi, x in zip(w, T)))
        for _ in range(2):
            for j in range(len(M)):
                for v in (0.0, 0.5, 2.0, 3.0):
                    w2 = list(w); w2[j] = v
                    if sum(w2) == 0:
                        continue
                    f = f1(sum(wi * x for wi, x in zip(w2, T)))
                    if f > cur + 1e-4:
                        cur, w = f, w2
        print(f"  {kind}: weights {dict(zip(names, w))} -> dev F1@7 {cur:.4f}")
        if best is None or cur > best[0]:
            best = (cur, kind, w)
    cur, kind, w = best
    print(f"chosen: {kind} weights {dict(zip(names, w))} | dev F1@7 {cur:.4f} (best single {max(f1(m['dev']) for m in M):.4f})")
    Sd = sum(wi * tr(m["dev"], kind) for wi, m in zip(w, M))
    St = sum(wi * tr(m["test"], kind) for wi, m in zip(w, M))
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    pcr.officialdev(cfg, run, d, Sd)
    pcr.writesub(cfg, t, St, {"method": f"ensemble of cross-fitted rankers ({kind})", "members": names, "weights": w, "dev_f1": cur})


def main():
    cfg, modes = args()
    {"chunkemb": chunkemb, "ltr": ltr, "ens": ens}[modes[0] if modes else "ltr"](cfg)


if __name__ == "__main__":
    main()
