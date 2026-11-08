"""Estimate the source mix of the official test prompts (CPU only).

python testsrc.py phase1/configs/legalgpt.yaml [srctok=2e7]

A multinomial naive-Bayes classifier over hashed token bigrams is fitted on random windows from each
source's *train* cache, checked on the dev prompts (validation documents with known sources), then
applied to the 1000 test prompts. Writes runs/testsrc.json.
"""
import json

import numpy as np

from config import args, path
from data import writejson

B = 1 << 22
WIN = 4096


def grams(ids, V):
    ids = np.asarray(ids, dtype=np.int64)
    return (ids[:-1] * V + ids[1:]) % B


def fit(streams, V, ntok, seed=0):
    rng = np.random.default_rng(seed)
    logp = {}
    for s, st in streams.items():
        parts, got = [], 0
        while got < ntok:
            i = int(rng.integers(0, max(1, st.n - WIN - 1)))
            parts.append(grams(st.get(i, min(WIN + 1, st.n)), V))
            got += WIN
        c = np.bincount(np.concatenate(parts), minlength=B).astype(np.float64)
        logp[s] = np.log((c + 1.0) / (c.sum() + B))
    return logp


def score(logp, ids, V):
    g = grams(ids, V)
    return {s: float(lp[g].sum()) / max(1, len(g)) for s, lp in logp.items()}


def predict(logp, ids, V):
    sc = score(logp, ids, V)
    order = sorted(sc, key=sc.get, reverse=True)
    return order[0], sc[order[0]] - sc[order[1]], sc


def main():
    cfg, _ = args()
    from cache import Stream, cacheroot
    from tokenizer import gettok
    tok = gettok(cfg)
    root = cacheroot(cfg, tok)
    srcs = list(cfg.data.files)
    streams = {s: Stream(root / "train" / s) for s in srcs}
    ntok = int(float(cfg.get("srctok", 2e7)))
    print(f"fitting on {ntok / 1e6:.0f}M train tokens per source ({', '.join(srcs)})")
    logp = fit(streams, tok.vocab, ntok, int(cfg.seed))

    dev = json.load(open(path(cfg.data.dev) / "prompts.json", encoding="utf-8"))
    conf = {a: {b: 0 for b in srcs} for a in srcs}
    for rec in dev.values():
        p, _, _ = predict(logp, tok.encode(rec["text"]), tok.vocab)
        conf[rec["source"]][p] += 1
    ok = sum(conf[s][s] for s in srcs)
    n = sum(sum(r.values()) for r in conf.values())
    print(f"\ndev accuracy {ok}/{n} = {ok / max(1, n):.3f}   (rows = true source, cols = predicted)")
    print(f"{'':14s}" + "".join(f"{s[:12]:>13s}" for s in srcs))
    for a in srcs:
        print(f"{a:14s}" + "".join(f"{conf[a][b]:13d}" for b in srcs))

    test = json.load(open(path(cfg.data.root) / cfg.data.test, encoding="utf-8"))
    preds, counts, margins = {}, {s: 0 for s in srcs}, []
    for k, rec in test.items():
        p, m, sc = predict(logp, tok.encode(rec["text"]), tok.vocab)
        preds[k] = {"source": p, "margin": m, "scores": sc}
        counts[p] += 1
        margins.append(m)
    recall = {s: conf[s][s] / max(1, sum(conf[s].values())) for s in srcs}
    print(f"\ntest prompts: {len(test)}")
    print(f"{'source':14s} {'predicted':>9s} {'share':>7s} {'dev recall':>11s}")
    for s in srcs:
        print(f"{s:14s} {counts[s]:9d} {counts[s] / len(test):7.3f} {recall[s]:11.3f}")
    print(f"median per-token log-prob margin {np.median(margins):.3f} (larger = more confident)")
    writejson(path("runs/testsrc.json"), {"dev_confusion": conf, "dev_accuracy": ok / max(1, n),
                                         "test_counts": counts, "predictions": preds}, 1)
    print("wrote runs/testsrc.json")


if __name__ == "__main__":
    main()
