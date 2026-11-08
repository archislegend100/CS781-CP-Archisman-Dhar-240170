"""Candidate pools and minimum-Bayes-risk system combination (inference only, no test labels).

python pool.py gen phase1/configs/z5_mbr.yaml tag=z3 k=15     -> runs/pool/z3.json  (greedy + k epsilon-samples per prompt)
python pool.py select phase1/configs/f3_mix.yaml              -> submissions/<id>_<subname>/prediction.json  (then submit.py)

Selection: for every prompt, the candidate with the highest mean utility against the other candidates, where
utility = wr * ROUGE-L + wb * BLEU-4 + we * EmbF1. EmbF1 is BERTScore-style greedy token matching on the hidden
states of one of the pooled models (our own weights), not an external model.
"""
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import ckptpath
from config import args, path, plain, subdir, wrap
from data import loadtest, writejson
from generate import autocast, complete, loadmodel, loadpath
from selfrank import bleu, rougel
from tokenizer import gettok


def gen(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = gettok(cfg)
    g = cfg.generation
    model, state, ck = loadmodel(cfg, g.get("ckpt", "best"), dev)
    k = int(cfg.get("k", 15))
    gs = wrap({**plain(g), "mode": "sample", "temperature": float(cfg.get("temp", 0.7)), "topk": 0, "topp": 1.0,
               "eps": float(cfg.get("eps", 0.02))})
    prompts = loadtest(cfg, cfg.get("which", "test"))
    enc = {i: tok.encode(t) for i, t in prompts.items()}
    order = sorted(enc, key=lambda i: len(enc[i]))
    m = cfg.model
    ctx, pr, bs = int(m.get("ctx", m.seq)), cfg.train.get("precision", "bf16"), int(g.batch)
    greedy = not cfg.get("nogreedy")
    off = int(cfg.get("seedoff", 0))
    cands, t0 = {}, time.time()
    for b in range(0, len(order), bs):
        chunk = order[b:b + bs]
        x = [enc[i] for i in chunk]
        outs = [complete(model, tok, x, g, dev, pr, ctx, int(g.seed) + b)[0]] if greedy else []
        outs += [complete(model, tok, x, gs, dev, pr, ctx, int(g.seed) + b + off + 7919 * (j + 1))[0] for j in range(k)]
        for n, i in enumerate(chunk):
            cands[i] = [o[n] for o in outs]
        print(f"{min(b + bs, len(order))}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    meta = {"tag": cfg.tag, "checkpoint": str(ck), "step": state.get("step"), "k": k, "has_greedy": greedy,
            "seedoff": off, "greedy": plain(g),
            "sample": {"temperature": gs.temperature, "eps": gs.eps}, "seconds": round(time.time() - t0)}
    writejson(path(f"runs/pool/{cfg.tag}.json"), {"meta": meta, "cands": {i: cands[i] for i in prompts}})
    print(f"wrote runs/pool/{cfg.tag}.json ({k + int(greedy)} candidates per prompt)")


def lexmat(job):
    cs, wr, wb = job
    n = len(cs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = wr * rougel(cs[i], cs[j]) if wr else 0.0
            M[i, j] += r + (wb * bleu(cs[i], cs[j]) if wb else 0.0)
            M[j, i] += r + (wb * bleu(cs[j], cs[i]) if wb else 0.0)
    return M


@torch.no_grad()
def embmat(model, tok, cs, layer, dev, maxlen=400):
    """Pairwise BERTScore-style F1 (greedy cosine matching of token states, no idf) between candidates.
    `layer` may be a list: one forward pass, F1 averaged over those layers."""
    ids = [tok.encode(c)[:maxlen] or [tok.eos] for c in cs]
    n, T = len(ids), max(map(len, ids))
    x = torch.full((n, T), tok.pad, dtype=torch.long)
    mask = torch.zeros((n, T), dtype=torch.bool)
    for i, s in enumerate(ids):
        x[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = True
    x, mask = x.to(dev), mask.to(dev)
    with autocast(dev):
        hs = model(x, hiddens=True)["hiddens"]
    layers = layer if isinstance(layer, (list, tuple)) else [layer]
    out = 0.0
    for l in layers:
        h = F.normalize(hs[l].float(), dim=-1)
        S = torch.einsum("itd,jsd->ijts", h, h).masked_fill(~mask[None, :, None, :], -2.0)
        P = (S.max(-1).values * mask[:, None, :]).sum(-1) / mask.sum(-1)[:, None]
        R = P.transpose(0, 1)
        out = out + 2 * P * R / (P + R).clamp_min(1e-6)
    return (out / len(layers)).cpu().numpy()


def zs(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-9)


def choose(M, bonus=None, wl=0.0):
    """MBR choice: highest expected utility against the other candidates; optionally plus wl * z(log-likelihood)."""
    n = len(M)
    eu = (M.sum(1) - np.diag(M)) / max(1, n - 1)
    score = eu if bonus is None or not wl else zs(eu) + wl * zs(bonus)
    return int(np.argmax(score))


@torch.no_grad()
def loglik(model, tok, prompt, cands, dev, ctx=2048, chunk=16):
    """Mean per-token log p(candidate | prompt) under one causal model (quality-aware decoding signal)."""
    from selfrank import seqlogp
    x = tok.encode(prompt)
    ys = [tok.encode(c) for c in cands]
    out = []
    for k in range(0, len(ys), chunk):
        with autocast(dev):
            lp, _ = seqlogp(model, tok, [x] * len(ys[k:k + chunk]), ys[k:k + chunk], ctx, dev, "mean")
        out.extend(lp.float().cpu().tolist())
    return np.asarray(out)


def runs(pc, key, single=None, ckkey=None, default="best"):
    if pc.get(key):
        return [tuple(r) for r in pc[key]]
    return [(pc[single], pc.get(ckkey, default))] if single and pc.get(single) else []


def select(cfg):
    pc = cfg.pool
    pools = [json.load(open(path(f"runs/pool/{t}.json"))) for t in pc.tags]
    prompts = loadtest(cfg, "test")
    ids = list(prompts)
    for t, p in zip(pc.tags, pools):
        assert set(p["cands"]) == set(ids), f"pool {t} does not cover the official test ids"
    allc = {i: [c for p in pools for c in p["cands"][i]] for i in ids}
    src = [t for t, p in zip(pc.tags, pools) for _ in p["cands"][ids[0]]]
    wr, wb, we, wl = (float(pc.get("wr", 1.0)), float(pc.get("wb", 0.5)), float(pc.get("we", 0.0)),
                      float(pc.get("wl", 0.0)))
    t0 = time.time()
    mats = {i: np.zeros((len(allc[i]), len(allc[i]))) for i in ids}
    if wr or wb:
        with mp.get_context("fork").Pool(min(32, os.cpu_count() or 4)) as P:
            for i, M in zip(ids, P.map(lexmat, [(allc[i], wr, wb) for i in ids], chunksize=4)):
                mats[i] += M
        print(f"lexical utility done ({time.time() - t0:.0f}s)", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = gettok(cfg)
    used = []
    embs = runs(pc, "embruns", "embrun", "embckpt")
    if we:
        for run, ck in embs:
            em, _, ep = loadpath(ckptpath(path(run), ck), dev)
            layer = list(pc.layers) if pc.get("layers") else int(pc.get("layer", 2 * len(em.blocks) // 3))
            for n, i in enumerate(ids):
                mats[i] += we / len(embs) * embmat(em, tok, allc[i], layer, dev)
                if n % 200 == 199:
                    print(f"embedding utility [{run}] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
            used.append(f"emb {ep} layer {layer}")
            del em
            torch.cuda.empty_cache()
    bonus = {i: np.zeros(len(allc[i])) for i in ids}
    lms = runs(pc, "lmruns")
    if wl:
        for run, ck in lms:
            lm, _, lp = loadpath(ckptpath(path(run), ck), dev)
            for n, i in enumerate(ids):
                bonus[i] += loglik(lm, tok, prompts[i], allc[i], dev) / len(lms)
                if n % 200 == 199:
                    print(f"log-likelihood [{run}] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
            used.append(f"lm {lp}")
            del lm
            torch.cuda.empty_cache()
    picks = {i: choose(mats[i], bonus[i], wl) for i in ids}
    preds = {i: {"text": allc[i][picks[i]]} for i in ids}
    by = Counter(src[picks[i]] for i in ids)
    starts, o = set(), 0
    for p in pools:
        if p["meta"].get("has_greedy", True):
            starts.add(o)
        o += len(p["cands"][ids[0]])
    greedy = sum(1 for i in ids if picks[i] in starts)
    out = subdir(cfg)
    writejson(out / "prediction.json", preds, 2)
    writejson(out / "genlog.json", {"pool": list(pc.tags), "candidates_per_prompt": len(src),
                                    "utility": {"wr": wr, "wb": wb, "we": we, "wl": wl}, "models_used": used,
                                    "picked_from": dict(by), "picked_greedy": greedy, "checkpoint": None,
                                    "pool_meta": [p["meta"] for p in pools], "seconds": round(time.time() - t0)}, 1)
    print(f"wrote {out / 'prediction.json'}; picked from {dict(by)}, greedy {greedy}/{len(ids)}")


def main():
    import sys
    argv = sys.argv[1:]
    if argv and argv[0] in ("gen", "select"):
        mode, argv = argv[0], argv[1:]
    else:
        mode = None
    cfg, modes = args(argv)
    mode = mode or (modes[0] if modes else None)
    if mode not in ("gen", "select"):
        raise SystemExit("usage: python pool.py gen|select phase1/configs/x.yaml [key=value ...]")
    {"gen": gen, "select": select}[mode](cfg)


if __name__ == "__main__":
    main()
