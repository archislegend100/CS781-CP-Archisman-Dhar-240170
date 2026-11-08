"""Self-ranking post-training on official training text only.

python selfrank.py phase1/configs/selfrank.yaml pairs   sample K continuations per (x, y*), keep best/worst as (y+, y-)
python selfrank.py phase1/configs/selfrank.yaml train   L = -log s(beta * margin) + lmweight * CE(y* | x)

margin = (log p(y+|x) - log p(y-|x)), or the DPO form with a frozen copy of the base model:
margin = (log p(y+|x) - log p_ref(y+|x)) - (log p(y-|x) - log p_ref(y-|x)).
"""
import copy
import json
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from cache import Batches
from checkpoint import capmem, ckptpath, saveckpt, savebest
from config import args, path, rundir
from data import getdocs, loadsplit, writejson
from generate import autocast, complete, loadmodel
from tokenizer import gettok
from train import validstep


def lcs(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks = {}
    for j, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << j)
    s = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | s
        s = x & ~(x - ((s << 1) | 1))
    return bin(s).count("1")


def rougel(p, r):
    p, r = p.split(), r.split()
    l = lcs(p, r)
    if not l:
        return 0.0
    pp, rr = l / len(p), l / len(r)
    return 2 * pp * rr / (pp + rr)


def bleu(p, r, n=4):
    p, r = p.split(), r.split()
    logs = 0.0
    for k in range(1, n + 1):
        pc = Counter(tuple(p[i:i + k]) for i in range(len(p) - k + 1))
        rc = Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
        hit = sum(min(c, rc[g]) for g, c in pc.items())
        logs += math.log((hit + 1) / (sum(pc.values()) + 1))
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(1, len(p)))
    return bp * math.exp(logs / n)


def repetition(t, n=3):
    w = t.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(g)) / max(1, len(g))


def reward(y, ref, rc):
    return rougel(y, ref) + rc.bleu * bleu(y, ref) - rc.rep * repetition(y)


def examples(cfg, n):
    """(x, y*) from training documents, round-robin over sources, deterministic offsets."""
    rc = cfg.rank
    split = loadsplit(cfg)
    rng = np.random.default_rng(cfg.seed)
    need = rc.promptwords + rc.words
    its = {s: getdocs(cfg, s, keys=split["train"][s]) for s in cfg.data.sources}
    out = []
    while len(out) < n and its:
        for s in list(its):
            try:
                i, rid, text = next(its[s])
            except StopIteration:
                del its[s]
                continue
            w = text.split()
            if len(w) < need:
                continue
            o = int(rng.integers(0, len(w) - need + 1))
            out.append({"src": s, "doc": i, "x": " ".join(w[o:o + rc.promptwords]),
                        "ref": " ".join(w[o + rc.promptwords:o + need])})
    return out[:n]


def pairs(cfg, dev):
    rc = cfg.rank
    tok = gettok(cfg)
    model, state, p = loadmodel(fromcfg(cfg), cfg.fromckpt, dev)
    ex = examples(cfg, rc.prompts + rc.monitor)
    g = rc.sample
    m = cfg.model
    ctx = int(m.get("srclen", m.seq)) if model.family == "seq2seq" else int(m.seq)
    out = path(rc.pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for b in range(0, len(ex), g.batch):
            chunk = ex[b:b + g.batch]
            ids = [tok.encode(e["x"]) for e in chunk]
            cands = [complete(model, tok, ids, g, dev, cfg.train.precision, ctx, int(g.seed) + 1000 * k + b)[0]
                     for k in range(rc.k)]
            for j, e in enumerate(chunk):
                ys = [c[j] for c in cands]
                sc = [reward(y, e["ref"], rc) for y in ys]
                hi, lo = int(np.argmax(sc)), int(np.argmin(sc))
                rec = {**e, "cands": ys, "scores": sc, "chosen": ys[hi], "rejected": ys[lo],
                       "monitor": b + j >= rc.prompts}
                if rec["monitor"] or sc[hi] - sc[lo] >= rc.margin:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
            print(f"{min(b + g.batch, len(ex))}/{len(ex)} prompts, {kept} kept, {time.time() - t0:.0f}s", flush=True)
    writejson(out.with_suffix(".meta.json"), {"base": str(p), "base_step": state.get("step"), "k": rc.k,
                                              "prompts": len(ex), "kept": kept, "reward": "rougeL + bleu*BLEU4 - rep*rep3",
                                              "bleu": rc.bleu, "rep": rc.rep, "margin": rc.margin}, 1)


def rerank(cfg, dev):
    """Re-rank existing candidates with an embedding reward: the policy's own hidden-state F1 (BERTScore-style)
    against the gold continuation, plus optional lexical terms, minus repetition. No new sampling."""
    from pool import embmat
    rc = cfg.rank
    tok = gettok(cfg)
    model, state, p = loadmodel(fromcfg(cfg), cfg.fromckpt, dev)
    layer = int(rc.get("layer", 2 * len(model.blocks) // 3))
    src = [json.loads(l) for l in open(path(rc.pairsrc), encoding="utf-8")]
    out = path(rc.pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    rwe, rwr, rwb = float(rc.get("rwe", 1.0)), float(rc.get("rwr", 0.0)), float(rc.get("rwb", 0.0))
    kept = changed = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for n, r in enumerate(src):
            ys = r["cands"]
            E = embmat(model, tok, ys + [r["ref"]], layer, dev)
            emb = [float(v) for v in E[:-1, -1]]
            sc = [rwe * e + rwr * rougel(y, r["ref"]) + rwb * bleu(y, r["ref"]) - rc.rep * repetition(y)
                  for e, y in zip(emb, ys)]
            hi, lo = int(np.argmax(sc)), int(np.argmin(sc))
            changed += ys[hi] != r["chosen"]
            rec = {**r, "emb": emb, "scores": sc, "chosen": ys[hi], "rejected": ys[lo], "reward": "embedding"}
            if r["monitor"] or sc[hi] - sc[lo] >= rc.margin:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if n % 200 == 199:
                print(f"{n + 1}/{len(src)} re-ranked, {kept} kept ({time.time() - t0:.0f}s)", flush=True)
    writejson(out.with_suffix(".meta.json"), {"source": str(rc.pairsrc), "base": str(p), "layer": layer,
                                              "reward": f"{rwe}*EmbF1 + {rwr}*ROUGE-L + {rwb}*BLEU - {rc.rep}*rep3",
                                              "pairs": len(src), "kept": kept, "chosen_changed": changed}, 1)
    print(f"wrote {out}: {kept} pairs kept, chosen changed vs lexical reward in {changed}/{len(src)}")


def fromcfg(cfg):
    c = copy.deepcopy(cfg)
    c["run"] = cfg.fromrun
    return c


def seqlogp(model, tok, xs, ys, ctx, dev, norm="mean"):
    """Per-example log p(y | x) (sum or mean over y tokens) and mean token entropy.

    y is capped at ctx // 2 tokens and x keeps only what still fits, so degenerate samples (e.g. 128
    "words" that re-encode to thousands of byte tokens) are scored on their first ctx // 2 tokens.
    """
    ys = [y[:ctx // 2] for y in ys]
    if model.family == "seq2seq":
        S = min(ctx, max(len(x) for x in xs))
        T = max(len(y) for y in ys)
        src = torch.full((len(xs), S), tok.pad, dtype=torch.long)
        mask = torch.zeros((len(xs), S), dtype=torch.bool)
        din = torch.full((len(xs), T), tok.pad, dtype=torch.long)
        tgt = torch.full((len(xs), T), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x = x[-S:]
            src[i, :len(x)], mask[i, :len(x)] = torch.tensor(x), True
            tgt[i, :len(y)] = torch.tensor(y)
            din[i, 0] = tok.bos
            din[i, 1:len(y)] = torch.tensor(y[:-1])
        logits = model(src.to(dev), mask.to(dev), din.to(dev))["logits"]
    else:
        rows = [(x[-(ctx - len(y)):] + y, len(y)) for x, y in zip(xs, ys)]
        T = max(len(r) for r, _ in rows)
        ids = torch.full((len(rows), T), tok.pad, dtype=torch.long)
        tgt = torch.full((len(rows), T), -100, dtype=torch.long)
        for i, (r, n) in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            tgt[i, len(r) - n - 1:len(r) - 1] = torch.tensor(r[len(r) - n:])
        logits = model(ids.to(dev))["logits"]
    tgt = tgt.to(dev)
    lp = F.log_softmax(logits.float(), -1)
    keep = tgt != -100
    tok_lp = lp.gather(-1, tgt.clamp(min=0)[..., None]).squeeze(-1) * keep
    s = tok_lp.sum(-1)
    if norm == "mean":
        s = s / keep.sum(-1)
    ent = -(lp.exp() * lp).sum(-1)[keep].mean()
    return s, ent


def monitor(model, tok, mon, cfg, dev, ctx):
    g = copy.deepcopy(cfg.generation)
    texts, log = complete(model, tok, [tok.encode(e["x"]) for e in mon], g, dev, cfg.train.precision, ctx, int(g.seed))
    rep = sum(repetition(t) for t in texts) / len(texts)
    rl = sum(rougel(t, e["ref"]) for t, e in zip(texts, mon)) / len(texts)
    ntok = sum(len(tok.encode(t)) for t in texts) / len(texts)
    return {"rep3": rep, "rougeL": rl, "gen_tokens": ntok, "eospress": log["eospress"], "padded": log["padded"]}


def trainrank(cfg, dev):
    rc = cfg.rank
    tok = gettok(cfg)
    policy, state, p = loadmodel(fromcfg(cfg), cfg.fromckpt, dev)
    ref = None
    if rc.dpo:
        ref = copy.deepcopy(policy).eval()
        for q in ref.parameters():
            q.requires_grad_(False)
    policy.train()
    recs = [json.loads(l) for l in open(path(rc.pairs), encoding="utf-8")]
    data = [r for r in recs if not r["monitor"]]
    mon = [r for r in recs if r["monitor"]]
    m = cfg.model
    ctx = int(m.get("srclen", m.seq)) if policy.family == "seq2seq" else int(m.seq)
    long = {k: sum(len(tok.encode(r[k])) > ctx // 2 for r in data) for k in ("chosen", "rejected", "ref")}
    print(f"pairs {len(data)} (+{len(mon)} monitor); continuations over {ctx // 2} tokens, scored on their "
          f"first {ctx // 2}: chosen {long['chosen']}, rejected {long['rejected']}, reference {long['ref']}")
    opt = torch.optim.AdamW(policy.parameters(), lr=rc.lr, betas=(0.9, 0.95), weight_decay=0.0)
    ac = autocast(dev, cfg.train.precision)
    valid = Batches(cfg, "valid")
    vcfg = copy.deepcopy(cfg)
    vcfg.valid.windows = min(32, cfg.valid.windows)
    run = rundir(cfg)
    ck = run / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    log = open(run / "metrics.jsonl", "a")
    rng = np.random.default_rng(cfg.seed)
    base = {"valid": validstep(policy, valid, vcfg, dev, ac), **monitor(policy, tok, mon, cfg, dev, ctx)}
    print(f"base {p} | ce {base['valid']['all']:.4f} | rep3 {base['rep3']:.3f} | rougeL {base['rougeL']:.4f}")
    maxdrift = cfg.rank.get("maxdrift", 0.1)
    log.write(json.dumps({"step": 0, **base}) + "\n")
    st = {"step": 0, "base": str(p), "base_step": state.get("step"), "tokfp": tok.fp, "params": state.get("params"),
          "best": base["rougeL"] - base["rep3"], "tokens": state.get("tokens")}
    t0 = time.time()
    for step in range(rc.steps):
        lr = rc.lr * min(1.0, (step + 1) / rc.warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr
        stats = Counter()
        for _ in range(rc.accum):
            batch = [data[i] for i in rng.integers(0, len(data), rc.batch)]
            xs = [tok.encode(r["x"]) for r in batch]
            yc = [tok.encode(r["chosen"]) for r in batch]
            yr = [tok.encode(r["rejected"]) for r in batch]
            ys = [tok.encode(r["ref"]) for r in batch]
            with ac:
                lc, ent = seqlogp(policy, tok, xs, yc, ctx, dev, rc.norm)
                lr_, _ = seqlogp(policy, tok, xs, yr, ctx, dev, rc.norm)
                margin = lc - lr_
                if ref is not None:
                    with torch.no_grad():
                        margin = margin - (seqlogp(ref, tok, xs, yc, ctx, dev, rc.norm)[0]
                                           - seqlogp(ref, tok, xs, yr, ctx, dev, rc.norm)[0])
                lrank = -F.logsigmoid(rc.beta * margin).mean()
                lm = -seqlogp(policy, tok, xs, ys, ctx, dev, "mean")[0].mean() if rc.lmweight else torch.zeros((), device=dev)
                loss = (lrank + rc.lmweight * lm) / rc.accum
            loss.backward()
            stats.update({"loss": float(loss.detach()), "rank": float(lrank.detach()) / rc.accum,
                          "lm": float(lm.detach()) / rc.accum, "acc": float((margin > 0).float().mean()) / rc.accum,
                          "margin": float(margin.detach().mean()) / rc.accum, "entropy": float(ent.detach()) / rc.accum})
        if not all(math.isfinite(v) for v in stats.values()):
            raise SystemExit(f"non-finite self-ranking loss at step {step}")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        s = step + 1
        st["step"] = s
        if s % 10 == 0:
            print(f"step {s} | rank {stats['rank']:.4f} | lm {stats['lm']:.3f} | acc {stats['acc']:.2f} | "
                  f"margin {stats['margin']:.3f} | ent {stats['entropy']:.2f} | {time.time() - t0:.0f}s", flush=True)
        log.write(json.dumps({"step": s, **stats}) + "\n")
        if s % rc.every == 0 or s == rc.steps:
            v = validstep(policy, valid, vcfg, dev, ac)
            mo = monitor(policy, tok, mon, cfg, dev, ctx)
            policy.train()
            print(f"eval {s} | ce {v['all']:.4f} (base {base['valid']['all']:.4f}) | rep3 {mo['rep3']:.3f} | "
                  f"rougeL {mo['rougeL']:.4f} | gen tok {mo['gen_tokens']:.0f} | padded {mo['padded']}", flush=True)
            log.write(json.dumps({"step": s, "valid": v, **mo}) + "\n")
            log.flush()
            st.update({"valid": v, "monitor": mo})
            drift = v["all"] - base["valid"]["all"]
            if drift > maxdrift:
                print(f"warning: validation CE drifted {drift:.3f} > {maxdrift} from the base model")
            score = mo["rougeL"] - mo["rep3"]
            if score > st["best"] and drift <= maxdrift:
                st["best"] = score
                savebest(ck, policy, dict(st), cfg, 1)
                print(f"best: step {s} (rougeL - rep3 {score:.4f}, drift {drift:.3f})")
            saveckpt(ck / "last.pt", policy, dict(st), cfg, opt, None)
    log.close()
    if not (ck / "best.pt").exists():
        raise SystemExit("no checkpoint beat the base model within maxdrift; best.pt not written (keep the base submission)")


def main():
    cfg, modes = args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capmem(cfg.train.get("maxmem"))
    mode = modes[0] if modes else "pairs"
    if path(cfg.fromrun).resolve() == rundir(cfg).resolve():
        raise SystemExit("selfrank run directory must differ from the base run")
    if mode == "pairs":
        pairs(cfg, dev)
    elif mode == "train":
        trainrank(cfg, dev)
    elif mode == "rerank":
        rerank(cfg, dev)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
