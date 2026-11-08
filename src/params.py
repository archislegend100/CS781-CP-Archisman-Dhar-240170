"""python params.py [phase1/configs/x.yaml ...]   exact parameter counts (meta device, no memory allocated)"""
import glob
import sys
from collections import Counter

import torch

from config import ROOT, loadcfg
from models import build, countparams


def breakdown(model):
    parts = Counter()
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        key = next((k for k in ("emb", "wpe", "attn", "self", "cross", "ffn", "mix", "rnn") if f".{k}." in f".{name}"), "norm/other")
        parts[key] += p.numel()
    return parts


def count(cfg):
    with torch.device("meta"):
        m = build(cfg.model)
    return countparams(m), breakdown(m)


def main():
    paths = sys.argv[1:] or [p for p in sorted(glob.glob(str(ROOT / "phase1/configs/*.yaml"))) if not p.endswith("base.yaml")]
    print(f"{'id':4s} {'config':22s} {'model':12s} {'total':>12s} {'non-emb':>12s}  range  breakdown")
    for p in paths:
        cfg = loadcfg(p)
        n, parts = count(cfg)
        ok = 50e6 <= n["total"] <= 300e6
        br = " ".join(f"{k} {v / 1e6:.1f}M" for k, v in parts.most_common())
        print(f"{cfg.id:4s} {cfg.get('subname', cfg.name)[:22]:22s} {cfg.model.name:12s} {n['total']:>12,} "
              f"{n['nonembedding']:>12,}  {'ok' if ok else 'OUT'}    {br}")


if __name__ == "__main__":
    main()
