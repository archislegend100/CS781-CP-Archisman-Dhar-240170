"""python validate.py cfg [ckpt]            validation CE / perplexity per source for a checkpoint
   python validate.py cfg devset [n]       512-word prompts + 128-word references from validation documents
   python validate.py cfg score [pred]     official evaluator (BERTScore F1, ROUGE-L F1, BLEU-4) on the dev set
"""
import json
import math
import subprocess
import sys

import numpy as np
import torch

from config import args, path, rundir
from data import getdocs, loadsplit, writejson


def devset(cfg, n=500):
    split = loadsplit(cfg)
    rng = np.random.default_rng(cfg.data.splitseed)
    per = n // len(cfg.data.files)
    prompts, truth = {}, {}
    for src in cfg.data.files:
        got = 0
        for i, rid, text in getdocs(cfg, src, keys=split["valid"][src]):
            w = text.split()
            if len(w) < 640:
                continue
            s = int(rng.integers(0, len(w) - 640 + 1))
            key = f"{src}-{i}"
            prompts[key] = {"text": " ".join(w[s:s + 512]), "source": src, "id": rid}
            truth[key] = {"text": " ".join(w[s + 512:s + 640])}
            got += 1
            if got >= per:
                break
        print(f"{src:14s} {got} dev examples")
    d = path(cfg.data.dev)
    writejson(d / "prompts.json", prompts, 1)
    writejson(d / "truth.json", truth, 1)
    print(f"wrote {d}/prompts.json, truth.json ({len(truth)} examples)")


def score(cfg, pred):
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase1.py"))
    truth = path(cfg.data.dev) / "truth.json"
    r = subprocess.run([sys.executable, str(ev), "--truth", str(truth), "--prediction", str(pred)],
                       capture_output=True, text=True)
    print(r.stdout + r.stderr)
    out = {}
    for line in r.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    if out:
        writejson(path(str(pred)).with_suffix(".scores.json"), out, 1)
    return out


def main():
    cfg, modes = args()
    mode = modes[0] if modes else "best"
    if mode == "devset":
        devset(cfg, int(modes[1]) if len(modes) > 1 else 500)
        return
    if mode == "score":
        pred = modes[1] if len(modes) > 1 else rundir(cfg) / f"dev_{cfg.get('subname', cfg.name)}.json"
        score(cfg, pred)
        return
    from cache import Batches, cacheroot
    from generate import autocast, loadmodel
    from train import validstep
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, p = loadmodel(cfg, mode, dev)
    valid = Batches(cfg, "valid")
    v = validstep(model, valid, cfg, dev, autocast(dev, cfg.train.precision))
    print(f"{p} step {state['step']} | ce {v['all']:.4f} | ppl {math.exp(v['all']):.2f}")
    for s in valid.srcs:
        print(f"  {s:14s} ce {v[s]:.4f} ppl {math.exp(v[s]):.2f}")
    writejson(rundir(cfg) / f"valid_{mode}.json", {"ckpt": str(p), "step": state["step"], **v}, 1)


if __name__ == "__main__":
    main()
