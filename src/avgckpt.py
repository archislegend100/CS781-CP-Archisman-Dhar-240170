"""Average model weights of several checkpoints of one run (late-checkpoint averaging, cf. LAWA).

python avgckpt.py runs/legalhybrid_sc step_500 best last      -> runs/legalhybrid_sc/ckpt/avg.pt
Then: python generate.py <cfg> test avg   (or generation.ckpt=avg)
"""
import sys
from pathlib import Path

import torch

from checkpoint import ckptpath
from config import path


def main():
    run, names = path(sys.argv[1]), sys.argv[2:]
    objs = [torch.load(ckptpath(run, n), map_location="cpu", weights_only=False) for n in names]
    ref = objs[-1]
    for o in objs:
        assert o["cfg"]["model"] == ref["cfg"]["model"], "checkpoints come from different architectures"
    avg = {k: sum(o["model"][k].float() for o in objs) / len(objs) for k in ref["model"]}
    avg = {k: v.to(ref["model"][k].dtype) for k, v in avg.items()}
    state = dict(ref["state"], averaged=[f"{n} (step {o['state'].get('step')})" for n, o in zip(names, objs)])
    out = run / "ckpt" / "avg.pt"
    tmp = out.with_name("avg.pt.tmp")
    torch.save({"model": avg, "state": state, "cfg": ref["cfg"]}, tmp)
    tmp.replace(out)
    print(f"averaged {', '.join(state['averaged'])} -> {out}")


if __name__ == "__main__":
    main()
