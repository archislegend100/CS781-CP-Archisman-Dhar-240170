import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from config import plain


def capmem(gib):
    """Hard per-process GPU memory cap (GiB), e.g. 15 to stay within a 16 GB Colab/Kaggle T4/P100."""
    if not gib or not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(gib) * 2 ** 30 / total), 0)
    return float(gib)


def unwrap(model):
    return getattr(model, "_orig_mod", model)


def rngstate():
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def setrng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def saveckpt(p, model, state, cfg, opt=None, scaler=None, full=True):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {"model": unwrap(model).state_dict(), "state": state, "cfg": plain(cfg)}
    if full:
        obj["opt"] = opt.state_dict()
        obj["sched"] = {"kind": cfg.train.get("sched", "cosine"), "step": state["step"]}
        obj["scaler"] = scaler.state_dict() if scaler is not None and scaler.is_enabled() else None
        obj["rng"] = rngstate()
    tmp = p.with_name(p.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, p)


def loadckpt(p, model=None, opt=None, scaler=None, rng=False, device="cpu"):
    obj = torch.load(p, map_location=device, weights_only=False)
    if model is not None:
        unwrap(model).load_state_dict(obj["model"])
    if opt is not None:
        opt.load_state_dict(obj["opt"])
    if scaler is not None and obj.get("scaler"):
        scaler.load_state_dict(obj["scaler"])
    if rng and "rng" in obj:
        setrng(obj["rng"])
    return obj


def ckptpath(run, which="best"):
    run = Path(run) / "ckpt"
    if which in ("best", "last"):
        return run / f"{which}.pt"
    if str(which).isdigit():
        return run / f"step_{int(which)}.pt"
    named = run / f"{which}.pt"
    return named if named.exists() else Path(which)


def stepof(p):
    m = re.search(r"_(\d+)\.pt$", str(p))
    return int(m.group(1)) if m else -1


def retain(d, keep, keepbest):
    """Delete old step_* and best_* checkpoints. last.pt and best.pt are never touched."""
    d = Path(d)
    for pat, n in (("step_*.pt", keep), ("best_*.pt", max(0, keepbest - 1))):
        files = sorted(d.glob(pat), key=stepof)
        for f in files[:max(0, len(files) - n)]:
            f.unlink()


def savebest(d, model, state, cfg, keepbest):
    import json
    d = Path(d)
    best, info = d / "best.pt", d / "best.json"
    if keepbest > 1 and best.exists() and info.exists():
        os.replace(best, d / f"best_{json.load(open(info))['step']}.pt")
    saveckpt(best, model, state, cfg, full=False)
    tmp = info.with_name("best.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, default=str)
    os.replace(tmp, info)
