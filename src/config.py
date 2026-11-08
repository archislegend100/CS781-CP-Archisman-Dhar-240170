import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        v = dict.get(self, k, default)
        return default if v is None else v


def wrap(x):
    if isinstance(x, dict):
        return Cfg({k: wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [wrap(v) for v in x]
    return x


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [plain(v) for v in x]
    return x


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def readyaml(p):
    p = path(p)
    y = yaml.safe_load(open(p)) or {}
    base = y.pop("base", None)
    if base:
        y = merge(readyaml(p.parent / base), y)
    return y


def scalar(v):
    """yaml.safe_load, plus floats like 5e-6 that YAML 1.1 would leave as strings."""
    x = yaml.safe_load(v)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            pass
    return x


def setkey(d, key, val):
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = val


def loadcfg(p, overrides=()):
    y = readyaml(p)
    for o in overrides:
        k, v = o.split("=", 1)
        setkey(y, k, scalar(v))
    y.setdefault("cfgpath", str(path(p).relative_to(ROOT)) if path(p).is_relative_to(ROOT) else str(p))
    cfg = wrap(y)
    check(cfg)
    return cfg


def check(cfg):
    m = cfg.get("model")
    if not m:
        return
    if "dim" in m and "qheads" in m:
        assert m.dim % m.qheads == 0, f"dim {m.dim} not divisible by qheads {m.qheads}"
        kv = m.get("kvheads", m.qheads)
        assert m.qheads % kv == 0, f"qheads {m.qheads} not divisible by kvheads {kv}"
        if m.get("pos", "rope") == "rope":
            assert (m.dim // m.qheads) % 2 == 0, "rope needs even head dim"


def args(argv=None):
    """python x.py cfg.yaml [mode ...] [a.b=val ...]"""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: script.py phase1/configs/x.yaml [mode] [key=value ...]")
    over = [a for a in argv[1:] if "=" in a]
    modes = [a for a in argv[1:] if "=" not in a]
    return loadcfg(argv[0], over), modes


def rundir(cfg):
    return path(cfg.get("run", f"runs/{cfg.name}"))


def subdir(cfg):
    return path(f"submissions/{cfg.id}_{cfg.get('subname', cfg.name)}")


def dump(cfg):
    return json.dumps(plain(cfg), sort_keys=True, default=str)


def fingerprint(obj, n=16):
    s = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:n]
