"""Official Phase-1 data: download, streaming reader, normalisation, document-level split.

Records are normalised internally to (index, id, text). `index` is the position in the
source file and is the stable key used by the split and the cache. `id` is the original ID
when the record has one.
"""
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from config import args, fingerprint, path

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
HSPACE = re.compile(r"[ \t  - 　]+")
BLANKS = re.compile(r"\n{3,}")


def clean(text, ws=True):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CTRL.sub(" ", text)
    if ws:
        text = HSPACE.sub(" ", text)
        text = BLANKS.sub("\n\n", text)
    return text.strip()


def availram():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 << 30


def firstchar(p):
    with open(p, "rb") as f:
        while True:
            c = f.read(1)
            if not c or not c.isspace():
                return c.decode()


def records(p, reader="auto"):
    """Yield raw records from a JSON array, a JSON object {id: record} or JSONL."""
    p = Path(p)
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if reader == "auto":
        reader = "json" if p.stat().st_size * 5 < availram() else "ijson"
    obj = firstchar(p) == "{"
    if reader == "json":
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, v in items:
            yield (str(k), v) if obj else v
        return
    import ijson
    with open(p, "rb") as f:
        it = ijson.kvitems(f, "") if obj else ijson.items(f, "item")
        for x in it:
            yield (str(x[0]), x[1]) if obj else x


def unpack(rec, textkey="text", idkey="id"):
    rid = None
    if isinstance(rec, tuple):
        rid, rec = rec
    if isinstance(rec, str):
        return rid, rec
    if not isinstance(rec, dict):
        return rid, None
    text = rec.get(textkey)
    if rid is None and rec.get(idkey) is not None:
        rid = str(rec[idkey])
    return rid, text if isinstance(text, str) else None


def srcfile(cfg, src):
    return path(cfg.data.root) / cfg.data.files[src]


def getdocs(cfg, src, keys=None, stats=None, raw=False):
    """Yield (index, id, text) for one source; `keys` restricts to a set of indices."""
    d = cfg.data
    ws = d.get("normalize", {}).get("whitespace", True)
    for i, rec in enumerate(records(srcfile(cfg, src), d.get("reader", "auto"))):
        if keys is not None and i not in keys:
            continue
        rid, text = unpack(rec, d.get("textkey", "text"), d.get("idkey", "id"))
        if text is None or not text.strip():
            if stats is not None:
                stats["malformed"] = stats.get("malformed", 0) + 1
                if len(stats.setdefault("examples", [])) < 5:
                    stats["examples"].append({"index": i, "id": rid, "type": type(rec).__name__})
            continue
        yield i, rid if rid is not None else str(i), text if raw else clean(text, ws)


def filefp(p, n=1 << 20):
    p = Path(p)
    h = hashlib.sha256(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(n))
        if p.stat().st_size > n:
            f.seek(-n, 2)
            h.update(f.read(n))
    return h.hexdigest()[:16]


def datafp(cfg):
    return fingerprint({s: filefp(srcfile(cfg, s)) for s in sorted(cfg.data.files)})


def loadtest(cfg, which="test"):
    p = path(cfg.data.root) / cfg.data.test if which == "test" else path(cfg.data.dev) / "prompts.json"
    if not p.exists():
        raise SystemExit(f"prompt file missing: {p}")
    d = json.load(open(p, encoding="utf-8"))
    return {str(k): v["text"] for k, v in d.items()}


def writejson(p, obj, indent=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def bucket(text, seed):
    h = hashlib.blake2b(f"{seed}\x00{text}".encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def makesplit(cfg):
    """Document-level split over all four files (independent of which sources a run trains on).

    Hashing the text rather than the position keeps exact duplicates on one side of the split.
    """
    d = cfg.data
    out = {"train": {}, "valid": {}}
    meta = {"seed": d.splitseed, "ratio": d.valid, "hash": "blake2b-64(seed|text)",
            "normalize": d.get("normalize", {}), "dataset": datafp(cfg), "files": {}, "sources": {}}
    for src in d.files:
        t0 = time.time()
        st = {}
        tr, va = [], []
        wtr = wva = 0
        for i, rid, text in getdocs(cfg, src, stats=st):
            n = len(text.split())
            if bucket(text, d.splitseed) < d.valid:
                va.append([i, rid])
                wva += n
            else:
                tr.append([i, rid])
                wtr += n
        total = len(tr) + len(va) + st.get("malformed", 0)
        bad = st.get("malformed", 0)
        meta["files"][src] = {"path": str(d.files[src]), "fp": filefp(srcfile(cfg, src))}
        meta["sources"][src] = {"records": total, "malformed": bad, "malformed_examples": st.get("examples", []),
                                "train_docs": len(tr), "valid_docs": len(va),
                                "train_words": wtr, "valid_words": wva}
        print(f"{src:14s} docs {total:>9,}  train {len(tr):>9,}  valid {len(va):>7,}  "
              f"malformed {bad:,}  words {wtr + wva:,}  ({time.time() - t0:.0f}s)")
        if total and bad / total > d.get("maxbad", 0.01):
            raise SystemExit(f"{src}: {bad}/{total} malformed records; inspect with peek.py before splitting")
        out["train"][src], out["valid"][src] = tr, va
    root = path(d.split)
    writejson(root / "train.json", out["train"])
    writejson(root / "valid.json", out["valid"])
    meta["fp"] = fingerprint({"dataset": meta["dataset"], "seed": d.splitseed, "ratio": d.valid,
                              "normalize": meta["normalize"]})
    writejson(root / "meta.json", meta, indent=1)
    print(f"split fp {meta['fp']} -> {root}")


def loadsplit(cfg, check=True):
    root = path(cfg.data.split)
    if not (root / "meta.json").exists():
        raise SystemExit(f"split missing: {root}/meta.json  (run: python data.py {cfg.cfgpath} split)")
    meta = json.load(open(root / "meta.json"))
    if check:
        for src in cfg.data.files:
            fp = filefp(srcfile(cfg, src))
            if meta["files"][src]["fp"] != fp:
                raise SystemExit(f"{src}: data file changed since split ({fp} != {meta['files'][src]['fp']})")
    out = {"meta": meta}
    for name in ("train", "valid"):
        d = json.load(open(root / f"{name}.json"))
        out[name] = {s: {i for i, _ in d[s]} for s in d}
    return out


def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    d = cfg.data
    files = list_repo_files(d.repo, repo_type="dataset")
    phase = [f for f in files if f.startswith("Phase_1")]
    print("remote Phase-1 files:", *phase, sep="\n  ")
    want = list(d.files.values()) + [d.test]
    for f in want:
        if f not in files:
            base = Path(f).name.lower().replace(" ", "")
            hits = [x for x in phase if Path(x).name.lower().replace(" ", "") == base]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix data.files/data.test in the config")
            print(f"note: {f} -> {hits[0]}")
            f = hits[0]
        hf_hub_download(d.repo, f, repo_type="dataset", local_dir=str(path(d.root)))
        print(f"ok {f}")


def main():
    cfg, modes = args()
    mode = modes[0] if modes else "split"
    if mode == "download":
        download(cfg)
    elif mode == "split":
        makesplit(cfg)
    elif mode == "fp":
        print(datafp(cfg))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
