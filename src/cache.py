"""Sharded, resumable token cache and batch sampling.

cache/<fp>/<split>/<source>/shard_00000.{bin,idx,doc,json}
  bin   token ids (uint16 when vocab <= 65535), each document followed by EOS
  idx   int64 document start offsets within the shard (ndocs + 1)
  doc   int64 source-file index of every document
  json  written last; a shard without it is incomplete and is removed on resume
<fp> = hash(tokenizer fp, split fp). All models sharing tokenizer + split share one cache.
"""
import json
import multiprocessing as mp
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from config import args, fingerprint, path
from data import getdocs, loadsplit, writejson

VERSION = 1
_tok = None


def cachefp(tokfp, splitfp, eos):
    return fingerprint({"tok": tokfp, "split": splitfp, "eos": eos, "v": VERSION})


def cacheroot(cfg, tok=None):
    if tok is None:
        from tokenizer import gettok
        tok = gettok(cfg)
    m = path(cfg.data.split) / "meta.json"
    if not m.exists():
        raise SystemExit(f"split missing: {m}  (run: python data.py {cfg.cfgpath} split)")
    meta = json.load(open(m))
    return path(cfg.data.cache) / cachefp(tok.fp, meta["fp"], tok.eos)


def _init(tcfg):
    global _tok
    from tokenizer import BPE, HFTok
    _tok = HFTok(text=tcfg["json"]) if tcfg["kind"] == "hf" else BPE.load(tcfg["path"])


def _encode(texts):
    lens, out = [], []
    for t in texts:
        ids = _tok.encode(t)
        ids.append(_tok.eos)
        out.extend(ids)
        lens.append(len(ids))
    return np.asarray(out, dtype=np.int64), np.asarray(lens, dtype=np.int64)


def _save(p, arr):
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        arr.tofile(f)
    tmp.replace(p)


def shardok(js):
    m = json.load(open(js))
    b = js.with_suffix(".bin")
    return b.exists() and b.stat().st_size == m["tokens"] * np.dtype(m["dtype"]).itemsize


def tidy(d):
    """Remove temporary files and incomplete shards; return the complete shards."""
    for f in d.glob("*.tmp"):
        f.unlink()
    done = []
    for js in sorted(d.glob("shard_*.json")):
        if shardok(js) and len(done) == int(js.stem.split("_")[1]):
            done.append(js)
        else:
            break
    keep = {j.stem for j in done}
    for f in d.glob("shard_*"):
        if f.stem not in keep:
            f.unlink()
    return done


def buildpart(cfg, keys, src, out, dtype, tok):
    out.mkdir(parents=True, exist_ok=True)
    fin = out / "part.json"
    if fin.exists():
        return json.load(open(fin))
    done = tidy(out)
    skip = json.load(open(done[-1]))["doc_end"] if done else 0
    shard = len(done)
    d = cfg.data
    size, chunk, workers = int(d.shard), int(d.get("chunk", 512)), int(d.get("workers", 8))
    tcfg = {"kind": tok.kind, "path": str(path(cfg.tokenizer.get("path", ""))),
            "json": tok.tok.to_str() if tok.kind == "hf" else None}
    buf, lens, docs = [], [], []
    nbuf = 0
    ndoc = skip
    t0 = time.time()

    def flush():
        nonlocal buf, lens, docs, nbuf, shard
        if not lens:
            return
        base = out / f"shard_{shard:05d}"
        arr = np.concatenate(buf).astype(dtype)
        ln = np.concatenate(lens)
        idx = np.zeros(len(ln) + 1, dtype=np.int64)
        np.cumsum(ln, out=idx[1:])
        _save(base.with_suffix(".bin"), arr)
        _save(base.with_suffix(".idx"), idx)
        _save(base.with_suffix(".doc"), np.asarray(docs, dtype=np.int64))
        writejson(base.with_suffix(".json"), {"tokens": int(arr.size), "docs": len(docs), "dtype": np.dtype(dtype).name,
                                              "doc_start": ndoc - len(docs), "doc_end": ndoc})
        print(f"  {out.parent.name}/{src} shard {shard} {arr.size / 1e6:.1f}M tok {len(docs)} docs "
              f"({time.time() - t0:.0f}s)", flush=True)
        buf, lens, docs, nbuf = [], [], [], 0
        shard += 1

    def chunks():
        texts, ids = [], []
        for n, (i, _, text) in enumerate(getdocs(cfg, src, keys=keys)):
            if n < skip:
                continue
            texts.append(text)
            ids.append(i)
            if len(texts) == chunk:
                yield ids, texts
                texts, ids = [], []
        if texts:
            yield ids, texts

    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(tcfg,)) as pool:
        pending = deque()

        def drain():
            nonlocal nbuf, ndoc
            ids, res = pending.popleft()
            arr, ln = res.get()
            buf.append(arr)
            lens.append(ln)
            docs.extend(ids)
            nbuf += arr.size
            ndoc += len(ids)
            if nbuf >= size:
                flush()

        for ids, texts in chunks():
            pending.append((ids, pool.apply_async(_encode, (texts,))))
            if len(pending) >= 2 * workers:
                drain()
        while pending:
            drain()
    flush()
    shards = sorted(out.glob("shard_*.json"))
    info = {"shards": len(shards), "docs": sum(json.load(open(s))["docs"] for s in shards),
            "tokens": sum(json.load(open(s))["tokens"] for s in shards)}
    writejson(fin, info)
    return info


def buildcache(cfg):
    from tokenizer import gettok
    tok = gettok(cfg)
    split = loadsplit(cfg)
    root = cacheroot(cfg, tok)
    dtype = np.uint16 if tok.vocab <= 65535 else np.uint32
    print(f"cache {root}\ntok {tok.kind} {tok.fp} vocab {tok.vocab} dtype {np.dtype(dtype).name}")
    for name in ("valid", "train"):
        for src in cfg.data.sources:
            info = buildpart(cfg, split[name][src], src, root / name / src, dtype, tok)
            print(f"{name:5s} {src:14s} {info['docs']:>9,} docs {info['tokens']:>14,} tok")
    writemeta(root, tok, split["meta"]["fp"], dtype)


def writemeta(root, tok, splitfp, dtype):
    parts = {}
    for fin in sorted(root.glob("*/*/part.json")):
        parts.setdefault(fin.parent.parent.name, {})[fin.parent.name] = json.load(open(fin))
    writejson(root / "meta.json", {"tok": tok.fp, "tokkind": tok.kind, "vocab": tok.vocab, "eos": tok.eos,
                                   "split": splitfp, "dtype": np.dtype(dtype).name, "version": VERSION,
                                   "parts": parts}, indent=1)


def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


class Stream:
    """All complete shards of one (split, source) as a single token array."""

    def __init__(self, d):
        d = Path(d)
        js = sorted(d.glob("shard_*.json"))
        if not js or not (d / "part.json").exists():
            raise SystemExit(f"cache part missing or incomplete: {d}  (run cache.py)")
        self.arrs, offs, starts = [], [0], []
        for j in js:
            m = json.load(open(j))
            self.arrs.append(np.memmap(j.with_suffix(".bin"), dtype=m["dtype"], mode="r", shape=(m["tokens"],)))
            starts.append(np.fromfile(j.with_suffix(".idx"), dtype=np.int64)[:-1] + offs[-1])
            offs.append(offs[-1] + m["tokens"])
        self.offs = np.asarray(offs, dtype=np.int64)
        self.n = int(self.offs[-1])
        self.starts = np.append(np.concatenate(starts), self.n)

    def get(self, i, n):
        out = np.empty(n, dtype=np.int64)
        k = int(np.searchsorted(self.offs, i, side="right") - 1)
        got = 0
        while got < n:
            a = self.arrs[k]
            j = i + got - int(self.offs[k])
            m = min(n - got, a.size - j)
            out[got:got + m] = a[j:j + m]
            got += m
            k += 1
        return out

    def doc(self, i):
        d = int(np.searchsorted(self.starts, i, side="right") - 1)
        return int(self.starts[d]), int(self.starts[d + 1])


def mixture(cfg, sizes):
    d = cfg.data
    srcs = list(sizes)
    n = np.asarray([sizes[s] for s in srcs], dtype=np.float64)
    mode = d.get("mixing", "natural")
    if mode == "natural":
        w = n
    elif mode == "balanced":
        w = np.ones_like(n)
    elif mode == "temperature":
        w = n ** float(d.alpha)
    elif mode == "weights":
        w = np.asarray([float(d.weights[s]) for s in srcs])
    else:
        raise ValueError(mode)
    return dict(zip(srcs, w / w.sum()))


class Batches:
    """Deterministic random windows. Batch (step, micro) depends only on the seed, so resume is exact."""

    def __init__(self, cfg, split="train", root=None):
        self.cfg = cfg
        root = Path(root or cacheroot(cfg))
        self.streams = {s: Stream(root / split / s) for s in cfg.data.sources}
        self.w = mixture(cfg, {s: st.n for s, st in self.streams.items()})
        self.srcs = list(self.w)
        self.p = np.asarray([self.w[s] for s in self.srcs])
        self.seed = int(cfg.seed)
        m = cfg.model
        self.seq2seq = m.name == "legalseq"
        self.seq = int(m.seq)
        if self.seq2seq:
            self.srclen, self.tgtlen, self.srcmin = int(m.srclen), int(m.tgtlen), int(m.get("srcmin", 64))
        from tokenizer import gettok
        tok = gettok(cfg)
        self.pad, self.bos, self.eos = tok.pad, tok.bos, tok.eos

    def tokens(self):
        return {s: st.n for s, st in self.streams.items()}

    def lm(self, st, rng):
        i = int(rng.integers(0, st.n - self.seq - 1))
        return st.get(i, self.seq + 1)

    def pair(self, st, rng):
        for _ in range(16):
            ds, de = st.doc(int(rng.integers(0, st.n)))
            if de - ds >= self.srcmin + 8:
                break
        lo = ds + max(1, min(self.srcmin, de - ds - 1))
        cut = int(rng.integers(lo, max(lo + 1, de - 8)))
        src = st.get(max(ds, cut - self.srclen), cut - max(ds, cut - self.srclen))
        tgt = st.get(cut, min(de, cut + self.tgtlen) - cut)
        return src, tgt

    def collate(self, rows):
        if not self.seq2seq:
            a = torch.from_numpy(np.stack(rows))
            return {"ids": a[:, :-1].contiguous(), "targets": a[:, 1:].contiguous()}
        b = len(rows)
        src = torch.full((b, self.srclen), self.pad, dtype=torch.long)
        mask = torch.zeros((b, self.srclen), dtype=torch.bool)
        din = torch.full((b, self.tgtlen), self.pad, dtype=torch.long)
        tgt = torch.full((b, self.tgtlen), -100, dtype=torch.long)
        for r, (s, t) in enumerate(rows):
            src[r, :len(s)] = torch.from_numpy(s)
            mask[r, :len(s)] = True
            t = torch.from_numpy(t)
            tgt[r, :len(t)] = t
            din[r, 0] = self.bos
            din[r, 1:len(t)] = t[:-1]
        return {"src": src, "srcmask": mask, "ids": din, "targets": tgt}

    def get(self, step, micro, batch):
        rng = np.random.default_rng([self.seed, int(step), int(micro)])
        srcs = rng.choice(len(self.srcs), size=batch, p=self.p)
        rows = []
        for k in srcs:
            st = self.streams[self.srcs[k]]
            rows.append(self.pair(st, rng) if self.seq2seq else self.lm(st, rng))
        return self.collate(rows)

    def fixed(self, src, count, batch):
        """Deterministic validation batches for one source."""
        st = self.streams[src]
        rng = np.random.default_rng([self.seed, 7919, self.srcs.index(src)])
        rows = []
        if self.seq2seq:
            rows = [self.pair(st, rng) for _ in range(count)]
        else:
            span = max(1, (st.n - self.seq - 1) // max(1, count))
            rows = [st.get(k * span, self.seq + 1) for k in range(count) if k * span + self.seq + 1 <= st.n]
        for i in range(0, len(rows), batch):
            yield self.collate(rows[i:i + batch])


def main():
    cfg, modes = args()
    mode = modes[0] if modes else "build"
    if mode == "build":
        buildcache(cfg)
    elif mode == "list":
        for d in sorted(path(cfg.data.cache).glob("*")):
            m = d / "meta.json"
            info = json.load(open(m)) if m.exists() else {}
            print(f"{d.name}  {du(d) / 2**30:7.2f} GiB  tok {info.get('tok', '?')}  split {info.get('split', '?')}  "
                  f"parts {sum(len(v) for v in info.get('parts', {}).values())}")
    elif mode == "clean":
        for d in path(cfg.data.cache).glob("*/*/*"):
            if d.is_dir() and not (d / "part.json").exists():
                n = len(tidy(d))
                print(f"{d}: {n} complete shards kept, temporaries removed")
    elif mode == "drop":
        fp = modes[1]
        d = path(cfg.data.cache) / fp
        if d.exists():
            shutil.rmtree(d)
            print(f"removed {d}")
    elif mode == "info":
        root = cacheroot(cfg)
        b = Batches(cfg)
        print(root)
        for s, n in b.tokens().items():
            print(f"{s:14s} {n:>14,} tok  weight {b.w[s]:.3f}")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
