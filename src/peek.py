"""Remote data inspection (named peek.py because inspect.py would shadow the stdlib module).

python peek.py cfg            structure of every train file + test file, tokens/word on a sample
python peek.py cfg full       stream every file: record counts, malformed records, word totals
python peek.py cfg overlap    how much of each test prompt occurs verbatim in the training files
"""
import itertools
import json
import statistics
import time

from config import args, path
from data import clean, firstchar, getdocs, loadtest, records, srcfile, unpack
from tokenizer import gettok


def structure(cfg, n=3):
    tok = gettok(cfg)
    for src in cfg.data.files:
        p = srcfile(cfg, src)
        if not p.exists():
            print(f"{src}: missing {p}")
            continue
        print(f"== {src}: {p.name} {p.stat().st_size / 2**30:.2f} GiB, top-level {firstchar(p)!r}")
        recs = list(itertools.islice(records(p, "ijson" if p.stat().st_size > 2**30 else "json"), 200))
        for r in recs[:n]:
            body = r[1] if isinstance(r, tuple) else r
            keys = {k: type(v).__name__ for k, v in body.items()} if isinstance(body, dict) else type(body).__name__
            rid, text = unpack(r, cfg.data.textkey, cfg.data.idkey)
            print(f"   id {rid!r} keys {keys} text {len(text or '')} chars: {(text or '')[:120]!r}")
        words = [len(unpack(r)[1].split()) for r in recs if unpack(r)[1]]
        ntok = sum(len(tok.encode(clean(unpack(r)[1]))) for r in recs[:50] if unpack(r)[1])
        nw = sum(words[:50])
        print(f"   first {len(recs)} docs: median {statistics.median(words or [0]):.0f} words, "
              f"{ntok / max(1, nw):.3f} tok/word ({tok.kind})")
    t = path(cfg.data.root) / cfg.data.test
    if t.exists():
        d = json.load(open(t))
        w = [len(v["text"].split()) for v in d.values()]
        ids = list(d)
        print(f"== test: {len(d)} prompts, ids {ids[:3]}..{ids[-2:]}, words min {min(w)} median "
              f"{statistics.median(w):.0f} max {max(w)}, keys {sorted(next(iter(d.values())))}")
        sample = [v["text"] for v in list(d.values())[:100]]
        nt = [len(tok.encode(s)) for s in sample]
        print(f"   prompt tokens ({tok.kind}): median {statistics.median(nt):.0f}, max {max(nt)}")


def full(cfg):
    for src in cfg.data.files:
        st, n, w, t0 = {}, 0, 0, time.time()
        for _, _, text in getdocs(cfg, src, stats=st, raw=True):
            n += 1
            w += len(text.split())
        print(f"{src:14s} {n:>10,} docs {w:>15,} words  malformed {st.get('malformed', 0):,} {st.get('examples', [])[:2]} "
              f"({time.time() - t0:.0f}s)")


def overlap(cfg, n=8, stride=8):
    """Index every word 8-gram of every test prompt; probe training text at every 8th position.
    Any shared span of >= 15 words is guaranteed to be hit."""
    prompts = loadtest(cfg)
    index = {}
    for k, text in prompts.items():
        w = text.split()
        for i in range(len(w) - n + 1):
            index.setdefault(hash(tuple(w[i:i + n])), set()).add(k)
    hits = {k: 0 for k in prompts}
    for src in cfg.get("overlap") or list(cfg.data.files):
        t0 = time.time()
        for _, _, text in getdocs(cfg, src):
            w = text.split()
            for i in range(0, len(w) - n + 1, stride):
                ks = index.get(hash(tuple(w[i:i + n])))
                if ks:
                    for k in ks:
                        hits[k] += 1
        print(f"{src} scanned ({time.time() - t0:.0f}s)")
    probes = {k: max(1, (len(v.split()) - n + 1) // stride) for k, v in prompts.items()}
    frac = sorted(hits[k] / probes[k] for k in prompts)
    print(f"prompts with any 8-gram hit: {sum(1 for v in hits.values() if v)}/{len(hits)}")
    print(f"hit fraction quartiles: {frac[len(frac) // 4]:.3f} {frac[len(frac) // 2]:.3f} {frac[3 * len(frac) // 4]:.3f}")
    path("runs").mkdir(exist_ok=True)
    json.dump(hits, open(path("runs/overlap.json"), "w"))


def main():
    cfg, modes = args()
    mode = modes[0] if modes else "structure"
    {"structure": structure, "full": full, "overlap": overlap}[mode](cfg)


if __name__ == "__main__":
    main()
