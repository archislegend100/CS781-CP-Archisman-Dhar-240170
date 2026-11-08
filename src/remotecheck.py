"""Server-side checks that cannot run locally. Remote only (needs CUDA; data/cache parts are skipped if absent).

python remotecheck.py            all stages
python remotecheck.py gpu        CUDA / BF16 forward+backward+AdamW step for every enabled production config
python remotecheck.py ckpt       checkpoint save -> load -> identical logits; resume state round trip
python remotecheck.py data       official file parsing, split, cache batches
python remotecheck.py gen        2-prompt test generation with an untrained LegalGPT (pipeline only)
"""
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

from config import loadcfg, path

CONFIGS = ["starter_bpe", "legalgpt", "legalseq", "longlegal", "legalhybrid", "gpt_alibi_s", "gpt_mha_s"]


def gpu():
    from models import build, countparams
    from train import optimizer, run
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda")
    print(f"gpu {torch.cuda.get_device_name(0)} | bf16 {torch.cuda.is_bf16_supported()} | torch {torch.__version__} "
          f"| cuda {torch.version.cuda}")
    torch.backends.cuda.matmul.allow_tf32 = True
    from checkpoint import capmem
    for name in CONFIGS:
        cfg = loadcfg(f"phase1/configs/{name}.yaml")
        m, t = cfg.model, cfg.train
        capmem(t.get("maxmem"))
        model = build(m).to(dev)
        model.ckpt = bool(t.get("ckpt"))
        opt = optimizer(model, t, True)
        sp = max(int(os.environ.get("SPLIT", 1)), int(t.get("minsplit", 1)))
        while t.batch % sp:
            sp -= 1
        b = int(t.batch) // sp
        torch.cuda.reset_peak_memory_stats()
        if m.name == "legalseq":
            batch = {"src": torch.randint(4, m.vocab, (b, m.srclen)), "srcmask": torch.ones(b, m.srclen, dtype=torch.bool),
                     "ids": torch.randint(4, m.vocab, (b, m.tgtlen)), "targets": torch.randint(4, m.vocab, (b, m.tgtlen))}
        else:
            x = torch.randint(4, m.vocab, (b, m.seq + 1))
            batch = {"ids": x[:, :-1], "targets": x[:, 1:]}
        times = []
        for _ in range(3):
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = run(model, batch, dev)["loss"]
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        ok = torch.isfinite(loss).item()
        print(f"{name:14s} micro {b} {countparams(model)['total'] / 1e6:7.1f}M  loss {loss.item():.3f} (~ln V = {torch.log(torch.tensor(float(m.vocab))):.2f}) "
              f"finite {ok}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  micro-step {min(times):.3f}s")
        del model, opt
        torch.cuda.empty_cache()


def ckpt():
    from checkpoint import loadckpt, saveckpt
    from models import build
    from train import optimizer
    cfg = loadcfg("phase1/configs/legalgpt.yaml", ["model.layers=2", "debug=true"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(cfg.model).to(dev)
    opt = optimizer(model, cfg.train, dev.type == "cuda")
    x = torch.randint(4, 5000, (2, 64), device=dev)
    model(x, x)["loss"].backward()
    opt.step()
    model.eval()
    ref = model(x)["logits"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "last.pt"
        saveckpt(p, model, {"step": 7, "tokens": 123}, cfg, opt, None)
        m2 = build(cfg.model).to(dev).eval()
        o2 = optimizer(m2, cfg.train, dev.type == "cuda")
        obj = loadckpt(p, m2, o2, None, rng=True, device=dev)
        same = torch.equal(ref, m2(x)["logits"])
        st = o2.state_dict()["state"]
        print(f"checkpoint round trip: logits identical {same}, step {obj['state']['step']}, "
              f"adam state tensors {len(st)}, rng restored {'rng' in obj}, size {p.stat().st_size / 2**20:.1f} MiB")
        assert same


def data():
    import json
    from cache import Batches, cacheroot
    cfg = loadcfg("phase1/configs/legalgpt.yaml")
    import peek
    peek.structure(cfg)
    sp = path(cfg.data.split) / "meta.json"
    if sp.exists():
        meta = json.load(open(sp))
        print(f"split {meta['fp']}: " + ", ".join(f"{s} {v['train_docs']}/{v['valid_docs']}" for s, v in meta["sources"].items()))
    try:
        root = cacheroot(cfg)
        b = Batches(cfg, "train", root)
        x = b.get(0, 0, 4)
        assert x["ids"].shape == (4, cfg.model.seq) and int(x["ids"].max()) < 5000
        assert torch.equal(b.get(3, 1, 4)["ids"], Batches(cfg, "train", root).get(3, 1, 4)["ids"])
        v = next(Batches(cfg, "valid", root).fixed(cfg.data.sources[0], 4, 4))
        print(f"cache {root.name}: train batch {tuple(x['ids'].shape)}, deterministic, valid batch {tuple(v['ids'].shape)}")
        print("tokens:", {k: f"{n / 1e9:.2f}B" for k, n in b.tokens().items()}, "weights:", {k: round(w, 3) for k, w in b.w.items()})
    except SystemExit as e:
        print(f"cache not ready: {e}")


def gen():
    from generate import complete
    from models import build
    from data import loadtest
    from tokenizer import gettok
    cfg = loadcfg("phase1/configs/legalgpt.yaml")
    dev = torch.device("cuda")
    tok = gettok(cfg)
    model = build(cfg.model).to(dev).eval()
    prompts = loadtest(cfg)
    ids = list(prompts)[:2]
    t0 = time.time()
    texts, log = complete(model, tok, [tok.encode(prompts[i]) for i in ids], cfg.generation, dev, "bf16", cfg.model.seq, 0)
    print(f"untrained-model generation: words {[len(t.split()) for t in texts]}, log {log}, {time.time() - t0:.1f}s")
    assert all(len(t.split()) == cfg.generation.words for t in texts)


def main():
    stages = sys.argv[1:] or ["gpu", "ckpt", "data", "gen"]
    for s in stages:
        print(f"== {s}")
        {"gpu": gpu, "ckpt": ckpt, "data": data, "gen": gen}[s]()


if __name__ == "__main__":
    main()
