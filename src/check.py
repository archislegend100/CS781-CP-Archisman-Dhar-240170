"""Local mathematical / integration checks on tiny synthetic configs.

No dataset, cache, checkpoint or optimizer is touched; nothing here says anything about model quality.
python check.py
"""
import glob
import json
import math
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from config import ROOT, loadcfg, wrap
from models import build, countparams
from models.common import (Attention, alibi, alibislopes, causalmask, localattn, lmloss, rope, Rope, sdpa, shift,
                           SwiGLU)
from models.legalhybrid import fftconv

torch.manual_seed(0)
V = 97
TINY = {
    "starter": dict(name="starter", vocab=V, dim=32, layers=2, qheads=4, ctx=64, seq=64, dropout=0.0),
    "gpt_rope": dict(name="legalgpt", vocab=V, dim=32, layers=2, qheads=4, kvheads=2, ff=48, seq=64, pos="rope"),
    "gpt_alibi": dict(name="legalgpt", vocab=V, dim=48, layers=2, qheads=6, kvheads=2, ff=64, seq=64, pos="alibi"),
    "gpt_mha": dict(name="legalgpt", vocab=V, dim=32, layers=2, qheads=4, kvheads=4, ff=48, seq=64, pos="rope"),
    "seq": dict(name="legalseq", vocab=V, dim=32, enclayers=2, declayers=2, qheads=4, kvheads=2, ff=48,
                srclen=24, tgtlen=16, seq=24),
    "long": dict(name="longlegal", vocab=V, dim=32, layers=4, qheads=4, kvheads=2, ff=48, seq=64, window=5,
                 globalevery=2),
    "hybrid": dict(name="legalhybrid", vocab=V, dim=32, layers=4, qheads=4, kvheads=2, ff=48, seq=64, attnevery=2, ks=3),
    "gru": dict(name="legalgru", vocab=V, dim=32, layers=2, seq=64),
}
CAUSAL = ["starter", "gpt_rope", "gpt_alibi", "gpt_mha", "long", "hybrid", "gru"]
RESULTS = []


def model(k):
    m = build(wrap(TINY[k])).double()
    m.eval()
    return m


def ids(B=2, T=23):
    return torch.randint(4, V, (B, T))


def close(a, b, tol=1e-9):
    return torch.allclose(a, b, atol=tol, rtol=0)


def check(name):
    def deco(f):
        def run():
            try:
                f()
                RESULTS.append((name, True, ""))
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
        run.__name__ = f.__name__
        CHECKS.append(run)
        return run
    return deco


CHECKS = []


@check("01 model construction (all families)")
def _():
    for k in TINY:
        assert isinstance(model(k), torch.nn.Module), k


@check("02/03/19 logits, hidden, embedding and LM-head shapes")
def _():
    for k in CAUSAL:
        m = model(k)
        x = ids()
        out = m(x, hiddens=k != "gru")
        d = TINY[k]["dim"]
        assert out["logits"].shape == (2, 23, V), (k, out["logits"].shape)
        assert out["hidden"].shape == (2, 23, d), k
        assert m.emb.weight.shape == (V, d) and m.head.weight.shape == (V, d), k
        if k != "gru":
            assert len(out["hiddens"]) == len(m.blocks) + 1 and all(h.shape == (2, 23, d) for h in out["hiddens"])


@check("04 q/k/v projection dimensions")
def _():
    a = Attention(48, 6, 2)
    assert a.hd == 8 and a.q.weight.shape == (48, 48) and a.kv.weight.shape == (2 * 2 * 8, 48)
    assert a.o.weight.shape == (48, 48)


@check("05/06 head-dim divisibility and GQA qheads % kvheads == 0 enforced by config check")
def _():
    from config import check as cfgcheck
    for bad in (dict(dim=30, qheads=4, kvheads=2), dict(dim=32, qheads=4, kvheads=3), dict(dim=36, qheads=4, pos="rope")):
        try:
            cfgcheck(wrap({"model": {**bad, "kvheads": bad.get("kvheads", 4)}}))
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


@check("07 GQA attention output equals explicit softmax(QK^T/sqrt(d))V with shared KV heads")
def _():
    B, Hq, Hk, T, d = 2, 6, 2, 9, 8
    q, k, v = torch.randn(B, Hq, T, d).double(), torch.randn(B, Hk, T, d).double(), torch.randn(B, Hk, T, d).double()
    y = sdpa(q, k, v, causal=True)
    kk, vv = k.repeat_interleave(3, 1), v.repeat_interleave(3, 1)
    s = q @ kk.transpose(-1, -2) / math.sqrt(d)
    s = s.masked_fill(~causalmask(0, T, T, "cpu"), float("-inf"))
    assert y.shape == (B, Hq, T, d) and close(y, s.softmax(-1) @ vv)


@check("08 RoPE dims, norm preservation and relative-position property")
def _():
    r = Rope(8)
    cos, sin = r(0, 12, "cpu")
    assert cos.shape == (12, 4)
    x = torch.randn(1, 1, 12, 8).double()
    y = rope(x, (cos.double(), sin.double()))
    assert close(y.norm(dim=-1), x.norm(dim=-1), 1e-6)
    q, k = torch.randn(8).double(), torch.randn(8).double()

    def dot(m, n):
        cm, sm = r(m, 1, "cpu")
        cn, sn = r(n, 1, "cpu")
        return (rope(q[None, None, None], (cm.double(), sm.double())) * rope(k[None, None, None], (cn.double(), sn.double()))).sum()
    assert abs(dot(7, 3) - dot(14, 10)) < 1e-5 and abs(dot(7, 3) - dot(5, 3)) > 1e-3


@check("09 ALiBi slopes and bias shape / causal structure")
def _():
    s = alibislopes(8)
    assert torch.allclose(s, torch.tensor([2.0 ** -(i + 1) for i in range(8)]))
    assert len(alibislopes(12)) == 12
    b = alibi(s, 3, 4, 7, "cpu", torch.float32)
    assert b.shape == (1, 8, 4, 7)
    assert torch.isinf(b[0, :, 0, 4:]).all() and (b[0, :, 0, :4] <= 0).all()
    assert abs(float(b[0, 0, 0, 0]) - (-0.5 * 3)) < 1e-6


@check("10 causal-mask shape and bottom-right alignment under a KV cache")
def _():
    m = causalmask(5, 3, 8, "cpu")
    assert m.shape == (3, 8) and m[0, :6].all() and not m[0, 6:].any() and m[2].all()
    w = causalmask(5, 3, 8, "cpu", window=3)
    assert w[0].tolist() == [False, False, False, True, True, True, False, False]


@check("11 padding-mask broadcasting: encoder output at real positions ignores padding content")
def _():
    m = model("seq")
    src = ids(2, 12)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 8:] = False
    a, _ = m.encode(src, mask)
    src2 = src.clone()
    src2[1, 8:] = torch.randint(4, V, (4,))
    b, _ = m.encode(src2, mask)
    assert close(a[0], b[0]) and close(a[1, :8], b[1, :8])


@check("12/13/14 seq2seq encoder, decoder and cross-attention dimensions")
def _():
    m = model("seq")
    src, din = ids(2, 12), ids(2, 7)
    mask = torch.ones(2, 12, dtype=torch.bool)
    out = m(src, mask, din, hiddens=True)
    assert out["enc"].shape == (2, 12, 32) and out["hidden"].shape == (2, 7, 32) and out["logits"].shape == (2, 7, V)
    assert len(out["enchiddens"]) == 3 and len(out["hiddens"]) == 3
    c = m.dec[0].cross
    assert c.q.weight.shape == (32, 32) and c.kv.weight.shape == (2 * 2 * 8, 32)


@check("15/16 SwiGLU and residual dimensions")
def _():
    f = SwiGLU(32, 48)
    assert f.up.weight.shape == (96, 32) and f.down.weight.shape == (32, 48)
    x = torch.randn(2, 5, 32)
    assert f(x).shape == x.shape
    for k in ("gpt_rope", "hybrid", "long"):
        blk = model(k).blocks[0]
        ex = model(k).extras(5, 0, "cpu", torch.float64)
        assert blk(x.double(), ex).shape == x.shape


@check("17 causal short conv keeps length; FFT long conv equals direct causal convolution")
def _():
    m = model("hybrid")
    mix = m.blocks[0].mix
    x = torch.randn(2, 11, 32).double()
    assert mix(x).shape == x.shape
    z = torch.randn(2, 96, 11).double()
    assert mix.short(F.pad(z, (2, 0))).shape == z.shape
    u, h = torch.randn(2, 4, 11).double(), torch.randn(4, 11).double()
    direct = torch.stack([(u[..., :t + 1] * h[:, :t + 1].flip(-1)).sum(-1) for t in range(11)], -1)
    assert close(fftconv(u, h), direct, 1e-9)


@check("18 LongLegal blockwise local attention equals dense windowed attention (incl. first block, ragged T)")
def _():
    for T, w in ((23, 5), (20, 5), (7, 8), (16, 4)):
        q, k, v = torch.randn(2, 4, T, 8).double(), torch.randn(2, 2, T, 8).double(), torch.randn(2, 2, T, 8).double()
        dense = sdpa(q, k, v, mask=causalmask(0, T, T, "cpu", window=w))
        assert close(localattn(q, k, v, w), dense, 1e-9), (T, w)


@check("20 CE label shifting and -100 masking")
def _():
    x = ids(2, 6)
    t = shift(x)
    assert (t[:, :-1] == x[:, 1:]).all() and (t[:, -1] == -100).all()
    logits = torch.randn(2, 6, V).double()
    manual = -F.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).mean()
    assert abs(float(lmloss(logits, t)) - float(manual)) < 1e-9
    from cache import Batches
    ns = SimpleNamespace(seq2seq=False)
    b = Batches.collate(ns, [np.arange(7), np.arange(10, 17)])
    assert b["ids"].shape == (2, 6) and (b["targets"] == b["ids"] + 1).all()
    ns = SimpleNamespace(seq2seq=True, srclen=6, tgtlen=5, pad=0, bos=2)
    b = Batches.collate(ns, [(np.array([5, 6, 7]), np.array([8, 9, 3])), (np.arange(10, 16), np.arange(20, 25))])
    assert b["ids"][0].tolist() == [2, 8, 9, 0, 0] and b["targets"][0].tolist() == [8, 9, 3, -100, -100]
    assert b["srcmask"][0].tolist() == [True] * 3 + [False] * 3 and b["ids"][1].tolist() == [2, 20, 21, 22, 23]


def gptformula(V, d, L, qh, kvh, ff, tie=True):
    hd = d // qh
    layer = d * qh * hd + d * 2 * kvh * hd + qh * hd * d + 3 * d * ff + 2 * d
    return L * layer + V * d * (1 if tie else 2) + d


@check("21/22 parameter-count formula and tied-weight counting")
def _():
    for q, kv, ff in ((12, 4, 2048), (12, 12, 1728)):
        c = wrap(dict(name="legalgpt", vocab=5000, dim=768, layers=20, qheads=q, kvheads=kv, ff=ff, seq=2048))
        with torch.device("meta"):
            n = countparams(build(c))["total"]
        assert n == gptformula(5000, 768, 20, q, kv, ff), (n, gptformula(5000, 768, 20, q, kv, ff))
    with torch.device("meta"):
        tied = countparams(build(wrap({**TINY["gpt_rope"], "tie": True})))["total"]
        untied = countparams(build(wrap({**TINY["gpt_rope"], "tie": False})))["total"]
        gpt2 = countparams(build(wrap(dict(name="starter", vocab=50257, dim=560, layers=8, qheads=8, ctx=2048))))["total"]
    assert untied - tied == V * 32
    d, L = 560, 8
    assert gpt2 == 50257 * d + 2048 * d + L * (12 * d * d + 13 * d) + 2 * d == 59_455_760


@check("23 no NaN/Inf in tiny forward passes")
def _():
    for k in CAUSAL:
        assert torch.isfinite(model(k)(ids())["logits"]).all(), k
    m = model("seq")
    assert torch.isfinite(m(ids(2, 12), torch.ones(2, 12, dtype=torch.bool), ids(2, 7))["logits"]).all()


@check("65 causality: changing a future token leaves earlier logits unchanged")
def _():
    for k in CAUSAL:
        m = model(k)
        x = ids(2, 23)
        for t in (0, 9, 17):
            y = x.clone()
            y[:, t + 1:] = torch.randint(4, V, y[:, t + 1:].shape)
            a, b = m(x)["logits"], m(y)["logits"]
            assert close(a[:, :t + 1], b[:, :t + 1], 1e-9), (k, t)
            assert not close(a[:, t + 1:], b[:, t + 1:], 1e-9), (k, t)
    m = model("seq")
    src, mask = ids(2, 12), torch.ones(2, 12, dtype=torch.bool)
    x = ids(2, 9)
    y = x.clone()
    y[:, 5:] = torch.randint(4, V, (2, 4))
    a, b = m(src, mask, x)["logits"], m(src, mask, y)["logits"]
    assert close(a[:, :5], b[:, :5], 1e-9) and not close(a[:, 5:], b[:, 5:], 1e-9)


@check("KV-cache / recurrent-state incremental decoding equals full forward")
def _():
    for k in CAUSAL:
        m = model(k)
        x = ids(2, 19)
        full = m(x)["logits"]
        cache = m.newcache()
        outs = [m(x[:, :11], cache=cache)["logits"]]
        for t in range(11, 19):
            outs.append(m(x[:, t:t + 1], cache=cache)["logits"])
        assert close(torch.cat(outs, 1), full, 1e-8), k
    m = model("seq")
    src, mask, din = ids(2, 12), torch.ones(2, 12, dtype=torch.bool), ids(2, 8)
    full = m(src, mask, din)["logits"]
    enc, _ = m.encode(src, mask)
    cache = m.newcache()
    outs = [m.decode(din[:, t:t + 1], enc, mask, cache)[0] for t in range(8)]
    assert close(torch.cat(outs, 1), full, 1e-8)


@check("LongLegal local layers see exactly `window` keys; global layers see all")
def _():
    m = model("long")
    assert m.kinds == ["local", "global", "local", "global"]
    assert [b.window for b in m.blocks] == [5, 0, 5, 0]
    x = ids(1, 30)
    cache = m.newcache()
    m(x, cache=cache)
    assert cache["layers"][0]["k"].size(2) == 5 and cache["layers"][1]["k"].size(2) == 30


@check("tokenizer: Assignment-1 BPE loads, specials, fingerprint, round trip")
def _():
    from tokenizer import BPE, SPECIALS, selfcheck
    cfg = loadcfg("phase1/configs/legalgpt.yaml")
    t = BPE.load(cfg.tokenizer.path)
    assert t.vocab == 5000 and len(t.merges) == 4740 and t.itos[:4] == SPECIALS
    assert (t.pad, t.unk, t.bos, t.eos) == (0, 1, 2, 3)
    assert t.fp == cfg.tokenizer.fp == BPE.load(cfg.tokenizer.path).fp
    assert selfcheck(t, verbose=False)


@check("generation: exactly 128 words, prompt excluded, EOS banned until done (random tiny model)")
def _():
    from generate import complete
    from tokenizer import BPE
    tok = BPE.load("assets/bpe_en.json")
    m = build(wrap({**TINY["gpt_rope"], "vocab": 5000, "seq": 700})).eval()
    prompt = "The appellant was convicted under Section 302 of the Indian Penal Code and sentenced"
    g = wrap(dict(mode="sample", temperature=1.0, topk=50, topp=0.95, repetition=1.1, norepeat=3, words=128,
                  maxnew=400, rounds=3, seed=1))
    texts, log = complete(m, tok, [tok.encode(prompt), tok.encode(prompt + " to life")], g, torch.device("cpu"),
                          "fp32", 700, 0)
    assert all(len(t.split()) == 128 for t in texts), [len(t.split()) for t in texts]
    assert not any(t.startswith(prompt[:40]) for t in texts)
    g2 = wrap({**g, "mode": "greedy"})
    a, _ = complete(m, tok, [tok.encode(prompt)], g2, torch.device("cpu"), "fp32", 700, 0)
    b, _ = complete(m, tok, [tok.encode(prompt)], g2, torch.device("cpu"), "fp32", 700, 5)
    assert a == b


@check("submission validator on tiny fake data; zip has prediction.json at root")
def _():
    from submit import validate, zipit
    prompts = {"1": "alpha beta gamma " * 200, "2": "delta epsilon " * 300}
    good = {i: {"text": " ".join(["word"] * 128)} for i in prompts}
    assert validate(good, prompts) == []
    assert any("missing" in e for e in validate({"1": good["1"]}, prompts))
    assert any("unknown" in e for e in validate({**good, "3": good["1"]}, prompts))
    assert any("127 words" in e for e in validate({**good, "2": {"text": " ".join(["w"] * 127)}}, prompts))
    assert any("empty" in e for e in validate({**good, "2": {"text": "  "}}, prompts))
    echo = {"text": " ".join(prompts["1"].split()[:128])}
    assert any("prompt" in e for e in validate({**good, "1": echo}, prompts))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "prediction.json"
        p.write_text(json.dumps(good))
        zipit(p, Path(d) / "submission.zip")
        with zipfile.ZipFile(Path(d) / "submission.zip") as z:
            assert z.namelist() == ["prediction.json"]


@check("score table parsing (in-memory)")
def _():
    from scores import table
    s = {"G1": [{"bert": 0.8, "rouge": 0.2, "bleu": 0.03, "date": "2026-10-02", "notes": "x"}],
         "B1": [{"bert": 0.7, "rouge": 0.1, "bleu": 0.01, "date": "2026-10-02", "notes": ""}]}
    out = table(s, {"G1": {"name": "legalgpt", "category": "architecture"}})
    lines = out.splitlines()
    assert lines[1].startswith("G1") and lines[2].startswith("B1")


@check("data: reader formats, malformed records, normalisation keeps legal text")
def _():
    from data import clean, records, unpack
    t = "Sec. 302 IPC,\r\n(2014) 5 SCC 1;\x00 Art. 226(1)(a)   r/w  O. XXI"
    c = clean(t)
    assert c == "Sec. 302 IPC,\n(2014) 5 SCC 1; Art. 226(1)(a) r/w O. XXI"
    recs = [{"id": "a", "text": "x y"}, {"text": 5}, "bare", None]
    assert [unpack(r) for r in recs] == [("a", "x y"), (None, None), (None, "bare"), (None, None)]
    with tempfile.TemporaryDirectory() as d:
        for name, body in (("a.json", json.dumps(recs)), ("b.json", json.dumps({"k1": {"text": "t"}})),
                           ("c.jsonl", "\n".join(json.dumps(r) for r in recs[:2]))):
            p = Path(d) / name
            p.write_text(body)
            got = list(records(p, "json"))
            assert len(got) == {"a.json": 4, "b.json": 1, "c.jsonl": 2}[name]
        assert list(records(Path(d) / "b.json", "json")) == [("k1", {"text": "t"})]


@check("cache: tiny shard stream across boundaries, incomplete shard removal")
def _():
    from cache import Stream, tidy, _save
    from data import writejson
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        docs = [np.array([5, 6, 3]), np.array([7, 8, 9, 3]), np.array([10, 3])]
        for s, part in enumerate((docs[:2], docs[2:])):
            base = d / f"shard_{s:05d}"
            arr = np.concatenate(part).astype(np.uint16)
            idx = np.concatenate([[0], np.cumsum([len(x) for x in part])]).astype(np.int64)
            _save(base.with_suffix(".bin"), arr)
            _save(base.with_suffix(".idx"), idx)
            writejson(base.with_suffix(".json"), {"tokens": int(arr.size), "docs": len(part), "dtype": "uint16"})
        (d / "shard_00002.bin").write_bytes(b"\x00\x01")
        assert len(tidy(d)) == 2 and not (d / "shard_00002.bin").exists()
        writejson(d / "part.json", {})
        st = Stream(d)
        assert st.n == 9 and st.get(5, 4).tolist() == [9, 3, 10, 3]
        assert st.doc(4) == (3, 7) and st.doc(8) == (7, 9)


@check("mixing: temperature p_i = n_i^a / sum n_j^a, natural, balanced")
def _():
    from cache import mixture
    n = {"a": 100.0, "b": 400.0}
    w = mixture(wrap({"data": {"mixing": "temperature", "alpha": 0.5}}), n)
    assert abs(w["a"] - 10 / 30) < 1e-12
    assert abs(mixture(wrap({"data": {"mixing": "natural"}}), n)["b"] - 0.8) < 1e-12
    assert mixture(wrap({"data": {"mixing": "balanced"}}), n)["a"] == 0.5


@check("lr schedule: warmup, cosine to minlr, linear, constant")
def _():
    from train import lrat
    t = wrap(dict(lr=1.0, minlr=0.1, warmup=10, sched="cosine"))
    assert abs(lrat(0, t, 110) - 0.1) < 1e-12 and abs(lrat(9, t, 110) - 1.0) < 1e-12
    assert abs(lrat(110, t, 110) - 0.1) < 1e-12 and abs(lrat(60, t, 110) - 0.55) < 1e-12
    assert abs(lrat(60, wrap({**t, "sched": "linear"}), 110) - 0.55) < 1e-12
    assert lrat(80, wrap({**t, "sched": "constant"}), 110) == 1.0


@check("all configs parse; production models within [50M, 300M]")
def _():
    for p in sorted(glob.glob(str(ROOT / "phase1/configs/*.yaml"))):
        if p.endswith("base.yaml"):
            continue
        cfg = loadcfg(p)
        with torch.device("meta"):
            n = countparams(build(cfg.model))["total"]
        assert 50e6 <= n <= 300e6, (p, n)


@check("experiments.yaml: ids match configs, params match exact counts, enabled ids unique")
def _():
    import yaml
    reg = yaml.safe_load(open(ROOT / "phase1/experiments.yaml"))["experiments"]
    ids = [e["id"] for e in reg]
    assert len(ids) == len(set(ids))
    z = [i for i, e in enumerate(reg) if e["category"] == "self-ranking"]
    assert z and z == list(range(len(reg) - len(z), len(reg))), "self-ranking entries must come last"
    for e in reg:
        cfg = loadcfg(e["config"])
        assert cfg.id == e["id"], (e["id"], cfg.id)
        if isinstance(e.get("params"), int):
            with torch.device("meta"):
                assert countparams(build(cfg.model))["total"] == e["params"], e["id"]


@check("self-ranking objective: -log sigmoid(beta * margin), local ROUGE-L matches official formula")
def _():
    from selfrank import rougel, repetition
    ev = ROOT / "evaluation_script" / "evaluate_phase1.py"
    if ev.exists():
        src = ev.read_text()
        ns = {}
        exec(src[:src.index("try:")] + src[src.index("class DuplicateJSONKeyError"):src.index("def main")], ns)
        for a, b in (("a b c d e", "a c e f"), ("x y z", "x y z"), ("p q", "r s")):
            assert abs(rougel(a, b) - ns["rouge_l_f1"](a, b)) < 1e-12
    beta, m = 0.1, torch.tensor([2.0, -1.0])
    assert torch.allclose(-F.logsigmoid(beta * m), torch.log1p(torch.exp(-beta * m)))
    assert repetition("a b c a b c a b c") > 0.5 and repetition("a b c d e f") == 0.0


def main():
    for c in CHECKS:
        c()
    bad = [r for r in RESULTS if not r[1]]
    for name, ok, msg in RESULTS:
        print(f"{'pass' if ok else 'FAIL'}  {name}")
        if not ok:
            print("      " + msg.replace("\n", "\n      "))
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
