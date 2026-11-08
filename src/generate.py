"""Continuation generation: official test predictions, dev-set predictions and training samples.

Prompts are sorted by token length and batched. Each batch is cut to its shortest prompt by
dropping tokens from the left, so no padding is needed (the lost tokens are the oldest context
and are logged). EOS is banned until 128 words are reached, and generation stops once a row has at
least 129 words, so the 128th word is complete.
"""
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from checkpoint import capmem, ckptpath
from config import args, path, plain, rundir, subdir, wrap
from data import loadtest, writejson


def autocast(dev, precision="bf16"):
    if dev.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def loadmodel(cfg, which, dev):
    return loadpath(ckptpath(rundir(cfg), which), dev)


def loadpath(p, dev):
    from models import build
    if not p.exists():
        raise SystemExit(f"checkpoint missing: {p}")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    m = wrap(obj["cfg"]["model"])
    model = build(m)
    model.load_state_dict(obj["model"])
    model.to(dev).eval()
    return model, obj["state"], p


class Picker:
    def __init__(self, g, B, vocab, ban, dev, seed):
        self.g = g
        self.seen = torch.zeros(B, vocab, dtype=torch.bool, device=dev)
        self.ban = torch.tensor(sorted(set(ban)), device=dev)
        self.tables = [defaultdict(set) for _ in range(B)] if g.get("norepeat", 0) else None
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed))
        self.eospress = 0

    def __call__(self, logits, hist, active, eos):
        g = self.g
        logits = logits.float()
        r = float(g.get("repetition", 1.0))
        if r != 1.0:
            logits = torch.where(self.seen, torch.where(logits > 0, logits / r, logits * r), logits)
        self.eospress += int(((logits.argmax(-1) == eos) & active).sum())
        banrows = active[:, None] & torch.zeros_like(logits, dtype=torch.bool).index_fill_(1, self.ban, True)
        logits = logits.masked_fill(banrows, float("-inf"))
        n = int(g.get("norepeat", 0))
        if n:
            for b, h in enumerate(hist):
                if len(h) >= n - 1:
                    bad = self.tables[b].get(tuple(h[len(h) - n + 1:]))
                    if bad:
                        logits[b, list(bad)] = float("-inf")
        if g.mode == "greedy":
            nxt = logits.argmax(-1)
        else:
            logits = logits / max(float(g.temperature), 1e-5)
            e = float(g.get("eps", 0.0))
            if e > 0:
                # epsilon sampling: drop tokens below probability e, always keep the most likely one
                pr = logits.softmax(-1)
                keep = (pr >= e) | (pr == pr.max(-1, keepdim=True).values)
                logits = logits.masked_fill(~keep, float("-inf"))
            k = int(g.get("topk", 0))
            if k:
                kth = logits.topk(k, -1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            p = float(g.get("topp", 1.0))
            if p < 1.0:
                sl, si = logits.sort(-1, descending=True)
                cum = sl.softmax(-1).cumsum(-1)
                drop = cum - sl.softmax(-1) > p
                logits = logits.masked_fill(drop.scatter(1, si, drop), float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=self.gen).squeeze(-1)
        self.seen[torch.arange(len(nxt)), nxt] = True
        return nxt

    def update(self, hist):
        n = int(self.g.get("norepeat", 0))
        if n:
            for b, h in enumerate(hist):
                if len(h) >= n:
                    self.tables[b][tuple(h[-n:-1])].add(h[-1])


@torch.no_grad()
def genbatch(model, tok, prompts, g, dev, precision, ctx, seed, need):
    """Generate for one batch of prompt-id lists until every row has `need` words or maxnew tokens."""
    B = len(prompts)
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    model = models[0]
    seq2seq = model.family == "seq2seq"
    maxnew = int(g.maxnew)
    ac = autocast(dev, precision)
    if seq2seq:
        L = min(len(p) for p in prompts)
        L = min(L, ctx)
        src = torch.tensor([p[-L:] for p in prompts], device=dev)
        mask = torch.ones_like(src, dtype=torch.bool)
        with ac:
            enc, _ = model.encode(src, mask)
        cache = model.newcache()
        step = lambda ids: model.decode(ids, enc, mask, cache, last=True)[0]
        with ac:
            logits = step(torch.full((B, 1), tok.bos, device=dev))[:, -1]
    else:
        L = min(min(len(p) for p in prompts), ctx - maxnew)
        x = torch.tensor([p[-L:] for p in prompts], device=dev)
        caches = [mm.newcache() for mm in models]

        def step(ids):
            if len(models) == 1:
                return model(ids, cache=caches[0], last=True)["logits"]
            # ensemble: average of member log-probabilities (product of experts)
            return torch.stack([F.log_softmax(mm(ids, cache=c, last=True)["logits"].float(), -1)
                                for mm, c in zip(models, caches)]).mean(0)
        with ac:
            logits = step(x)[:, -1]
    cut = sum(len(p) - L for p in prompts)
    pick = Picker(g, B, logits.size(-1), [tok.eos, tok.pad, tok.bos, tok.unk], dev, seed)
    hist = [[] for _ in range(B)]
    active = torch.ones(B, dtype=torch.bool, device=dev)
    for t in range(maxnew):
        nxt = pick(logits, hist, active, tok.eos)
        for b, i in enumerate(nxt.tolist()):
            hist[b].append(i)
        pick.update(hist)
        if t % 8 == 7 or t == maxnew - 1:
            for b in range(B):
                if active[b] and len(tok.decode(hist[b]).split()) >= need[b]:
                    active[b] = False
            if not active.any():
                break
        with ac:
            logits = step(nxt[:, None])[:, -1]
    return hist, {"cut": cut, "eospress": pick.eospress, "steps": t + 1}


def complete(model, tok, prompts, g, dev, precision="bf16", ctx=2048, seed=0):
    """Return exactly-`words` continuations (text only, prompt excluded) for prompt-id lists."""
    words = int(g.words)
    gen = [[] for _ in prompts]
    log = {"cut": 0, "eospress": 0, "rounds": 0, "extended": 0, "padded": 0}
    todo = list(range(len(prompts)))
    for r in range(int(g.get("rounds", 3))):
        if not todo:
            break
        need = [words + 1 - len(tok.decode(gen[i]).split()) for i in todo]
        need = [max(2, n) for n in need]
        hist, info = genbatch(model, tok, [prompts[i] + gen[i] for i in todo], g, dev, precision, ctx, seed + r, need)
        for i, h in zip(todo, hist):
            gen[i] += h
        log["cut"] += info["cut"] if r == 0 else 0
        log["eospress"] += info["eospress"]
        log["rounds"] = r + 1
        if r:
            log["extended"] += len(todo)
        todo = [i for i in todo if len(tok.decode(gen[i]).split()) < words]
    out = []
    for i in range(len(prompts)):
        w = tok.decode(gen[i]).split()
        if len(w) < words:
            log["padded"] += 1
            fill = w or tok.decode(prompts[i]).split() or ["the"]
            w = (w + fill * (words // len(fill) + 1))[:words]
        out.append(" ".join(w[:words]))
    return out, log


def mbrpick(cands, wbleu=0.5):
    """Minimum-Bayes-risk choice: the candidate with the highest mean ROUGE-L + wbleu * BLEU against the others."""
    from selfrank import bleu, rougel
    best, bi = -1.0, 0
    for i, a in enumerate(cands):
        s = sum(rougel(a, b) + wbleu * bleu(a, b) for j, b in enumerate(cands) if j != i)
        if s > best:
            best, bi = s, i
    return bi


def predict(cfg, which="test", ckpt=None, dev=None, model=None, tok=None, limit=None):
    from tokenizer import gettok
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capmem(cfg.train.get("maxmem"))
    g = cfg.generation
    tok = tok or gettok(cfg)
    state, cp = {}, None
    if model is None:
        model, state, cp = loadmodel(cfg, ckpt or g.get("ckpt", "best"), dev)
        if state.get("tokfp") not in (None, tok.fp):
            raise SystemExit(f"checkpoint tokenizer {state['tokfp']} != configured tokenizer {tok.fp}")
    members = []
    for r in g.get("ensemble") or []:
        em, es, ep = loadpath(ckptpath(path(r), g.get("ensckpt", "last")), dev)
        if es.get("tokfp") not in (None, tok.fp) or em.family != "causal" or model.family != "causal":
            raise SystemExit(f"ensemble member {ep} must be a causal model with the same tokenizer")
        members.append(em)
        print(f"ensemble member: {ep} (step {es.get('step')})")
    prompts = loadtest(cfg, which)
    ids = list(prompts)[:limit] if limit else list(prompts)
    enc = {i: tok.encode(prompts[i]) for i in ids}
    m = cfg.model
    ctx = int(m.get("srclen", m.seq)) if model.family == "seq2seq" else int(m.get("ctx", m.seq))
    runner = [model] + members if members else model
    nmbr = int(g.get("mbr", 0))
    gs = wrap({**plain(g), "mode": "sample", "temperature": g.get("mbrtemp", 0.7), "topp": g.get("mbrtopp", 0.95)})
    chosen = 0
    order = sorted(ids, key=lambda i: len(enc[i]))
    preds, logs = {}, []
    t0 = time.time()
    bs = int(g.batch)
    for k in range(0, len(order), bs):
        chunk = order[k:k + bs]
        pr = cfg.train.get("precision", "bf16")
        texts, log = complete(runner, tok, [enc[i] for i in chunk], g, dev, pr, ctx, int(g.seed) + k)
        if nmbr:
            cands = [texts] + [complete(runner, tok, [enc[i] for i in chunk], gs, dev, pr, ctx,
                                        int(g.seed) + k + 7919 * (j + 1))[0] for j in range(nmbr)]
            picks = [mbrpick([c[b] for c in cands], float(g.get("mbrbleu", 0.5))) for b in range(len(chunk))]
            chosen += sum(p == 0 for p in picks)
            texts = [cands[p][b] for b, p in enumerate(picks)]
        preds.update({i: {"text": t} for i, t in zip(chunk, texts)})
        logs.append(log)
        print(f"{min(k + bs, len(order))}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    preds = {i: preds[i] for i in ids}
    summary = {k: sum(l[k] for l in logs) for k in ("cut", "eospress", "extended", "padded")}
    summary.update({"prompts": len(ids), "seconds": round(time.time() - t0, 1), "checkpoint": str(cp) if cp else None,
                    "step": state.get("step"), "tokens": state.get("tokens"),
                    "prompt_tokens_mean": sum(len(v) for v in enc.values()) / max(1, len(enc)),
                    "ensemble": [str(r) for r in g.get("ensemble") or []], "mbr": nmbr,
                    "mbr_main_chosen": chosen if nmbr else None})
    return preds, summary


def main():
    cfg, modes = args()
    which = modes[0] if modes else "test"
    ckpt = modes[1] if len(modes) > 1 else None
    torch.backends.cuda.matmul.allow_tf32 = True
    preds, summary = predict(cfg, which, ckpt)
    if which == "test":
        out = subdir(cfg)
        writejson(out / "prediction.json", preds, indent=2)
        writejson(out / "genlog.json", summary, indent=1)
        print(f"wrote {out / 'prediction.json'}")
    else:
        out = rundir(cfg) / f"dev_{cfg.get('subname', cfg.name)}.json"
        writejson(out, preds, indent=2)
        writejson(out.with_suffix(".log.json"), summary, indent=1)
        print(f"wrote {out}")
    if torch.cuda.is_available():
        summary["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        writejson((subdir(cfg) / "genlog.json") if which == "test" else out.with_suffix(".log.json"), summary, indent=1)
    print(json.dumps(summary))
    if summary["padded"]:
        print(f"warning: {summary['padded']} predictions padded to {cfg.generation.words} words")


if __name__ == "__main__":
    main()
