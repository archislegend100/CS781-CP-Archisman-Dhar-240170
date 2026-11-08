"""python train.py phase1/configs/legalgpt.yaml [resume|fresh] [key=value ...]"""
import atexit
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from cache import Batches, cacheroot
from checkpoint import capmem, ckptpath, loadckpt, retain, saveckpt, savebest
from config import args, dump, path, plain, rundir
from data import writejson
from generate import autocast, complete
from models import build, countparams
from tokenizer import gettok

LOW, HIGH = 50_000_000, 300_000_000


def lrat(step, t, steps):
    if step < t.warmup:
        return t.lr * (step + 1) / t.warmup
    if t.sched == "constant":
        return t.lr
    frac = min(1.0, (step - t.warmup) / max(1, steps - t.warmup))
    if t.sched == "linear":
        return t.minlr + (t.lr - t.minlr) * (1 - frac)
    return t.minlr + 0.5 * (t.lr - t.minlr) * (1 + math.cos(math.pi * frac))


def seqtokens(m):
    return int(m.srclen) + int(m.tgtlen) if m.name == "legalseq" else int(m.seq)


def numsteps(cfg, batches):
    t = cfg.train
    per = t.batch * t.accum * seqtokens(cfg.model)
    if t.get("steps"):
        return int(t.steps)
    if t.get("epochs"):
        return math.ceil(t.epochs * sum(batches.tokens().values()) / per)
    return math.ceil(float(t.tokens) / per)


def optimizer(model, t, cuda):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t.wd}, {"params": rest, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t.lr, betas=tuple(t.betas), eps=t.eps, fused=cuda)


def run(model, b, dev):
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    if model.family == "seq2seq":
        return model(b["src"], b["srcmask"], b["ids"], b["targets"])
    return model(b["ids"], b["targets"])


def trainstep(model, opt, scaler, batches, step, cfg, dev, ac):
    t = cfg.train
    total = torch.zeros((), device=dev)
    ntgt = 0
    for micro in range(t.accum):
        b = batches.get(step, micro, t.batch)
        ntgt += int((b["targets"] != -100).sum())
        with ac:
            loss = run(model, b, dev)["loss"] / t.accum
        scaler.scale(loss).backward()
        total += loss.detach()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t.clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return float(total), float(gn), ntgt


@torch.no_grad()
def validstep(model, valid, cfg, dev, ac):
    model.eval()
    v = cfg.valid
    out, S, N = {}, 0.0, 0
    for src in valid.srcs:
        s = n = 0
        for b in valid.fixed(src, v.windows, v.batch):
            with ac:
                loss = float(run(model, b, dev)["loss"])
            k = int((b["targets"] != -100).sum())
            s += loss * k
            n += k
        out[src] = s / max(1, n)
        S, N = S + s, N + n
    model.train()
    out["all"] = S / max(1, N)
    out["macro"] = sum(out[s] for s in valid.srcs) / len(valid.srcs)
    return out


def samples(model, tok, valid, cfg, dev, step, out):
    """Small fixed generation set: one 512-word-ish prompt per validation source."""
    model.eval()
    g = cfg.generation
    prompts, refs = [], []
    for src in valid.srcs[: cfg.train.get("samples", 4)]:
        st = valid.streams[src]
        ids = st.get(0, min(st.n, 1600)).tolist()
        text = tok.decode(ids).split()
        prompts.append(tok.encode(" ".join(text[:512])))
        refs.append(" ".join(text[512:640]))
    m = cfg.model
    ctx = int(m.get("srclen", m.seq)) if model.family == "seq2seq" else int(m.get("ctx", m.seq))
    texts, _ = complete(model, tok, prompts, g, dev, cfg.train.precision, ctx, int(g.seed))
    writejson(out / f"step_{step}.json", [{"source": s, "ref": r, "gen": t} for s, r, t in zip(valid.srcs, refs, texts)], 1)
    model.train()


def lockrun(run_dir):
    """One trainer per run directory, so two GPU queues can never train the same run at once."""
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "train.lock"
    if p.exists():
        try:
            pid = int(p.read_text().split()[0])
            os.kill(pid, 0)
            raise SystemExit(f"{run_dir} is already being trained by pid {pid}; not starting a second copy")
        except (ProcessLookupError, ValueError, IndexError):
            p.unlink(missing_ok=True)
        except PermissionError:
            raise SystemExit(f"{run_dir} is locked by another user's process")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{run_dir} was just locked by another trainer")
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    atexit.register(lambda: p.exists() and p.read_text().split()[:1] == [str(os.getpid())] and p.unlink())


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def main():
    cfg, modes = args()
    if cfg.get("notrain"):
        raise SystemExit(f"{cfg.cfgpath} is a decoding-only config (run: {cfg.run}); use generate.py")
    t = cfg.train
    split = max(int(t.get("split", 1)), int(t.get("minsplit", 1)))
    while t.batch % split:
        split -= 1
    t.batch, t.accum = t.batch // split, t.accum * split
    run_dir = rundir(cfg)
    ck = run_dir / "ckpt"
    resume = "resume" in modes
    if (ck / "last.pt").exists() and not resume and "fresh" not in modes:
        raise SystemExit(f"{run_dir} already has checkpoints; pass 'resume' or 'fresh'")
    lockrun(run_dir)
    torch.manual_seed(cfg.seed)
    cuda = torch.cuda.is_available()
    cap = capmem(t.get("maxmem"))
    dev = torch.device("cuda" if cuda else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = gettok(cfg)
    assert tok.vocab <= cfg.model.vocab, f"tokenizer vocab {tok.vocab} > model vocab {cfg.model.vocab}"
    root = cacheroot(cfg, tok)
    model = build(cfg.model).to(dev)
    model.ckpt = bool(t.get("ckpt", False))
    n = countparams(model)
    if not (LOW <= n["total"] <= HIGH) and not cfg.get("debug"):
        raise SystemExit(f"{n['total']:,} parameters outside the Phase-1 range [50M, 300M]")
    batches, valid = Batches(cfg, "train", root), Batches(cfg, "valid", root)
    steps = numsteps(cfg, batches)
    per = t.batch * t.accum * seqtokens(cfg.model)
    opt = optimizer(model, t, cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=t.precision == "fp16" and cuda)
    ac = autocast(dev, t.precision)
    fwd = torch.compile(model) if t.get("compile") else model

    state = {"step": 0, "tokens": 0, "target_tokens": 0, "elapsed": 0.0, "best": float("inf"), "valid": None,
             "params": n["total"], "tokfp": tok.fp, "cachefp": root.name, "steps": steps}
    if t.get("init") and not resume:
        obj = torch.load(path(t.init), map_location="cpu", weights_only=False)
        if obj["cfg"]["model"] != plain(cfg.model) or obj["state"].get("tokfp") != tok.fp:
            raise SystemExit(f"init checkpoint {t.init} has a different model config or tokenizer")
        model.load_state_dict(obj["model"])
        state["init"] = {"from": str(t.init), "step": obj["state"].get("step"), "tokens": obj["state"].get("tokens")}
        print(f"init: weights from {t.init} (step {obj['state'].get('step')}, {fmt(obj['state'].get('tokens') or 0)} tok); "
              f"fresh optimizer and schedule")
        del obj
    if resume:
        obj = loadckpt(ck / "last.pt", model, opt, scaler, rng=True, device=dev)
        old = obj["cfg"]
        if old["model"] != plain(cfg.model) or obj["state"]["tokfp"] != tok.fp or obj["state"]["cachefp"] != root.name:
            raise SystemExit("resume refused: model config, tokenizer or cache differs from checkpoint")
        state.update(obj["state"])
        state["steps"] = steps
        del obj
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(dump(cfg))
    metrics = open(run_dir / "metrics.jsonl", "a")

    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB" if cuda else "-"
    print(f"model: {cfg.model.name} ({cfg.name}, id {cfg.id})")
    print(f"params: {n['total'] / 1e6:.1f}M (non-embedding {n['nonembedding'] / 1e6:.1f}M)")
    print(f"tok: {cfg.tokenizer.source} {tok.kind} vocab {tok.vocab} fp {tok.fp}")
    print(f"gpu: {gpu} | cuda {torch.version.cuda} | torch {torch.__version__} | vram {vram}")
    print(f"precision: {t.precision} | seq: {seqtokens(cfg.model)} | batch: {t.batch} x {t.accum} = {t.batch * t.accum}"
          f" | {fmt(per)} tok/step | memory cap: {f'{cap:g} GiB' if cap else 'none'}")
    print(f"data: {', '.join(f'{s} {w:.2f}' for s, w in batches.w.items())} | cache {root.name}")
    print(f"steps: {state['step']} -> {steps} ({fmt(steps * per)} tok)")

    model.train()
    t0 = time.time()
    last, lastt, acc, cnt = state["step"], time.time(), 0.0, 0
    for step in range(state["step"], steps):
        lr = lrat(step, t, steps)
        for gr in opt.param_groups:
            gr["lr"] = lr
        loss, gn, ntgt = trainstep(fwd, opt, scaler, batches, step, cfg, dev, ac)
        if not math.isfinite(loss):
            print(f"step {step} | loss {loss} | non-finite loss, stopping (last good checkpoint kept)")
            metrics.write(json.dumps({"step": step, "event": "nonfinite", "loss": str(loss)}) + "\n")
            metrics.close()
            sys.exit(2)
        state["step"] = step + 1
        state["tokens"] += per
        state["target_tokens"] += ntgt
        acc, cnt = acc + loss, cnt + 1
        s = state["step"]
        if s % t.log == 0 or s == steps:
            now = time.time()
            tps = (s - last) * per / (now - lastt)
            el = state["elapsed"] + now - t0
            avg = acc / cnt
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
            state["peak_gib"] = max(state.get("peak_gib", 0.0), peak)
            print(f"step {s} | loss {avg:.3f} | ppl {math.exp(min(avg, 20)):.1f} | lr {lr:.2e} | gn {gn:.2f} | "
                  f"{tps / 1e3:.1f}k tok/s | {(s - last) / (now - lastt):.2f} it/s | {fmt(state['tokens'])} | "
                  f"mem {peak:.1f}G | {el / 3600:.2f}h", flush=True)
            metrics.write(json.dumps({"step": s, "loss": avg, "lr": lr, "gnorm": gn, "tok_s": tps, "peak_gib": peak,
                                      "tokens": state["tokens"], "target_tokens": state["target_tokens"],
                                      "elapsed": el}) + "\n")
            metrics.flush()
            last, lastt, acc, cnt = s, now, 0.0, 0
        evaled = s % cfg.valid.every == 0 or s == steps
        if evaled:
            v = validstep(fwd, valid, cfg, dev, ac)
            state["valid"] = v
            best = v["all"] < state["best"]
            print(f"valid {s} | ce {v['all']:.4f} | ppl {math.exp(v['all']):.2f} | "
                  + " ".join(f"{k} {math.exp(v[k]):.1f}" for k in valid.srcs) + (" | best" if best else ""), flush=True)
            metrics.write(json.dumps({"step": s, "valid": v, "tokens": state["tokens"]}) + "\n")
            metrics.flush()
            if best:
                state["best"] = v["all"]
                state["beststep"] = s
                state["elapsed"] += time.time() - t0
                t0 = time.time()
                savebest(ck, model, dict(state), cfg, t.get("keepbest", 1))
        if s % t.save == 0 or s == steps:
            state["elapsed"] += time.time() - t0
            t0 = time.time()
            saveckpt(ck / "last.pt", model, dict(state), cfg, opt, scaler)
            if t.get("keep", 0):
                saveckpt(ck / f"step_{s}.pt", model, dict(state), cfg, opt, scaler, full=t.get("fullsteps", False))
            retain(ck, t.get("keep", 0), t.get("keepbest", 1))
            lastt = time.time()
        if t.get("sample") and (s % t.sample == 0 or s == steps):
            samples(model, tok, valid, cfg, dev, s, run_dir / "samples")
            lastt = time.time()
    metrics.close()
    print(f"done {state['step']} steps, {fmt(state['tokens'])} tok, best ce {state['best']:.4f} at {state.get('beststep')}")


if __name__ == "__main__":
    main()
