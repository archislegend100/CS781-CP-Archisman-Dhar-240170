"""Phase 4: court judgment prediction (CJPE, binary outcome) with the Phase-1 legal LMs.

Long documents (ILDC: Malik et al., ACL 2021, found the final part of a judgment the most predictive and hierarchical
chunk models the strongest) are handled two ways:
  * fine-tuned models read a selected window of the document (`take`: last / first / firstlast tokens) with the causal
    mask removed, attention + mean pooling, and an MLP head (all Phase-1 architectures, many adaptation techniques);
  * a hierarchical model reads the WHOLE document as a sequence of frozen Phase-1 chunk embeddings (chunks of 512
    tokens from the end), BiGRU + attention over chunks; its attention mass per relative position shows which parts of
    the judgment the prediction relies on.
The decision threshold on P(label = 1) is tuned on dev for macro-F1; ensembles average member probabilities.

python cj.py phase4/configs/<cfg>.yaml download | prep | train | predict | feats | hier | ensemble | teach
"""
import json
import math
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner
from config import args, loadcfg, path, plain, subdir, wrap
from data import writejson
from generate import autocast
from peft import AdaLoRAController, count, layers
from tokenizer import gettok

SPLITS = ("train", "dev", "test")


# ----------------------------------------------------------------------------- data

def readjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def loadsplit(p):
    """[(id, text, label or None)] from a list of records or a dict keyed by id."""
    d = readjson(p)
    recs = d if isinstance(d, list) else [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in d.items()]
    out = []
    for i, r in enumerate(recs):
        lab = next((r[k] for k in ("label", "gold", "labels", "decision", "y") if k in r), None)
        if isinstance(lab, list):
            lab = lab[0]
        text = r.get("text", r.get("document", ""))
        if isinstance(text, list):
            text = " ".join(text)
        out.append((str(r.get("id", i)), text, None if lab is None else int(lab)))
    return out


def files(cfg):
    c = cfg.cj
    return {k: path(c.root) / v for k, v in c.files.items()}


def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    c = cfg.cj
    have = list_repo_files(c.repo, repo_type="dataset")
    print("remote Phase-4 files:", *[f for f in have if f.startswith("Phase_4")], sep="\n  ")
    for k, f in c.files.items():
        if f not in have:
            hits = [x for x in have if x.startswith("Phase_4") and Path(x).name == Path(f).name]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix cj.files.{k}")
            f = hits[0]
        hf_hub_download(c.repo, f, repo_type="dataset", local_dir=str(path(c.root)))
        print(f"ok {f}")


_TOK = None


def _enc(text):
    return np.asarray(_TOK.encode(text), dtype=np.int16)


def prep(cfg):
    """Tokenise every split once (A1 BPE, multiprocessing) -> runs/cj/tok/<split>.pt; print data statistics."""
    import multiprocessing as mp
    global _TOK
    _TOK = gettok(loadcfg(cfg.cj.backbone.config))
    out = path(cfg.cj.tokdir)
    out.mkdir(parents=True, exist_ok=True)
    f = files(cfg)
    for split in SPLITS:
        t0 = time.time()
        recs = loadsplit(f[split])
        with mp.get_context("fork").Pool(int(cfg.cj.get("workers", 32))) as pool:
            toks = pool.map(_enc, [r[1] for r in recs], chunksize=16)
        torch.save({"ids": [r[0] for r in recs], "tok": toks, "labels": [r[2] for r in recs]}, out / f"{split}.pt")
        n = np.array([len(t) for t in toks])
        lab = Counter(r[2] for r in recs)
        print(f"{split}: {len(recs)} docs | labels {dict(lab)} | tokens/doc median {np.median(n):.0f} mean {n.mean():.0f} "
              f"p90 {np.percentile(n, 90):.0f} max {n.max()} | <=2048: {np.mean(n <= 2048):.1%} | {time.time() - t0:.0f}s", flush=True)


_TESTKEYS = {}


def loadtok(cfg, split):
    """Tokenised split. Train / dev cases that also occur in the test set (same id or identical token content) are
    removed (`notest`, default on): test labels are private, so no test case may be used for training, threshold
    tuning or ensemble weighting."""
    d = torch.load(path(cfg.cj.tokdir) / f"{split}.pt", map_location="cpu", weights_only=False)
    if split == "test" or not cfg.cj.get("notest", True):
        return d
    import hashlib
    key = str(path(cfg.cj.tokdir))
    if key not in _TESTKEYS:
        t = torch.load(path(cfg.cj.tokdir) / "test.pt", map_location="cpu", weights_only=False)
        _TESTKEYS[key] = (set(t["ids"]), {hashlib.md5(x.tobytes()).hexdigest() for x in t["tok"]})
    ids, hashes = _TESTKEYS[key]
    keep = [i for i, (k, x) in enumerate(zip(d["ids"], d["tok"]))
            if k not in ids and hashlib.md5(x.tobytes()).hexdigest() not in hashes]
    if len(keep) < len(d["ids"]):
        print(f"[notest] {split}: removed {len(d['ids']) - len(keep)} of {len(d['ids'])} cases that also occur in the test set")
    return {k: [d[k][i] for i in keep] for k in ("ids", "tok", "labels")}


def take(arr, L, how):
    if len(arr) <= L:
        return arr
    if how == "first":
        return arr[:L]
    if how == "firstlast":
        a = L // 4
        return np.concatenate([arr[:a], arr[-(L - a):]])
    return arr[-L:]


def winstack(arr, K, W):
    """Up to K windows of W tokens covering the end of the document, earliest first. If the document is shorter than
    K * W the windows overlap evenly (so every window is exactly W long); a document shorter than W is one window."""
    n = len(arr)
    if n <= W:
        return arr[None]
    k = min(K, math.ceil(n / W))
    if k == 1:
        return arr[-W:][None]
    step = W if n >= k * W else (n - W) / (k - 1)
    ends = [n - round(j * step) for j in range(k)][::-1]
    return np.stack([arr[e - W:e] for e in ends])


def macro2(y, p):
    """Binary macro-F1 over both classes (sklearn f1_score(average='macro', labels=[0, 1], zero_division=0))."""
    y, p = np.asarray(y), np.asarray(p)
    fs = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(fs) / 2


def bestthr(y, p1):
    """Decision threshold on P(1) maximising macro-F1 (Lipton et al. 2014)."""
    best = (macro2(y, (np.asarray(p1) > 0.5).astype(int)), 0.5)
    for t in np.arange(0.05, 0.951, 0.01):
        f = macro2(y, (np.asarray(p1) > t).astype(int))
        if f > best[0] + 1e-9:
            best = (f, float(t))
    return best


# ----------------------------------------------------------------------------- fine-tuned window model

class CJModel(nn.Module):
    """Phase-1 encoder (any adaptation) -> attention pooling + mean pooling over the window -> MLP -> 2 logits.
    With `nwin` > 1 (end-to-end hierarchical, cf. Dai et al. 2022): each of up to `nwin` final windows is encoded by the
    shared encoder, pooled, given a window-position embedding (counted from the end) and mixed by a transformer layer
    across windows before attention pooling and the head."""

    def __init__(self, enc, dim, c):
        super().__init__()
        self.enc = enc
        self.enc.cls = nn.Identity()
        self.att = nn.Linear(dim, 1)
        self.nwin = int(c.get("nwin", 1))
        if self.nwin > 1:
            self.wproj = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
            self.wpos = nn.Embedding(self.nwin, dim)
            self.wmix = nn.TransformerEncoderLayer(dim, 8, 2 * dim, dropout=float(c.get("hdrop", 0.1)), batch_first=True,
                                                   norm_first=True)
            self.watt = nn.Linear(dim, 1)
        self.head = nn.Sequential(nn.Dropout(float(c.get("hdrop", 0.1))), nn.Linear(2 * dim, dim // 2), nn.GELU(),
                                  nn.Dropout(float(c.get("hdrop", 0.1))), nn.Linear(dim // 2, 2))

    def pool(self, ids):
        h = self.enc.hidden(ids).float()
        a = torch.softmax(self.att(h).squeeze(-1), -1)
        return torch.cat([(a[..., None] * h).sum(1), h.mean(1)], -1)

    def forward(self, ids):
        if ids.dim() == 2:
            return self.head(self.pool(ids))
        B, K, T = ids.shape                       # (docs, windows, tokens): hierarchical path
        z = self.wproj(self.pool(ids.reshape(B * K, T)).reshape(B, K, -1))
        z = self.wmix(z + self.wpos(torch.arange(K - 1, -1, -1, device=ids.device))[None])
        a = torch.softmax(self.watt(z).squeeze(-1), -1)
        return self.head(torch.cat([(a[..., None] * z).sum(1), z.mean(1)], -1))


def inputs(toks, c, offset=0):
    """Model inputs per document: a token window (`take`), or a (K, W) window stack when `nwin` > 1; `offset` drops the
    last `offset` tokens first (test-time augmentation with earlier windows)."""
    L, how, K = int(c.window), c.get("take", "last"), int(c.get("nwin", 1))
    out = []
    for t in toks:
        if offset and len(t) - offset >= L // 2:
            t = t[:len(t) - offset]
        out.append(winstack(t, K, L) if K > 1 else take(t, L, how))
    return out


def shapekey(x):
    return x.shape


def build(cfg, dev):
    t, info = ner.make(wrap({**cfg, "ner": cfg.cj}), dev, ["x"])
    return CJModel(t, int(info["mcfg"]["dim"]), cfg.cj).to(dev), info


def loadcj(run, dev):
    obj = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = wrap(obj["cfg"])
    m, _ = build(cfg, dev)
    m.load_state_dict(obj["model"])
    return m.eval(), obj


def batches(lens, B, rng, shuffle=True):
    """Equal-shape batches (no padding, so unmasked attention never sees pads); `lens` are lengths or shapes."""
    by = defaultdict(list)
    for i, n in enumerate(lens):
        by[n].append(i)
    out = []
    for g in by.values():
        if shuffle:
            rng.shuffle(g)
        out += [g[i:i + B] for i in range(0, len(g), B)]
    if shuffle:
        rng.shuffle(out)
    return out


@torch.no_grad()
def probs(model, toks, cfg, dev):
    c = cfg.cj
    was = model.training
    model.eval()
    pr = c.get("precision", "bf16")
    offs = list(c.get("tta", [0])) if c.get("tta") else [0]
    out = np.zeros(len(toks), dtype=np.float32)
    for off in offs:                       # test-time augmentation: average over windows ending earlier
        seqs = inputs(toks, c, off)
        for b in batches([shapekey(s) for s in seqs], int(c.get("evalbatch", 16)), None, shuffle=False):
            ids = torch.tensor(np.stack([seqs[i] for i in b]).astype(np.int64), device=dev)
            with autocast(dev, pr):
                lg = model(ids)
            out[b] += torch.softmax(lg.float(), -1)[:, 1].cpu().numpy() / len(offs)
    model.train(was)
    return out


def unfreezeplan(model, nl):
    """Gradual unfreezing (ULMFiT, Howard & Ruder 2018): blocks listed top first."""
    return list(reversed(layers(model.enc.backbone)))[:nl]


def train(cfg):
    c = cfg.cj
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = loadtok(cfg, "train"), loadtok(cfg, "dev")
    if c.get("alldocs"):
        tr = {k: tr[k] + va[k] for k in ("ids", "tok", "labels")}
    if c.get("devfold") is not None:      # dev cross-fitting: train + the other dev fold(s); select on this fold
        fo = devfolds(va["ids"], int(c.get("devfolds", 2)))
        k = int(c.devfold)
        keep = [i for i, d in enumerate(va["ids"]) if fo[d] != k]
        held = [i for i, d in enumerate(va["ids"]) if fo[d] == k]
        r = int(c.get("devrep", 1))           # oversample the in-distribution dev docs (importance weighting)
        tr = {key: tr[key] + [va[key][i] for i in keep] * r for key in ("ids", "tok", "labels")}
        va = {key: [va[key][i] for i in held] for key in ("ids", "tok", "labels")}
        print(f"dev fold {k}: +{len(keep)} dev docs in training (x{r}), {len(held)} held out for selection")
    L, how = int(c.window), c.get("take", "last")
    seqs = inputs(tr["tok"], c)
    views = [inputs(tr["tok"], c, o) for o in c.augoffsets] if c.get("augoffsets") else None   # window augmentation
    y = np.array(tr["labels"])
    sw = np.ones(len(y), dtype=np.float32)
    if c.get("lenweight"):
        # covariate-shift correction (Shimodaira 2000): weight each training doc by p_dev(len bucket) / p_train(len bucket),
        # length buckets = floor(log2(tokens)); uses document lengths only (dev / test are long, train has many short docs)
        ref = [len(t) for t in loadtok(cfg, "dev")["tok"]]
        bk = lambda n: min(17, max(6, int(math.log2(max(1, n)))))
        pt = Counter(bk(len(t)) for t in tr["tok"])
        pr_ = Counter(bk(n) for n in ref)
        sw = np.array([np.clip((pr_[bk(len(t))] / len(ref)) / (pt[bk(len(t))] / len(y)), 0.25, 4.0) for t in tr["tok"]],
                      dtype=np.float32)
        sw /= sw.mean()
        print("length reweighting: " + ", ".join(f"2^{b}: {np.clip((pr_[b] / len(ref)) / (pt[b] / len(y)), 0.25, 4.0):.2f}"
                                                 for b in sorted(pt)))
    cnt = np.bincount(y, minlength=2)
    cw = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))) ** float(c.get("cw", 0.0)), dtype=torch.float32, device=dev)
    model, info = build(cfg, dev)
    teacher = None
    if c.get("kd", {}).get("teacher"):
        tp = torch.load(path(c.kd.teacher), map_location="cpu", weights_only=False)["train"]
        teacher = np.array([tp[i] for i in tr["ids"]], dtype=np.float32)
    meth = c.method
    blocks = layers(model.enc.backbone)
    if meth == "unfreeze":            # discriminative learning rates (lr * decay^depth-from-top), all frozen at start
        dec = float(c.get("lrdecay", 0.85))
        groups = [{"params": list(b.parameters()), "lr": float(c.lr) * dec ** k} for k, b in enumerate(reversed(blocks))]
        for b in blocks:
            for p in b.parameters():
                p.requires_grad = False
        nrm = model.enc.backbone.encnorm if model.enc.family == "seq2seq" else model.enc.backbone.norm
        groups.append({"params": list(nrm.parameters()), "lr": float(c.lr)})
    else:
        groups = [{"params": [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith("enc.backbone.")],
                   "lr": float(c.lr)}]
    groups.append({"params": [p for n_, p in model.named_parameters() if not n_.startswith("enc.backbone.") and p.requires_grad],
                   "lr": float(c.headlr)})
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=float(c.get("wd", 0.01)))
    B, acc, E = int(c.batch), int(c.get("accum", 1)), float(c.epochs)
    nb = len(batches([shapekey(s) for s in seqs], B, random.Random(0)))
    steps = max(1, int(nb * E / acc))
    warm = int(float(c.get("warmup", 0.06)) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    ada = AdaLoRAController(model, float(c.get("target", 4)), steps, int(0.1 * steps), int(0.3 * steps)) if meth == "adalora" else None
    tot, trn = count(model)
    run = path(f"runs/cj/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | {info['phase1_model']} from {info['phase1']} | method {meth} | window {L} "
          f"({str(c.get('nwin', 1)) + ' windows' if int(c.get('nwin', 1)) > 1 else how}) | "
          f"params {tot / 1e6:.1f}M, trainable {trn / 1e6:.2f}M ({100 * trn / tot:.2f}%) | train {len(seqs)} docs "
          f"(labels {cnt.tolist()}) | {steps} optimizer steps" + (" | KD" if teacher is not None else ""), flush=True)
    pr = c.get("precision", "bf16")
    T, alpha = float(c.get("kd", {}).get("T", 2.0)), float(c.get("kd", {}).get("alpha", 0.5))
    every = int(c.get("evalevery", 0)) or max(1, steps // 4)
    best, step, micro, t0, tl = -1.0, 0, 0, time.time(), 0.0
    log = open(run / "metrics.jsonl", "a")
    done = False
    ep = 0
    while not done:
        if views is not None:      # each doc: one of the windows ending `augoffsets` tokens before the end
            seqs = [views[rng.randrange(len(views))][i] for i in range(len(y))]
        for b in batches([shapekey(s) for s in seqs], B, rng):
            if meth == "unfreeze":    # open the top ceil(f * L) blocks, all of them by `unfreezeat` of training
                k = min(len(blocks), 1 + int(len(blocks) * step / max(1, float(c.get("unfreezeat", 0.5)) * steps)))
                for blk in unfreezeplan(model, k):
                    for p in blk.parameters():
                        p.requires_grad = True
            ids = torch.tensor(np.stack([seqs[i] for i in b]).astype(np.int64), device=dev)
            yy = torch.tensor(y[b], device=dev)
            with autocast(dev, pr):
                lg = model(ids)
            lg = lg.float()
            loss = (F.cross_entropy(lg, yy, weight=cw, reduction="none") * torch.tensor(sw[b], device=dev)).mean()
            if teacher is not None:
                t1 = torch.tensor(teacher[b], device=dev).clamp(1e-6, 1 - 1e-6)
                tpr = torch.stack([1 - t1, t1], -1) ** (1 / T)
                loss = alpha * loss + (1 - alpha) * T * T * F.kl_div(F.log_softmax(lg / T, -1), tpr / tpr.sum(-1, keepdim=True),
                                                                      reduction="batchmean")
            (loss / acc).backward()
            tl += float(loss.detach())
            micro += 1
            if micro % acc:
                continue
            if ada is not None:
                (float(c.get("orth", 0.1)) * ada.penalty()).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(c.get("clip", 1.0)))
            if ada is not None:
                ada.update(step)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} | loss {tl / (50 * acc):.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                tl = 0.0
            if step % every == 0 or step >= steps:
                if not c.get("alldocs"):
                    p1 = probs(model, va["tok"], cfg, dev)
                    f, thr = bestthr(va["labels"], p1)
                    f5 = macro2(va["labels"], (p1 > 0.5).astype(int))
                    print(f"eval step {step}: dev macro-F1 {f:.4f} at threshold {thr:.2f} (0.5: {f5:.4f})", flush=True)
                    log.write(json.dumps({"step": step, "dev": f, "thr": thr, "dev05": f5}) + "\n")
                    log.flush()
                    if f > best:
                        best = f
                        save(run, model, cfg, info, tot, trn, step, {"f1": f, "thr": thr, "f1_05": f5})
                elif step >= steps:
                    save(run, model, cfg, info, tot, trn, step, None)
            if step >= steps:
                done = True
                break
        ep += 1
    log.close()
    print(f"best dev macro-F1 {best:.4f}" if not c.get("alldocs") else "trained on train + dev")


def save(run, model, cfg, info, tot, trn, step, dev):
    tot, trn = count(model)          # recount: gradual unfreezing opens layers during training
    torch.save({"model": model.state_dict(), "cfg": plain(cfg), "step": step, "dev": dev,
                "info": {k: v for k, v in info.items() if k != "mcfg"}, "params": tot, "trainable": trn}, run / "best.pt")
    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": cfg.cj.method, "step": step, "dev": dev,
                                  "params": tot, "trainable": trn, **{k: v for k, v in info.items() if k != "mcfg"}}, 1)


def writesub(cfg, ids, pred, meta):
    out = [{"id": i, "pred": int(p)} for i, p in zip(ids, pred)]
    assert len({r["id"] for r in out}) == len(out)
    sd = subdir(cfg)
    writejson(sd / "prediction.json", out, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, **meta, "test_cases": len(out),
                                 "test_pred_counts": dict(Counter(int(p) for p in pred))}, 1)
    print(f"ok: {len(out)} test cases (pred counts {dict(Counter(int(p) for p in pred))}) -> {sd / 'submission.zip'}")


def officialdev(cfg, run, ids, gold, pred):
    import subprocess
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase4.py"))
    if not ev.exists():
        return
    writejson(run / "dev_truth.json", [{"id": i, "gold": int(g)} for i, g in zip(ids, gold)])
    writejson(run / "dev_pred.json", [{"id": i, "pred": int(p)} for i, p in zip(ids, pred)])
    r = subprocess.run([sys.executable, str(ev), "--truth", str(run / "dev_truth.json"), "--prediction", str(run / "dev_pred.json")],
                       capture_output=True, text=True)
    print("official evaluator on dev: " + " | ".join(x.strip() for x in (r.stdout + r.stderr).splitlines() if x.strip()))


def devfolds(ids, K):
    """Deterministic K-fold split of the dev cases by id (repeated ids stay together)."""
    u = sorted(set(ids))
    random.Random(0).shuffle(u)
    return {d: i % K for i, d in enumerate(u)}


def devjoin(cfg):
    """Cross-fitted member from dev-fold runs: each dev case takes P(1) from the run that held it out, test the mean;
    threshold tuned on these honest dev predictions."""
    q = cfg.devjoin
    dv, ts = loadtok(cfg, "dev"), loadtok(cfg, "test")
    ms = [torch.load(path(r) / "probs.pt", map_location="cpu", weights_only=False) for r in q.runs]
    fo = devfolds(dv["ids"], len(ms))
    out = {"dev": {d: ms[fo[d]]["dev"][d] for d in set(dv["ids"])},
           "test": {d: float(np.mean([m["test"][d] for m in ms])) for d in ts["ids"]},
           "params": ms[0]["params"], "trainable": ms[0]["trainable"], "phase1": ms[0]["phase1"], "method": ms[0]["method"]}
    for k, r in enumerate(q.runs):
        held = [i for i, d in enumerate(dv["ids"]) if fo[d] == k]
        f, t = bestthr([dv["labels"][i] for i in held], [out["dev"][dv["ids"][i]] for i in held])
        print(f"  fold {k} ({r}): held-out dev macro-F1 {f:.4f} at {t:.2f}")
    run = path(f"runs/cj/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save(out, run / "probs.pt")
    finish(cfg, run, out, dv, ts, {"method": f"dev cross-fitted {out['method']} ({len(ms)} folds)", "folds": list(q.runs)})


def predict(cfg):
    """P(1) on dev / test (and train when `trainprobs`, for distillation) -> probs.pt; dev-tuned threshold submission."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = path(f"runs/cj/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    model, obj = loadcj(path(f"runs/cj/{cfg.cj.get('fromrun', cfg.name)}"), dev)
    out = {"params": obj["params"], "trainable": obj["trainable"], "phase1": obj["info"]["phase1"], "method": cfg.cj.method}
    splits = SPLITS if cfg.cj.get("trainprobs") else ("dev", "test")
    data = {s: loadtok(cfg, s) for s in splits}
    for s in splits:
        t0 = time.time()
        out[s] = dict(zip(data[s]["ids"], probs(model, data[s]["tok"], cfg, dev).tolist()))
        print(f"{s}: {len(out[s])} docs scored in {time.time() - t0:.0f}s", flush=True)
    torch.save(out, run / "probs.pt")
    finish(cfg, run, out, data["dev"], data["test"], {"method": cfg.cj.method, "phase1": obj["info"]["phase1"],
                                                      "params": obj["params"], "trainable": obj["trainable"], "step": obj["step"]})


def finish(cfg, run, out, dv, ts, meta):
    p1 = np.array([out["dev"][i] for i in dv["ids"]])
    f, thr = bestthr(dv["labels"], p1)
    src = cfg.cj.get("thrfrom") or cfg.get("ens", {}).get("thrfrom")
    if src:                                   # e.g. a train+dev refit reuses its train-only twin's dev threshold
        thr = float(readjson(path(src))["threshold"])
    elif cfg.cj.get("alldocs") or cfg.get("hier", {}).get("alldocs"):
        thr = 0.5
    pd_ = (p1 > thr).astype(int)
    y = np.array(dv["labels"])
    fp, fn = int(((pd_ == 1) & (y == 0)).sum()), int(((pd_ == 0) & (y == 1)).sum())
    print(f"dev macro-F1 {macro2(y, pd_):.4f} at threshold {thr:.2f} (0.5: {macro2(y, (p1 > 0.5).astype(int)):.4f}) | FP {fp} FN {fn}")
    officialdev(cfg, run, dv["ids"], dv["labels"], pd_)
    pt = np.array([out["test"][i] for i in ts["ids"]])
    writesub(cfg, ts["ids"], (pt > thr).astype(int), {**meta, "threshold": thr, "dev_f1": macro2(y, pd_),
                                                     "dev_f1_05": macro2(y, (p1 > 0.5).astype(int)), "dev_fp": fp, "dev_fn": fn})


# ----------------------------------------------------------------------------- hierarchical model over frozen chunks

@torch.no_grad()
def feats(cfg):
    """Frozen chunk embeddings of whole documents: chunks of `chunk` tokens aligned from the end (at most `maxchunks`),
    mean state at several depths (final norm applied); causal backbones run as pretrained, LegalSeq via its encoder."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = cfg.feats
    out = path(q.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {s: loadtok(cfg, s) for s in SPLITS}
    C, M = int(q.get("chunk", 512)), int(q.get("maxchunks", 64))
    for b in q.backbones:
        f = out / f"{b.name}.pt"
        if f.exists() or not path(b.run).exists():
            print(f"skip {b.name}: {'exists' if f.exists() else b.run + ' not found'}")
            continue
        c_ = wrap({**plain(cfg.cj), "backbone": {"config": b.config, "run": b.run, "ckpt": b.ckpt}, "method": "full",
                   "unmask": "none", "layers": None, "peftlayers": None, "mix": False})
        t, info = ner.make(wrap({**cfg, "ner": c_}), dev, ["x"])
        bb = t.backbone.eval()
        s2s = t.family == "seq2seq"
        nl = len(bb.enc) if s2s else len(bb.blocks)
        depth = sorted({max(1, round(fr * nl)) for fr in q.get("depths", [0.5, 1.0])})
        nrm = bb.encnorm if s2s else bb.norm
        res = {}
        t0 = time.time()
        for s in SPLITS:
            jobs = []          # (doc index, chunk index, token array)
            res[s] = []
            for di, tok in enumerate(data[s]["tok"]):
                n = len(tok)
                k = max(1, min(M, n // C))
                starts = [max(0, n - (j + 1) * C) for j in range(k)][::-1]
                res[s].append([None] * k)
                for ci, st in enumerate(starts):
                    jobs.append((di, ci, tok[st:st + C] if n >= C else tok))
            by = defaultdict(list)
            for j in jobs:
                by[len(j[2])].append(j)
            for grp in by.values():
                for a in range(0, len(grp), int(q.get("batch", 64))):
                    g = grp[a:a + int(q.get("batch", 64))]
                    ids = torch.tensor(np.stack([x[2] for x in g]).astype(np.int64), device=dev)
                    with autocast(dev, "bf16"):
                        hs = bb.encode(ids, torch.ones_like(ids, dtype=torch.bool), hiddens=True)[1] if s2s \
                            else bb(ids, last=True, hiddens=True)["hiddens"]
                    v = torch.cat([nrm(hs[k]).float().mean(1) for k in depth], -1).cpu().numpy().astype(np.float16)
                    for x, vv in zip(g, v):
                        res[s][x[0]][x[1]] = vv
            res[s] = [np.stack(r) for r in res[s]]
            print(f"  {b.name} {s}: {len(jobs)} chunks ({time.time() - t0:.0f}s)", flush=True)
        torch.save({"feats": {s: dict(zip(data[s]["ids"], res[s])) for s in SPLITS}, "dim": int(res["train"][0].shape[1]),
                    "params": sum(p.numel() for p in bb.parameters()), "phase1": f"{b.run}:{b.ckpt}",
                    "model": info["phase1_model"], "depths": depth}, f)
        print(f"ok {b.name} ({info['phase1_model']}): -> {f} ({time.time() - t0:.0f}s)", flush=True)
        del t, bb
        torch.cuda.empty_cache()


class HierModel(nn.Module):
    """Chunk embeddings -> projection + relative-position embedding -> BiGRU -> attention pooling -> MLP (cf. ILDC
    hierarchical models, Malik et al. 2021; HAN, Yang et al. 2016)."""

    def __init__(self, fin, q):
        super().__init__()
        p, h = int(q.get("proj", 256)), int(q.get("hid", 128))
        self.fdrop = nn.Dropout(float(q.get("fdrop", 0.3)))
        self.proj = nn.Sequential(nn.Linear(fin, p), nn.LayerNorm(p), nn.GELU())
        self.npos = int(q.get("posbuckets", 10))
        self.pos = nn.Embedding(self.npos, p)
        self.gru = nn.GRU(p, h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * h, 1)
        self.drop = nn.Dropout(float(q.get("sdrop", 0.3)))
        self.out = nn.Sequential(nn.Linear(4 * h, h), nn.GELU(), nn.Dropout(float(q.get("sdrop", 0.3))), nn.Linear(h, 2))

    def forward(self, x, lens):
        B, N, _ = x.shape
        rel = torch.arange(N, device=x.device)[None].float() / lens[:, None].clamp(min=1).float()
        pos = (rel * self.npos).long().clamp(max=self.npos - 1)
        z = self.proj(self.fdrop(x)) + self.pos(pos)
        packed = nn.utils.rnn.pack_padded_sequence(z, lens.cpu(), batch_first=True, enforce_sorted=False)
        g = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=N)[0]
        mask = torch.arange(N, device=x.device)[None] < lens[:, None]
        a = torch.softmax(self.att(self.drop(g)).squeeze(-1).masked_fill(~mask, -1e9), -1)
        mean = (g * mask[..., None]).sum(1) / lens[:, None].float()
        return self.out(torch.cat([(a[..., None] * g).sum(1), mean], -1)), a


def hier(cfg):
    """Train the hierarchical chunk model (bag of seeds, early stopping on dev macro-F1 at the tuned threshold); report
    the attention mass per relative position (which parts of the judgment matter); write probs.pt + submission."""
    q = cfg.hier
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    have = [n for n in q.feats if (path(cfg.feats.out) / f"{n}.pt").exists()]
    if len(have) < len(q.feats):
        print(f"missing feature files (skipped): {sorted(set(q.feats) - set(have))}")
    fs = [torch.load(path(cfg.feats.out) / f"{n}.pt", map_location="cpu", weights_only=False) for n in have]
    data = {s: loadtok(cfg, s) for s in SPLITS}
    region = q.get("region", "all")

    def doc(s, i):
        x = np.concatenate([f["feats"][s][i].astype(np.float32) for f in fs], 1)
        n = len(x)
        if region == "first" and n > 1:
            x = x[:max(1, n // 2)]
        elif region == "last" and n > 1:
            x = x[n // 2:]
        return x

    X = {s: [torch.from_numpy(doc(s, i).astype(np.float16)) for i in data[s]["ids"]] for s in SPLITS}
    tot = sum(len(x) for x in X["train"])
    mu = sum(x.float().sum(0) for x in X["train"]) / tot
    sd = (sum(((x.float() - mu) ** 2).sum(0) for x in X["train"]) / tot).sqrt() + 1e-5
    mu, sd = mu.to(dev), sd.to(dev)        # features stay fp16 in RAM, standardised per batch on the GPU
    trs = ("train", "dev") if q.get("alldocs") else ("train",)
    tx = [x for s in trs for x in X[s]]
    ty = np.array([l for s in trs for l in data[s]["labels"]])
    cnt = np.bincount(ty, minlength=2)
    cw = torch.tensor((cnt.sum() / (2 * cnt)) ** float(q.get("cw", 0.0)), dtype=torch.float32, device=dev)
    fin = tx[0].shape[1]
    run = path(f"runs/cj/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | frozen views {have} ({fin} dims) | region {region} | train {len(tx)} docs {cnt.tolist()} | seeds {list(q.seeds)}")

    def run_(m, xs, B=64):
        m.eval()
        ps, atts = [], []
        with torch.no_grad():
            for a in range(0, len(xs), B):
                xb = xs[a:a + B]
                lens = torch.tensor([len(x) for x in xb], device=dev)
                pad = (nn.utils.rnn.pad_sequence(xb, batch_first=True).to(dev).float() - mu) / sd
                lg, at = m(pad, lens)
                ps.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
                atts += [at[j, :lens[j]].cpu().numpy() for j in range(len(xb))]
        m.train()
        return np.concatenate(ps), atts

    seedp = {s: [] for s in SPLITS}
    attsum = np.zeros(10)
    for seed in q.seeds:
        torch.manual_seed(int(seed))
        rng = random.Random(int(seed))
        m = HierModel(fin, q).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=float(q.get("lr", 1e-3)), weight_decay=float(q.get("wd", 0.01)))
        best, bad, bstate, B = -1.0, 0, None, int(q.get("batch", 32))
        for ep in range(int(q.get("epochs", 30))):
            order = list(range(len(tx)))
            rng.shuffle(order)
            for a in range(0, len(order), B):
                idx = order[a:a + B]
                lens = torch.tensor([len(tx[i]) for i in idx], device=dev)
                pad = (nn.utils.rnn.pad_sequence([tx[i] for i in idx], batch_first=True).to(dev).float() - mu) / sd
                lg, _ = m(pad, lens)
                loss = F.cross_entropy(lg, torch.tensor(ty[idx], device=dev), weight=cw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if q.get("alldocs"):
                bstate = {k: v.detach().clone() for k, v in m.state_dict().items()}
                continue
            p1, _ = run_(m, X["dev"])
            f, thr = bestthr(data["dev"]["labels"], p1)
            if f > best:
                best, bad, bstate = f, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= int(q.get("patience", 5)):
                    break
        m.load_state_dict(bstate)
        print(f"  seed {seed}: best dev macro-F1 {best:.4f} (epoch {ep + 1 - bad})", flush=True)
        for s in SPLITS:
            p1, at = run_(m, X[s])
            seedp[s].append(p1)
            if s == "dev":
                for a_ in at:
                    rel = (np.arange(len(a_)) + 0.5) / len(a_)
                    np.add.at(attsum, np.minimum(9, (rel * 10).astype(int)), a_)
    out = {s: dict(zip(data[s]["ids"], np.mean(seedp[s], 0).tolist())) for s in SPLITS}
    head = sum(p.numel() for p in m.parameters()) * len(q.seeds)
    out.update({"params": head + sum(f["params"] for f in fs), "trainable": head, "phase1": ",".join(f["phase1"] for f in fs),
                "method": "frozen"})
    torch.save(out, run / "probs.pt")
    att = attsum / attsum.sum()
    print("attention mass per document decile (start -> end): " + " ".join(f"{a:.3f}" for a in att))
    finish(cfg, run, out, data["dev"], data["test"], {"method": "hierarchical frozen chunks + BiGRU attention", "views": have,
                                                      "region": region, "attention_by_decile": att.tolist(),
                                                      "params": out["params"], "trainable": head})


# ----------------------------------------------------------------------------- ensemble and distillation teacher

def loadprobs(runs):
    out = []
    for r in runs:
        p = path(r) / "probs.pt"
        if p.exists():
            out.append((r, torch.load(p, map_location="cpu", weights_only=False)))
        else:
            print(f"skip {r}: no probs.pt")
    return out


def ensemble(cfg):
    """Mean of member P(1) with greedy dev-tuned member weights, dev-tuned threshold."""
    e = cfg.ens
    dv, ts = loadtok(cfg, "dev"), loadtok(cfg, "test")
    ms = loadprobs(e.runs)
    y = np.array(dv["labels"])
    P = np.stack([[m["dev"][i] for i in dv["ids"]] for _, m in ms])
    w = np.ones(len(ms))
    best = bestthr(y, w @ P / w.sum())[0]
    print(f"{len(ms)} members, equal weights: dev {best:.4f} | singles: " +
          ", ".join(f"{Path(r).name} {bestthr(y, P[j])[0]:.4f}" for j, (r, _) in enumerate(ms)))
    if e.get("tuneweights", True):
        for _ in range(2):
            for j in range(len(ms)):
                for v in e.get("wgrid", [0.0, 0.5, 2.0, 3.0]):
                    t_ = w.copy()
                    t_[j] = v
                    if t_.sum() == 0:
                        continue
                    f = bestthr(y, t_ @ P / t_.sum())[0]
                    if f > best + 1e-4:
                        best, w = f, t_
    print(f"weights {dict(zip([Path(r).name for r, _ in ms], w.tolist()))}: dev {best:.4f}")
    out = {s: {} for s in ("dev", "test", "train")}
    for s, d in (("dev", dv), ("test", ts)):
        out[s] = dict(zip(d["ids"], (w @ np.stack([[m[s][i] for i in d["ids"]] for _, m in ms]) / w.sum()).tolist()))
    if all("train" in m for _, m in ms):
        ids = list(ms[0][1]["train"])
        out["train"] = dict(zip(ids, (w @ np.stack([[m["train"][i] for i in ids] for _, m in ms]) / w.sum()).tolist()))
    run = path(f"runs/cj/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save({**out, "params": 0, "trainable": 0, "phase1": "ensemble", "method": "ensemble"}, run / "probs.pt")
    p1 = np.array([out["dev"][i] for i in dv["ids"]])
    f, thr = bestthr(y, p1)
    writejson(run / "threshold.json", {"threshold": thr, "dev_f1": f})
    finish(cfg, run, out, dv, ts, {"method": "ensemble (weighted mean P(1)) + dev threshold", "members": [r for r, _ in ms],
                                   "weights": w.tolist()})


def teach(cfg):
    """Distillation targets: the (weighted) ensemble's P(1) on the training documents."""
    run = path(f"runs/cj/{cfg.teach.fromrun}")
    p = torch.load(run / "probs.pt", map_location="cpu", weights_only=False)
    assert p.get("train"), "the teacher ensemble needs train probabilities (trainprobs: true on its members)"
    out = path(cfg.teach.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": p["train"]}, out)
    print(f"ok: teacher P(1) for {len(p['train'])} training docs -> {out}")


def main():
    cfg, modes = args()
    {"download": download, "prep": prep, "train": train, "predict": predict, "feats": feats, "hier": hier,
     "ensemble": ensemble, "teach": teach, "devjoin": devjoin}[modes[0] if modes else "prep"](cfg)


if __name__ == "__main__":
    main()
