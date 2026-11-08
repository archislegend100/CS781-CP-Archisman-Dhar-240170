"""Phase 7: prior case retrieval (PCR) with the Phase-1 legal LMs.

Hybrid retrieval (IL-PCR / U-CREAT, Joshi et al. 2023: lexical n-gram BM25 is a strong baseline for Indian PCR):
  * lexical: BM25 over word uni- and bi-grams of the full documents;
  * dense: a Phase-1 encoder (any adaptation technique, causal mask removed) embeds 512-token chunks by mean pooling; a
    document is the normalised mean of up to `nchunks` evenly spaced chunks (long-document MeanP, Dai & Callan 2019);
    trained with InfoNCE over in-batch + BM25 hard negatives (DPR, Karpukhin et al. 2020) and a learned temperature;
  * fusion: per-query z-normalised BM25 and dense scores, weight tuned on dev; exactly 7 unique candidates per query.
Distillation: an 8-block student matches the teacher's similarity distribution over the batch candidates (listwise KL).

python pcr.py phase7/configs/<cfg>.yaml download | prep | bm25 | train | predict | fuse
"""
import json
import math
import random
import re
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
from peft import count
from tokenizer import gettok

K = 7
SPLITS = ("train", "dev", "test")
STOP = set("the a an of to in and or for on by with is was are were be been as at that this which it its from has have had "
           "not no shall may said such any all their there these those than then into upon under also he she they his her "
           "him them we our you your i".split())


# ----------------------------------------------------------------------------- data

def readjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def astext(v):
    if isinstance(v, dict):
        v = v.get("text", v.get("document", ""))
    return " ".join(v) if isinstance(v, list) else str(v)


def loadq(p):
    """{query id: (text, [relevant candidate ids] or None)} from a dict or a list of records."""
    d = readjson(p)
    items = d.items() if isinstance(d, dict) else [(r.get("id"), r) for r in d]
    out = {}
    for k, v in items:
        rel = v.get("relevant_candidates") if isinstance(v, dict) else None
        out[str(k)] = (astext(v), None if rel is None else [str(x) for x in rel])
    return out


def loadc(p):
    d = readjson(p)
    items = d.items() if isinstance(d, dict) else [(r.get("id"), r) for r in d]
    return {str(k): astext(v) for k, v in items}


def files(cfg):
    c = cfg.pcr
    return {k: path(c.root) / v for k, v in c.files.items()}


def download(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    c = cfg.pcr
    have = list_repo_files(c.repo, repo_type="dataset")
    print("remote Phase-7 files:", *[f for f in have if f.startswith("Phase_7")], sep="\n  ")
    for k, f in c.files.items():
        if f not in have:
            hits = [x for x in have if x.startswith("Phase_7") and Path(x).name == Path(f).name]
            if not hits:
                raise SystemExit(f"{f} not in repo; fix pcr.files.{k}")
            f = hits[0]
        hf_hub_download(c.repo, f, repo_type="dataset", local_dir=str(path(c.root)))
        print(f"ok {f}")


_TOK = None


def _enc(t):
    return np.asarray(_TOK.encode(t), dtype=np.int16)


def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 1]


def prep(cfg):
    """Tokenise queries + candidate pools of every split -> runs/pcr/data.pt; statistics."""
    import multiprocessing as mp
    global _TOK
    c = cfg.pcr
    out = path(c.datadir) / "data.pt"
    if out.exists():
        print(f"exists: {out}")
        return
    _TOK = gettok(loadcfg(c.backbone.config))
    f = files(cfg)
    data = {}
    for s in SPLITS:
        q, cand = loadq(f[f"{s}_queries"]), loadc(f[f"{s}_candidates"])
        texts = [t for t, _ in q.values()] + list(cand.values())
        with mp.get_context("fork").Pool(int(c.get("workers", 32))) as pool:
            toks = pool.map(_enc, texts, chunksize=8)
        nq = len(q)
        data[s] = {"qids": list(q), "qtok": toks[:nq], "rel": [r for _, r in q.values()], "cids": list(cand),
                   "ctok": toks[nq:], "qtext": [t for t, _ in q.values()], "ctext": list(cand.values())}
        rel = [r for r in data[s]["rel"] if r is not None]
        lens = np.array([len(t) for t in toks])
        miss = sum(1 for r in rel for x in r if x not in cand)
        selfin = sum(1 for qq in q if qq in cand)
        print(f"{s}: {nq} queries, {len(cand)} candidates | tokens/doc median {np.median(lens):.0f} p90 {np.percentile(lens, 90):.0f} | "
              + (f"relevant/query mean {np.mean([len(r) for r in rel]):.2f} median {np.median([len(r) for r in rel]):.0f} max "
                 f"{max(len(r) for r in rel)} | relevant ids missing from pool {miss} | " if rel else "")
              + f"queries also in their pool {selfin}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)


def loaddata(cfg):
    return torch.load(path(cfg.pcr.datadir) / "data.pt", map_location="cpu", weights_only=False)


# ----------------------------------------------------------------------------- metric

def f1at7(ranked, rel):
    """Micro P / R / F1 @ 7 over queries with labels (the official definition)."""
    hit = tot_p = tot_g = 0
    for q, r in rel.items():
        top = ranked[q][:K]
        g = set(r)
        hit += len(set(top) & g)
        tot_p += len(top)
        tot_g += len(g)
    p, rr = hit / max(1, tot_p), hit / max(1, tot_g)
    return {"f1": 2 * p * rr / (p + rr) if p + rr else 0.0, "p": p, "r": rr}


def topk(scores, cids, qid, k=K):
    """Top-k unique candidate ids from a score vector over the pool, never the query itself."""
    order = np.argsort(-scores)
    out = []
    for j in order:
        if cids[j] != qid:
            out.append(cids[j])
            if len(out) == k:
                break
    return out


# ----------------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25 over word uni- and bi-grams (stop words removed); k1, b standard."""

    def __init__(self, texts, k1=1.2, b=0.75, bigrams=True):
        self.k1, self.b, self.bi = k1, b, bigrams
        self.post = defaultdict(list)
        self.dl = np.zeros(len(texts), dtype=np.float32)
        for j, t in enumerate(texts):
            c = Counter(self.terms(t))
            self.dl[j] = sum(c.values())
            for w, n in c.items():
                self.post[w].append((j, n))
        self.N, self.avg = len(texts), float(self.dl.mean() or 1.0)
        self.idf = {w: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.post.items()}

    def terms(self, t):
        w = words(t)
        return w + ([a + "_" + b for a, b in zip(w, w[1:])] if self.bi else [])

    def scores(self, q):
        s = np.zeros(self.N, dtype=np.float32)
        for w, qn in Counter(self.terms(q)).items():
            p = self.post.get(w)
            if not p:
                continue
            idf = self.idf[w]
            for j, n in p:
                s[j] += idf * n * (self.k1 + 1) / (n + self.k1 * (1 - self.b + self.b * self.dl[j] / self.avg))
        return s


def bm25(cfg):
    """BM25 score matrices (query x pool) for every split -> runs/pcr/bm25.pt; dev F1@7 of BM25 alone; hard negatives."""
    c = cfg.pcr
    data = loaddata(cfg)
    out = {}
    for s in SPLITS:
        d = data[s]
        t0 = time.time()
        bm = BM25(d["ctext"], float(c.get("k1", 1.2)), float(c.get("b", 0.75)), bool(c.get("bigrams", True)))
        out[s] = np.stack([bm.scores(q) for q in d["qtext"]]).astype(np.float32)
        msg = f"{s}: BM25 {out[s].shape} in {time.time() - t0:.0f}s"
        if d["rel"][0] is not None:
            ranked = {q: topk(out[s][i], d["cids"], q) for i, q in enumerate(d["qids"])}
            m = f1at7(ranked, dict(zip(d["qids"], d["rel"])))
            msg += f" | BM25 alone F1@7 {m['f1']:.4f} P {m['p']:.4f} R {m['r']:.4f}"
        print(msg, flush=True)
    torch.save(out, path(c.datadir) / "bm25.pt")


# ----------------------------------------------------------------------------- dense retriever

class Encoder(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.enc.cls = nn.Identity()
        self.logt = nn.Parameter(torch.tensor(math.log(1 / 0.05)))     # learned inverse temperature (init 20)

    def forward(self, ids):
        return F.normalize(self.enc.hidden(ids).float().mean(1), dim=-1)


def build(cfg, dev):
    t, info = ner.make(wrap({**cfg, "ner": cfg.pcr}), dev, ["x"])
    return Encoder(t).to(dev), info


def loadenc(run, dev):
    obj = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    m, _ = build(wrap(obj["cfg"]), dev)
    m.load_state_dict(obj["model"])
    return m.eval(), obj


def chunk(tok, W, j=None, rng=None):
    """One W-token chunk: the j-th of the evenly spaced chunks, or a random start; short docs are tiled (no padding)."""
    n = len(tok)
    if n < W:
        reps = W // max(1, n) + 1
        return np.tile(tok if n else np.array([3], dtype=np.int16), reps)[:W]
    if rng is not None:
        s = rng.randrange(0, n - W + 1)
    else:
        s = 0 if j is None else j
    return tok[s:s + W]


def starts(n, W, M):
    if n <= W:
        return [0]
    k = min(M, math.ceil(n / W))
    return [round(i * (n - W) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]


@torch.no_grad()
def embed(m, toks, c, dev, B=32):
    """Document embeddings: normalised mean of the embeddings of up to `nchunks` evenly spaced W-token chunks."""
    W, M = int(c.window), int(c.get("nchunks", 4))
    jobs = [(i, chunk(t, W, s_)) for i, t in enumerate(toks) for s_ in starts(len(t), W, M)]
    sums, cnt = None, torch.zeros(len(toks), device=dev)
    was = m.training
    m.eval()
    for a in range(0, len(jobs), B):
        g = jobs[a:a + B]
        x = torch.tensor(np.stack([j[1] for j in g]).astype(np.int64), device=dev)
        with autocast(dev, c.get("precision", "bf16")):
            e = m(x)
        if sums is None:
            sums = torch.zeros(len(toks), e.size(-1), device=dev)
        idx = torch.tensor([j[0] for j in g], device=dev)
        sums.index_add_(0, idx, e.float())
        cnt.index_add_(0, idx, torch.ones(len(g), device=dev))
    m.train(was)
    return F.normalize(sums / cnt[:, None], dim=-1)


def dense_scores(m, d, c, dev):
    Q, C = embed(m, d["qtok"], c, dev), embed(m, d["ctok"], c, dev)
    return (Q @ C.T).float().cpu().numpy()


def scorefn(m, d, c, dev):
    if int(c.get("milk", 1)) > 1:
        import pcr4
        return pcr4.mscores(m, d, c, dev, int(c.get("nchunks", 6)))
    return dense_scores(m, d, c, dev)


def zs(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def fusetune(S_list, w_grid, d):
    """Choose fusion weights on a labelled split: score = sum_k w_k * z(S_k); returns (weights, F1@7)."""
    rel = dict(zip(d["qids"], d["rel"]))
    Z = [zs(S) for S in S_list]
    best = None
    for w in w_grid:
        S = sum(wk * z for wk, z in zip(w, Z))
        ranked = {q: topk(S[i], d["cids"], q) for i, q in enumerate(d["qids"])}
        m = f1at7(ranked, rel)
        if best is None or m["f1"] > best[1]["f1"]:
            best = (w, m)
    return best


def train(cfg):
    c = cfg.pcr
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = loaddata(cfg)
    B25 = torch.load(path(c.datadir) / "bm25.pt", map_location="cpu", weights_only=False)
    tr, dv = data["train"], data["dev"]
    cidx = {x: j for j, x in enumerate(tr["cids"])}
    pairs = [(i, cidx[p]) for i, r in enumerate(tr["rel"]) for p in r if p in cidx]
    # hard negatives: top BM25 candidates that are not relevant
    H = int(c.get("hardk", 30))
    hard = []
    for i, r in enumerate(tr["rel"]):
        rs = set(r)
        order = np.argsort(-B25["train"][i])
        hard.append([j for j in order[:H + len(rs) + 1] if tr["cids"][j] not in rs and tr["cids"][j] != tr["qids"][i]][:H])
    m, info = build(cfg, dev)
    teacher = None
    if c.get("kd", {}).get("teacher"):
        teacher, _ = loadenc(path(c.kd.teacher), dev)
        for p in teacher.parameters():
            p.requires_grad = False
    groups = [{"params": [p for n_, p in m.named_parameters() if p.requires_grad and n_.startswith("enc.backbone.")], "lr": float(c.lr)},
              {"params": [p for n_, p in m.named_parameters() if p.requires_grad and not n_.startswith("enc.backbone.")], "lr": float(c.headlr)}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=float(c.get("wd", 0.01)))
    B, E, W = int(c.batch), float(c.epochs), int(c.window)
    steps = max(1, int(len(pairs) / B * E))
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    tot, trn = count(m)
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | {info['phase1_model']} from {info['phase1']} | method {c.method} | window {W} x {c.get('nchunks', 4)} chunks | "
          f"params {tot / 1e6:.1f}M, trainable {trn / 1e6:.2f}M ({100 * trn / tot:.2f}%) | {len(pairs)} (query, relevant) pairs, "
          f"{len(tr['qids'])} queries, pool {len(tr['cids'])} | {steps} steps" + (" | KD" if teacher is not None else ""), flush=True)
    pr = c.get("precision", "bf16")
    T, alpha = float(c.get("kd", {}).get("T", 1.0)), float(c.get("kd", {}).get("alpha", 0.5))
    every = int(c.get("evalevery", 0)) or max(1, steps // 4)
    best, step, t0, tl = -1.0, 0, time.time(), 0.0
    nh = int(c.get("nhard", 1))
    while step < steps:
        rng.shuffle(pairs)
        for a in range(0, len(pairs) - B + 1, B):
            batch = pairs[a:a + B]
            mk = int(c.get("milk", 1))      # chunks per document for multiple-instance (MaxSim) training
            q = [chunk(tr["qtok"][i], W, rng=rng) for i, _ in batch for _ in range(mk)]
            pos = [chunk(tr["ctok"][j], W, rng=rng) for _, j in batch for _ in range(mk)]
            negd = [rng.choice(hard[i]) for i, _ in batch for _ in range(nh) if hard[i]]
            neg = [chunk(tr["ctok"][j], W, rng=rng) for j in negd for _ in range(mk)]
            x = torch.tensor(np.stack(q + pos + neg).astype(np.int64), device=dev)
            with autocast(dev, pr):
                e = m(x)
            if mk > 1:
                assert teacher is None, "KD is implemented for single-chunk students"
                e = e.float().view(-1, mk, e.size(-1))
                eq, ec = e[:B], e[B:]
                sim = torch.einsum("aid,bjd->abij", eq, ec)
                ms = 0.5 * (sim.max(-1).values.mean(-1) + sim.max(-2).values.mean(-1))     # symmetric chunk MaxSim
                lg = m.logt.exp().clamp(max=100) * ms
            else:
                eq, ec = e[:B], e[B:]
                # in-batch + hard negatives; duplicates of the same relevant candidate are not penalised
                lg = m.logt.exp().clamp(max=100) * eq @ ec.T
            same = torch.tensor([[batch[r][0] == batch[k][0] and k != r for k in range(B)] + [False] * (len(ec) - B)
                                 for r in range(B)], device=dev)
            lg = lg.masked_fill(same, -1e4)
            loss = F.cross_entropy(lg, torch.arange(B, device=dev))
            if teacher is not None:
                with torch.no_grad(), autocast(dev, pr):
                    te = teacher(x)
                tl_ = (teacher.logt.exp().clamp(max=100) * te[:B].float() @ te[B:].float().T).masked_fill(same, -1e4)
                loss = alpha * loss + (1 - alpha) * F.kl_div(F.log_softmax(lg / T, -1), F.softmax(tl_ / T, -1),
                                                             reduction="batchmean") * T * T
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tl += float(loss.detach())
            if step % 25 == 0:
                print(f"step {step}/{steps} | loss {tl / 25:.4f} | temp {1 / m.logt.exp().item():.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                tl = 0.0
            if step % every == 0 or step >= steps:
                S = scorefn(m, dv, c, dev)
                dense = fusetune([S], [(1.0,)], dv)[1]
                w, fused = fusetune([S, B25["dev"]], [(1 - a_, a_) for a_ in np.linspace(0, 1, 11)], dv)
                print(f"eval step {step}: dev dense F1@7 {dense['f1']:.4f} | fused with BM25 (w_bm25 {w[1]:.1f}) F1@7 {fused['f1']:.4f} "
                      f"P {fused['p']:.4f} R {fused['r']:.4f}", flush=True)
                if fused["f1"] > best:
                    best = fused["f1"]
                    t1, n1 = count(m)
                    torch.save({"model": m.state_dict(), "cfg": plain(cfg), "step": step, "dev_dense": dense["f1"],
                                "dev_fused": fused["f1"], "params": t1, "trainable": n1,
                                "info": {k: v for k, v in info.items() if k != "mcfg"}}, run / "best.pt")
                    writejson(run / "best.json", {"name": cfg.name, "id": cfg.id, "method": c.method, "step": step,
                                                  "dev_dense": dense["f1"], "dev_fused": fused["f1"], "params": t1,
                                                  "trainable": n1, "model": info["phase1_model"]}, 1)
            if step >= steps:
                break
    print(f"best dev fused F1@7 {best:.4f}")


def writesub(cfg, d, S, meta):
    ranked = {q: topk(S[i], d["cids"], q) for i, q in enumerate(d["qids"])}
    pool = set(d["cids"])
    for q, r in ranked.items():
        assert len(r) == K and len(set(r)) == K and all(x in pool for x in r) and q not in r, q
    sd = subdir(cfg)
    writejson(sd / "prediction.json", ranked, 1)
    with zipfile.ZipFile(sd / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sd / "prediction.json", arcname="prediction.json")
    writejson(sd / "meta.json", {"id": cfg.id, "name": cfg.name, **meta, "test_queries": len(ranked)}, 1)
    print(f"ok: {len(ranked)} test queries x {K} -> {sd / 'submission.zip'}")


def officialdev(cfg, run, d, S):
    import subprocess
    ev = path(cfg.get("evaluator", "evaluation_script/evaluate_phase7.py"))
    if not ev.exists():
        return
    ranked = {q: topk(S[i], d["cids"], q) for i, q in enumerate(d["qids"])}
    writejson(run / "dev_truth.json", {q: {"relevant_candidates": r} for q, r in zip(d["qids"], d["rel"])})
    writejson(run / "dev_pred.json", ranked)
    writejson(run / "dev_cands.json", d["cids"])
    cmd = [sys.executable, str(ev), "--truth", str(run / "dev_truth.json"), "--prediction", str(run / "dev_pred.json"),
           "--candidate-ids", str(run / "dev_cands.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("official evaluator on dev: " + " | ".join(x.strip() for x in (r.stdout + r.stderr).splitlines() if x.strip())[-400:])


def predict(cfg):
    """Dense scores for dev / test (saved for fusion), dev-tuned BM25 fusion, submission."""
    c = cfg.pcr
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = loaddata(cfg)
    B25 = torch.load(path(c.datadir) / "bm25.pt", map_location="cpu", weights_only=False)
    run = path(f"runs/pcr/{cfg.name}")
    m, obj = loadenc(run, dev)
    t0 = time.time()
    S = {s: scorefn(m, data[s], c, dev) for s in ("dev", "test")}
    enc_time = time.time() - t0
    torch.save(S, run / "scores.pt")
    dense = fusetune([S["dev"]], [(1.0,)], data["dev"])[1]
    w, fused = fusetune([S["dev"], B25["dev"]], [(1 - a_, a_) for a_ in np.linspace(0, 1, 21)], data["dev"])
    print(f"dev: dense F1@7 {dense['f1']:.4f} | fused (w_bm25 {w[1]:.2f}) F1@7 {fused['f1']:.4f} P {fused['p']:.4f} R {fused['r']:.4f} | "
          f"encoding dev + test pools {enc_time:.0f}s")
    Sd = w[0] * zs(S["dev"]) + w[1] * zs(B25["dev"])
    officialdev(cfg, run, data["dev"], Sd)
    St = w[0] * zs(S["test"]) + w[1] * zs(B25["test"])
    writesub(cfg, data["test"], St, {"method": c.method, "params": obj["params"], "trainable": obj["trainable"],
                                     "w_bm25": float(w[1]), "dev_dense_f1": dense["f1"], "dev_fused_f1": fused["f1"],
                                     "dev_p": fused["p"], "dev_r": fused["r"], "encode_seconds": enc_time})


def fuse(cfg):
    """Multi-retriever fusion: BM25 + several dense members, weights by greedy coordinate search on dev F1@7."""
    e = cfg.ens
    data = loaddata(cfg)
    B25 = torch.load(path(cfg.pcr.datadir) / "bm25.pt", map_location="cpu", weights_only=False)
    names, mats = ["bm25"], [B25]
    for r in e.runs:
        p = path(r) / "scores.pt"
        if p.exists():
            names.append(Path(r).name)
            mats.append(torch.load(p, map_location="cpu", weights_only=False))
        else:
            print(f"skip {r}: no scores.pt")
    dv = data["dev"]
    Z = [zs(mm["dev"]) for mm in mats]
    rel = dict(zip(dv["qids"], dv["rel"]))
    score = lambda w: f1at7({q: topk(sum(wk * z[i] for wk, z in zip(w, Z)), dv["cids"], q) for i, q in enumerate(dv["qids"])}, rel)["f1"]
    w = [1.0] * len(mats)
    best = score(w)
    print("singles: " + ", ".join(f"{n} {score([1.0 if k == j else 0.0 for k in range(len(mats))]):.4f}" for j, n in enumerate(names)))
    for _ in range(2):
        for j in range(len(mats)):
            for v in e.get("wgrid", [0.0, 0.5, 1.5, 2.0, 3.0]):
                t_ = list(w)
                t_[j] = v
                if sum(t_) == 0:
                    continue
                f_ = score(t_)
                if f_ > best + 1e-4:
                    best, w = f_, t_
    print(f"weights {dict(zip(names, w))}: dev F1@7 {best:.4f}")
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    Sd = sum(wk * zs(mm["dev"]) for wk, mm in zip(w, mats))
    m_ = f1at7({q: topk(Sd[i], dv["cids"], q) for i, q in enumerate(dv["qids"])}, rel)
    officialdev(cfg, run, dv, Sd)
    St = sum(wk * zs(mm["test"]) for wk, mm in zip(w, mats))
    writesub(cfg, data["test"], St, {"method": "BM25 + dense multi-retriever fusion", "members": names, "weights": w,
                                     "dev_f1": m_["f1"], "dev_p": m_["p"], "dev_r": m_["r"]})


def main():
    cfg, modes = args()
    {"download": download, "prep": prep, "bm25": bm25, "train": train, "predict": predict, "fuse": fuse}[modes[0] if modes else "prep"](cfg)


if __name__ == "__main__":
    main()
