"""Phase 7, round 6: chunk-pair cross-encoder re-ranker on a Phase-1 model.

monoBERT-style cross-encoding (Nogueira & Cho 2019) of long judgments via passage selection (BERT-MaxP, Dai & Callan
2019; PARADE, Li et al. 2023): a trained bi-encoder (P11) embeds every chunk; for a (query, candidate) pair the `npairs`
most similar chunk pairs are concatenated  [query chunk half | EOS | candidate chunk half]  (512 tokens) and scored by a
Phase-1 encoder (LoRMA+, causal mask removed, mean pooling + linear); the pair score is the max over its chunk pairs.
Training: localized contrastive estimation (Gao et al. ECIR 2021) -- each relevant candidate against `nneg` hard negatives
from the fused BM25 + dense ranking of the train split. Inference: re-score the top-`topk` candidates of a first-stage
ranking (the best cross-fitted ranker); output = dev-tuned fusion with that ranking; the scores are also saved as a view
for the cross-fitted LTR.

python pcr5.py phase7/configs/p25_ce.yaml prep | train | predict
"""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner
import pcr
import pcr2
from config import args, path, plain, wrap
from data import writejson
from generate import autocast
from peft import count


@torch.no_grad()
def prep(cfg):
    """Chunk embeddings of all splits with the selector retriever + first-stage candidate lists -> runs/pcr/ce_prep.pt."""
    e = cfg.ce
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_p = path(cfg.pcr.datadir) / "ce_prep.pt"
    if out_p.exists():
        print(f"exists: {out_p}")
        return
    data = pcr.loaddata(cfg)
    m, obj = pcr.loadenc(path(e.selector), dev)
    c = wrap(obj["cfg"]).pcr
    W, M = int(c.window), int(e.get("maxchunks", 8))
    out = {}
    for s in pcr.SPLITS:
        out[s] = {}
        for side in ("q", "c"):
            toks = data[s][f"{side}tok"]
            starts = [pcr.starts(len(t), W, M) for t in toks]
            jobs = [(i, st) for i, ss in enumerate(starts) for st in ss]
            E = []
            for a in range(0, len(jobs), 32):
                x = torch.tensor(np.stack([pcr.chunk(toks[i], W, st) for i, st in jobs[a:a + 32]]).astype(np.int64), device=dev)
                with autocast(dev, "bf16"):
                    E.append(m(x).float().half().cpu())
            out[s][side] = {"E": torch.cat(E), "doc": np.array([j[0] for j in jobs]), "start": np.array([j[1] for j in jobs])}
        print(f"{s}: chunks q {len(out[s]['q']['doc'])} c {len(out[s]['c']['doc'])}", flush=True)
    # train hard negatives: fused BM25 + mean-of-chunks dense ranking
    B25 = torch.load(path(cfg.pcr.datadir) / "bm25.pt", map_location="cpu", weights_only=False)
    tr = data["train"]

    def docemb(side):
        E, doc = out["train"][side]["E"].float(), torch.tensor(out["train"][side]["doc"])
        n = int(doc.max()) + 1
        S = torch.zeros(n, E.size(1)).index_add_(0, doc, E)
        return F.normalize(S, dim=-1)
    D = (docemb("q") @ docemb("c").T).numpy()
    Fz = pcr2.zr(D) + pcr2.zr(B25["train"])
    hard = []
    for i, r in enumerate(tr["rel"]):
        rs = set(r)
        order = np.argsort(-Fz[i])
        hard.append([int(j) for j in order[:int(e.get("hardk", 50)) + len(rs) + 1] if tr["cids"][j] not in rs and tr["cids"][j] != tr["qids"][i]][:int(e.get("hardk", 50))])
    out["hard"] = hard
    torch.save(out, out_p)
    print(f"ok -> {out_p}")


class CE(nn.Module):
    def __init__(self, enc, dim):
        super().__init__()
        self.enc = enc
        self.enc.cls = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 1))

    def forward(self, ids):
        return self.head(self.enc.hidden(ids).float().mean(1)).squeeze(-1)


def build(cfg, dev):
    t, info = ner.make(wrap({**cfg, "ner": cfg.pcr}), dev, ["x"])
    return CE(t, int(info["mcfg"]["dim"])).to(dev), info


class Pairs:
    """Builds the chunk-pair inputs of a (query, candidate) pair from the selector's chunk embeddings."""

    def __init__(self, P, data, s, W, L, npairs, eos):
        self.d, self.W, self.L, self.n, self.eos = data[s], W, L, npairs, eos
        self.q, self.c = P[s]["q"], P[s]["c"]
        self.qrows = self._rows(self.q["doc"])
        self.crows = self._rows(self.c["doc"])

    @staticmethod
    def _rows(doc):
        rows = {}
        for k, dd in enumerate(doc):
            rows.setdefault(int(dd), []).append(k)
        return rows

    def build(self, qi, ci):
        qr, cr = self.qrows[qi], self.crows[ci]
        sim = (self.q["E"][qr].float() @ self.c["E"][cr].float().T).flatten()
        top = torch.topk(sim, min(self.n, len(sim))).indices.tolist()
        h = self.L // 2
        out = []
        for t in top:
            a, b = qr[t // len(cr)], cr[t % len(cr)]
            qt = pcr.chunk(self.d["qtok"][qi], self.W, int(self.q["start"][a]))[:h]
            ct = pcr.chunk(self.d["ctok"][ci], self.W, int(self.c["start"][b]))[:self.L - h - 1]
            out.append(np.concatenate([qt, [self.eos], ct]).astype(np.int64))
        while len(out) < self.n:
            out.append(out[-1])
        return out


def score_pairs(m, PB, plist, dev, B=64):
    """Max-over-chunk-pairs scores for a list of (qi, ci)."""
    seqs = [x for qi, ci in plist for x in PB.build(qi, ci)]
    out = []
    m.eval()
    with torch.no_grad():
        for a in range(0, len(seqs), B):
            x = torch.tensor(np.stack(seqs[a:a + B]), device=dev)
            with autocast(dev, "bf16"):
                out.append(m(x).float().cpu())
    s = torch.cat(out).view(len(plist), PB.n)
    return s.max(1).values.numpy()


def firststage(cfg, s):
    return np.load(path(cfg.ce.firststage) / f"{s}_scores.npy")


def rescore(cfg, m, P, data, s, dev, tok):
    e = cfg.ce
    K = int(e.get("topk", 50))
    F0 = firststage(cfg, s)
    PB = Pairs(P, data, s, int(e.get("chunkw", 512)), int(e.get("seqlen", 512)), int(e.get("npairs", 2)), tok.eos)
    top = np.argsort(-F0, 1)[:, :K]
    sc = score_pairs(m, PB, [(i, int(j)) for i in range(len(top)) for j in top[i]], dev).reshape(top.shape)
    S = np.full(F0.shape, np.nan, dtype=np.float32)
    np.put_along_axis(S, top, sc, 1)
    S = np.where(np.isnan(S), np.nanmin(S, 1, keepdims=True) - 1.0, S)
    return S, F0, top


def fused(S, F0, top, a):
    z = lambda X: pcr2.zr(np.take_along_axis(X, top, 1))
    out = np.full(F0.shape, -1e9, dtype=np.float32)
    np.put_along_axis(out, top, z(S) + a * z(F0), 1)
    return out


def devf1(S, d):
    rel = dict(zip(d["qids"], d["rel"]))
    return pcr.f1at7({q: pcr.topk(S[i], d["cids"], q) for i, q in enumerate(d["qids"])}, rel)


def train(cfg):
    c, e = cfg.pcr, cfg.ce
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pcr.loaddata(cfg)
    from tokenizer import gettok
    from config import loadcfg
    tok = gettok(loadcfg(c.backbone.config))
    P = torch.load(path(c.datadir) / "ce_prep.pt", map_location="cpu", weights_only=False)
    tr = data["train"]
    cidx = {x: j for j, x in enumerate(tr["cids"])}
    groups = [(i, cidx[p]) for i, r in enumerate(tr["rel"]) for p in r if p in cidx]
    PB = Pairs(P, data, "train", int(e.get("chunkw", 512)), int(e.get("seqlen", 512)), int(e.get("npairs", 2)), tok.eos)
    m, info = build(cfg, dev)
    opt = torch.optim.AdamW([{"params": [p for n_, p in m.named_parameters() if p.requires_grad and n_.startswith("enc.backbone.")], "lr": float(c.lr)},
                             {"params": [p for n_, p in m.named_parameters() if p.requires_grad and not n_.startswith("enc.backbone.")], "lr": float(c.headlr)}],
                            weight_decay=0.01)
    G, nneg = int(e.get("groups", 2)), int(e.get("nneg", 7))
    steps = max(1, int(len(groups) / G * float(c.epochs)))
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    tot, trn = count(m)
    run = path(f"runs/pcr/{cfg.name}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name} | cross-encoder {info['phase1_model']} {c.method} | params {tot / 1e6:.1f}M trainable {trn / 1e6:.2f}M | "
          f"{len(groups)} groups (1 relevant + {nneg} hard negatives) x {PB.n} chunk pairs | {steps} steps", flush=True)
    every = max(1, steps // int(e.get("evals", 4)))
    best, step, t0, tl = -1.0, 0, time.time(), 0.0
    m.train()
    while step < steps:
        rng.shuffle(groups)
        for a in range(0, len(groups) - G + 1, G):
            seqs = []
            for qi, pj in groups[a:a + G]:
                rs = {cidx[x] for x in tr["rel"][qi] if x in cidx}
                negs = [j for j in P["hard"][qi] if j not in rs]
                negs = rng.sample(negs, min(nneg, len(negs))) if negs else []
                while len(negs) < nneg:
                    negs.append(rng.randrange(len(tr["cids"])))
                for cj in [pj] + negs:
                    seqs += PB.build(qi, cj)
            x = torch.tensor(np.stack(seqs), device=dev)
            with autocast(dev, "bf16"):
                s_ = m(x).float()
            s_ = s_.view(G, 1 + nneg, PB.n).max(-1).values
            loss = F.cross_entropy(s_, torch.zeros(G, dtype=torch.long, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tl += float(loss.detach())
            if step % 50 == 0:
                print(f"step {step}/{steps} | loss {tl / 50:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                tl = 0.0
            if step % every == 0 or step >= steps:
                S, F0, top = rescore(cfg, m, P, data, "dev", dev, tok)
                ce1 = devf1(fused(S, F0, top, 0.0), data["dev"])
                fb = max((devf1(fused(S, F0, top, a_), data["dev"])["f1"], a_) for a_ in (0.25, 0.5, 1.0, 2.0))
                print(f"eval step {step}: dev cross-encoder alone F1@7 {ce1['f1']:.4f} | fused with first stage (a={fb[1]}) {fb[0]:.4f} "
                      f"| first stage {devf1(F0, data['dev'])['f1']:.4f}", flush=True)
                m.train()
                if fb[0] > best:
                    best = fb[0]
                    torch.save({"model": m.state_dict(), "cfg": plain(cfg), "step": step, "dev_fused": fb[0], "dev_ce": ce1["f1"],
                                "params": tot, "trainable": trn}, run / "best.pt")
            if step >= steps:
                break
    print(f"best dev fused F1@7 {best:.4f}")


def predict(cfg):
    c = cfg.pcr
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pcr.loaddata(cfg)
    from tokenizer import gettok
    from config import loadcfg
    tok = gettok(loadcfg(c.backbone.config))
    P = torch.load(path(c.datadir) / "ce_prep.pt", map_location="cpu", weights_only=False)
    run = path(f"runs/pcr/{cfg.name}")
    obj = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    m, _ = build(wrap(obj["cfg"]), dev)
    m.load_state_dict(obj["model"])
    R = {s: rescore(cfg, m, P, data, s, dev, tok) for s in ("dev", "test")}
    torch.save({s: R[s][0] for s in ("dev", "test")}, run / "scores.pt")       # view for the LTR
    S, F0, top = R["dev"]
    fb = max((devf1(fused(S, F0, top, a_), data["dev"])["f1"], a_) for a_ in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    a = fb[1]
    print(f"dev: cross-encoder alone {devf1(fused(S, F0, top, 0.0), data['dev'])['f1']:.4f} | first stage {devf1(F0, data['dev'])['f1']:.4f} "
          f"| fused a={a}: {fb[0]:.4f}")
    Sd = fused(S, F0, top, a)
    np.save(run / "dev_scores.npy", Sd)
    St = fused(*R["test"], a)
    np.save(run / "test_scores.npy", St)
    pcr.officialdev(cfg, run, data["dev"], Sd)
    pcr.writesub(cfg, data["test"], St, {"method": "chunk-pair cross-encoder re-ranker fused with the first stage",
                                         "first_stage": str(cfg.ce.firststage), "fusion_a": a, "dev_f1": fb[0],
                                         "params": obj["params"], "trainable": obj["trainable"]})


def main():
    cfg, modes = args()
    {"prep": prep, "train": train, "predict": predict}[modes[0] if modes else "train"](cfg)


if __name__ == "__main__":
    main()
