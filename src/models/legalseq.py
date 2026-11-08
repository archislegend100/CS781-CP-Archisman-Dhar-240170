import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .common import Attention, RMSNorm, Rope, SwiGLU, initweights, lmloss


class EncLayer(nn.Module):
    def __init__(self, dim, heads, kv, ff, drop):
        super().__init__()
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, heads, kv, drop=drop)
        self.ffn = SwiGLU(dim, ff)
        self.drop = nn.Dropout(drop)

    def forward(self, x, rot, mask):
        x = x + self.drop(self.attn(self.n1(x), rot=rot, mask=mask))
        return x + self.drop(self.ffn(self.n2(x)))


class DecLayer(nn.Module):
    def __init__(self, dim, heads, kv, ff, drop):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.self = Attention(dim, heads, kv, drop=drop)
        self.cross = Attention(dim, heads, kv, drop=drop)
        self.ffn = SwiGLU(dim, ff)
        self.drop = nn.Dropout(drop)

    def forward(self, x, rot, enc, mask, cache=None):
        cs, cc = (cache["self"], cache["cross"]) if cache is not None else (None, None)
        x = x + self.drop(self.self(self.n1(x), rot=rot, causal=True, cache=cs))
        x = x + self.drop(self.cross(self.n2(x), mask=mask, ctx=enc, cache=cc))
        return x + self.drop(self.ffn(self.n3(x)))


class LegalSeq(nn.Module):
    """Encoder-decoder. Bidirectional encoder over the prefix x, causal decoder with cross-attention
    over the encoder states, teacher forcing: decoder input <s> y1..y(n-1), targets y1..yn."""
    family = "seq2seq"

    def __init__(self, m):
        super().__init__()
        kv = m.get("kvheads", m.qheads)
        drop = m.get("dropout", 0.0)
        self.emb = nn.Embedding(m.vocab, m.dim)
        self.enc = nn.ModuleList(EncLayer(m.dim, m.qheads, kv, m.ff, drop) for _ in range(m.enclayers))
        self.dec = nn.ModuleList(DecLayer(m.dim, m.qheads, kv, m.ff, drop) for _ in range(m.declayers))
        self.encnorm, self.decnorm = RMSNorm(m.dim), RMSNorm(m.dim)
        self.head = nn.Linear(m.dim, m.vocab, bias=False)
        self.head.weight = self.emb.weight
        self.edrop = nn.Dropout(drop)
        self.rope = Rope(m.dim // m.qheads, m.get("ropebase", 10000.0))
        self.ckpt = False
        initweights(self, m.get("init", 0.02), m.enclayers + m.declayers)

    def encode(self, src, srcmask, hiddens=False):
        x = self.edrop(self.emb(src))
        rot = self.rope(0, src.size(1), src.device)
        mask = srcmask[:, None, None, :]
        hs = [x] if hiddens else None
        for layer in self.enc:
            x = checkpoint(layer, x, rot, mask, use_reentrant=False) if self.ckpt and self.training else layer(x, rot, mask)
            if hiddens:
                hs.append(x)
        return self.encnorm(x), hs

    def newcache(self):
        return {"pos": 0, "layers": [{"self": {}, "cross": {}} for _ in self.dec]}

    def decode(self, ids, enc, srcmask, cache=None, hiddens=False, last=False):
        pos = cache["pos"] if cache is not None else 0
        x = self.edrop(self.emb(ids))
        rot = self.rope(pos, ids.size(1), ids.device)
        mask = srcmask[:, None, None, :]
        hs = [x] if hiddens else None
        for i, layer in enumerate(self.dec):
            c = cache["layers"][i] if cache is not None else None
            if self.ckpt and self.training and c is None:
                x = checkpoint(layer, x, rot, enc, mask, use_reentrant=False)
            else:
                x = layer(x, rot, enc, mask, c)
            if hiddens:
                hs.append(x)
        if cache is not None:
            cache["pos"] = pos + ids.size(1)
        h = self.decnorm(x)
        return self.head(h[:, -1:] if last else h), h, hs

    def forward(self, src, srcmask, ids, targets=None, hiddens=False):
        enc, ehs = self.encode(src, srcmask, hiddens)
        logits, h, dhs = self.decode(ids, enc, srcmask, hiddens=hiddens)
        out = {"logits": logits, "hidden": h, "enc": enc}
        if hiddens:
            out["enchiddens"], out["hiddens"] = ehs, dhs
        if targets is not None:
            out["loss"] = lmloss(logits, targets)
        return out
