import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Block, CausalLM, RMSNorm, Rope, SwiGLU, initweights, up


def fftconv(u, h):
    """Causal long convolution y[t] = sum_{s<=t} h[t-s] u[s]; u (B,D,T), h (D,T). Zero-padded to 2T, so no wrap-around."""
    T = u.size(-1)
    n = 2 * T
    y = torch.fft.irfft(torch.fft.rfft(up(u), n=n) * torch.fft.rfft(up(h), n=n), n=n)[..., :T]
    return y.type_as(u)


class Sin(nn.Module):
    def __init__(self, dim, w=1.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class Filter(nn.Module):
    """Implicit long filter (Hyena): h(t) = MLP(PE(t)) * exp(-|a_c| t), one channel per model dim.
    t is normalised by maxlen, so h(t) does not depend on the current sequence length."""

    def __init__(self, dim, maxlen, emb=33, hidden=64, fast=0.3, slow=1.5, target=1e-2):
        super().__init__()
        t = torch.arange(maxlen, dtype=torch.float32) / max(1, maxlen - 1)
        bands = (emb - 1) // 2
        f = torch.linspace(1e-4, bands - 1, bands)
        z = torch.cat([t[:, None], torch.cos(2 * math.pi * f * t[:, None]), torch.sin(2 * math.pi * f * t[:, None])], -1)
        self.register_buffer("z", z, persistent=False)
        self.register_buffer("t", t[:, None], persistent=False)
        a = torch.linspace(math.log(target) / slow, math.log(target) / fast, dim).abs()
        self.register_buffer("decay", a[None], persistent=False)
        self.mlp = nn.Sequential(nn.Linear(emb, hidden), Sin(hidden), nn.Linear(hidden, hidden), Sin(hidden),
                                 nn.Linear(hidden, dim, bias=False))
        self.maxlen = maxlen

    def forward(self, T):
        assert T <= self.maxlen, f"sequence {T} longer than filter length {self.maxlen}"
        with torch.autocast(self.z.device.type, enabled=False):
            h = self.mlp(self.z[:T]) * torch.exp(-self.t[:T] * self.decay)
        return h.transpose(0, 1)


class HyenaMixer(nn.Module):
    """Order-2 Hyena operator: (q, k, v) = shortconv(W x); y = q * (h * (k * v) + D * (k * v))."""

    def __init__(self, dim, maxlen, ks=3, femb=33, fhidden=64):
        super().__init__()
        self.dim, self.ks = dim, ks
        self.inp = nn.Linear(dim, 3 * dim, bias=False)
        self.short = nn.Conv1d(3 * dim, 3 * dim, ks, groups=3 * dim, bias=True)
        self.filter = Filter(dim, maxlen, femb, fhidden)
        self.D = nn.Parameter(torch.ones(dim))
        self.out = nn.Linear(dim, dim, bias=False)
        self.out.resid = True

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        z = self.inp(x).transpose(1, 2)
        prev = cache.get("z") if cache is not None else None
        if prev is None:
            prev = z.new_zeros(B, 3 * self.dim, self.ks - 1)
        zp = torch.cat([prev, z], -1)
        q, k, v = self.short(zp).chunk(3, 1)
        u = k * v
        pos = cache["pos"] if cache is not None and "u" in cache else 0
        if pos == 0:
            y = fftconv(u, self.filter(T))
            hist = u
        else:
            assert T == 1, "incremental hybrid decoding is one token at a time"
            hist = torch.cat([cache["u"], u], -1)
            h = self.filter(pos + 1).flip(-1)
            y = (up(hist) * h[None]).sum(-1, keepdim=True).type_as(u)
        y = q * (y + self.D[None, :, None] * u)
        if cache is not None:
            cache["u"], cache["z"], cache["pos"] = hist, zp[..., -(self.ks - 1):], pos + T
        return self.out(y.transpose(1, 2))


class MixBlock(nn.Module):
    def __init__(self, dim, ff, maxlen, ks, drop=0.0):
        super().__init__()
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.mix = HyenaMixer(dim, maxlen, ks)
        self.ffn = SwiGLU(dim, ff)
        self.drop = nn.Dropout(drop)

    def forward(self, x, ex, cache=None):
        x = x + self.drop(self.mix(self.n1(x), cache))
        return x + self.drop(self.ffn(self.n2(x)))


class LegalHybrid(CausalLM):
    """Hyena-style long-convolution mixers with a dense causal attention layer every `attnevery` layers."""

    def __init__(self, m):
        super().__init__()
        kv = m.get("kvheads", m.qheads)
        drop = m.get("dropout", 0.0)
        self.setup(m.vocab, m.dim, m.get("tie", True), drop)
        self.kinds = ["attn" if (i + 1) % m.attnevery == 0 else "conv" for i in range(m.layers)]
        self.blocks = nn.ModuleList(
            Block(m.dim, m.qheads, kv, m.ff, drop) if k == "attn" else MixBlock(m.dim, m.ff, m.seq, m.get("ks", 3), drop)
            for k in self.kinds)
        self.rope = Rope(m.dim // m.qheads, m.get("ropebase", 10000.0))
        initweights(self, m.get("init", 0.02), m.layers)
        for b in self.blocks:
            if isinstance(b, MixBlock):
                w = b.mix.short.weight
                nn.init.normal_(w, 0.0, 0.02)
                w.data[..., -1] += 1.0
                nn.init.zeros_(b.mix.short.bias)
                nn.init.ones_(b.mix.D)
                for lin in b.mix.filter.mlp:
                    if isinstance(lin, nn.Linear):
                        nn.init.normal_(lin.weight, 0.0, 1 / math.sqrt(lin.in_features))
                        if lin.bias is not None:
                            nn.init.zeros_(lin.bias)

    def extras(self, T, pos, device, dtype):
        return {"rot": self.rope(pos, T, device)}
