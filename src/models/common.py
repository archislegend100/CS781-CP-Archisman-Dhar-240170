import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    F.scaled_dot_product_attention(torch.zeros(1, 2, 1, 4), torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), enable_gqa=True)
    NATIVEGQA = True
except Exception:
    NATIVEGQA = False


def up(x):
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def sdpa(q, k, v, mask=None, causal=False, drop=0.0):
    """q (B,Hq,Tq,d), k/v (B,Hkv,Tk,d). Bool mask: True = attend. Float mask: additive bias."""
    if k.size(-3) != q.size(-3):
        # native GQA only where a flash kernel exists (sm80+); on older GPUs (e.g. T4) it falls back to the math kernel,
        # so expand K/V and let SDPA pick the memory-efficient kernel instead
        if NATIVEGQA and mask is None and q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 8:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=drop, enable_gqa=True)
        r = q.size(-3) // k.size(-3)
        k, v = k.repeat_interleave(r, -3), v.repeat_interleave(r, -3)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None, dropout_p=drop)


def causalmask(q0, tq, tk, device, window=0):
    i = torch.arange(q0, q0 + tq, device=device)[:, None]
    j = torch.arange(tk, device=device)[None]
    m = j <= i
    if window:
        m &= j > i - window
    return m


def bandmask(T, w, device):
    """Symmetric window |i - j| < w (non-causal local attention, used when a decoder is unmasked for tagging)."""
    i = torch.arange(T, device=device)
    return (i[None] - i[:, None]).abs() < w


def localattn(q, k, v, w):
    """Causal sliding-window attention, window w (keys i-w+1..i), in O(T*2w) memory.

    Queries are cut into blocks of w; each block attends to itself and the previous block.
    """
    B, H, T, D = q.shape
    if k.size(1) != H:
        r = H // k.size(1)
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    pad = (-T) % w
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    n = (T + pad) // w
    q, k, v = (t.reshape(B * H, n, w, D) for t in (q, k, v))
    kk = torch.cat([F.pad(k, (0, 0, 0, 0, 1, 0))[:, :-1], k], dim=2)
    vv = torch.cat([F.pad(v, (0, 0, 0, 0, 1, 0))[:, :-1], v], dim=2)
    i = torch.arange(w, device=q.device)[:, None]
    j = torch.arange(2 * w, device=q.device)[None]
    mask = ((j > i) & (j <= i + w)).expand(n, w, 2 * w).clone()
    mask[0, :, :w] = False
    out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask[None])
    return out.reshape(B, H, n * w, D)[:, :, :T]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = up(x)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


def norm(kind, dim):
    return RMSNorm(dim) if kind == "rms" else nn.LayerNorm(dim, eps=1e-5)


class Rope(nn.Module):
    """RoPE (Su et al. 2021); `scale` > 1 gives linear position interpolation (Chen et al. 2023)."""

    def __init__(self, hd, base=10000.0, scale=1.0):
        super().__init__()
        self.register_buffer("inv", 1.0 / base ** (torch.arange(0, hd, 2).float() / hd), persistent=False)
        self.scale = scale

    def forward(self, pos, T, device):
        t = torch.arange(pos, pos + T, device=device, dtype=torch.float32) / self.scale
        f = torch.outer(t, self.inv.to(device))
        return f.cos(), f.sin()


def rope(x, rot):
    cos, sin = rot
    a, b = up(x).chunk(2, -1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1).type_as(x)


def alibislopes(h):
    def pow2(n):
        s = 2 ** (-2 ** -(math.log2(n) - 3))
        return [s * s ** i for i in range(n)]

    if math.log2(h).is_integer():
        return torch.tensor(pow2(h))
    c = 2 ** math.floor(math.log2(h))
    return torch.tensor(pow2(c) + pow2(2 * c)[0::2][:h - c])


def alibi(slopes, q0, tq, tk, device, dtype):
    i = torch.arange(q0, q0 + tq, device=device)[:, None]
    j = torch.arange(tk, device=device)[None]
    b = -slopes.to(device)[:, None, None] * (i - j).float()
    return b.masked_fill(j > i, float("-inf"))[None].to(dtype)


class Attention(nn.Module):
    def __init__(self, dim, qheads, kvheads=None, bias=False, drop=0.0):
        super().__init__()
        kvheads = kvheads or qheads
        self.qh, self.kvh, self.hd = qheads, kvheads, dim // qheads
        self.q = nn.Linear(dim, qheads * self.hd, bias=bias)
        self.kv = nn.Linear(dim, 2 * kvheads * self.hd, bias=bias)
        self.o = nn.Linear(qheads * self.hd, dim, bias=bias)
        self.o.resid = True
        self.drop = drop

    def forward(self, x, rot=None, mask=None, causal=False, cache=None, ctx=None, window=0):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.qh, self.hd).transpose(1, 2)
        if ctx is not None and cache is not None and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            src = x if ctx is None else ctx
            k, v = self.kv(src).view(B, src.size(1), 2, self.kvh, self.hd).permute(2, 0, 3, 1, 4)
            if rot is not None and ctx is None:
                q, k = rope(q, rot), rope(k, rot)
            if cache is not None:
                if ctx is None and "k" in cache:
                    k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
                cache["k"], cache["v"] = (k[:, :, -window:], v[:, :, -window:]) if window else (k, v)
        drop = self.drop if self.training else 0.0
        tk = k.size(2)
        if window and not causal and cache is None:
            y = sdpa(q, k, v, mask=bandmask(T, window, x.device))
        elif window and T > 1 and tk == T:
            y = localattn(q, k, v, window)
        elif window and T == 1:
            y = sdpa(q, k[:, :, -window:], v[:, :, -window:])
        elif window:
            y = sdpa(q, k, v, mask=causalmask(tk - T, T, tk, x.device, window))
        elif mask is not None or not causal:
            y = sdpa(q, k, v, mask=mask, drop=drop)
        elif T == tk:
            y = sdpa(q, k, v, causal=True, drop=drop)
        elif T == 1:
            y = sdpa(q, k, v, drop=drop)
        else:
            y = sdpa(q, k, v, mask=causalmask(tk - T, T, tk, x.device), drop=drop)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim, ff, bias=False):
        super().__init__()
        self.up = nn.Linear(dim, 2 * ff, bias=bias)
        self.down = nn.Linear(ff, dim, bias=bias)
        self.down.resid = True

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class GeluMLP(nn.Module):
    def __init__(self, dim, ff, bias=True):
        super().__init__()
        self.up = nn.Linear(dim, ff, bias=bias)
        self.down = nn.Linear(ff, dim, bias=bias)
        self.down.resid = True

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class Block(nn.Module):
    """Pre-norm decoder block: x + Attn(N(x)), then x + FFN(N(x))."""

    def __init__(self, dim, qheads, kvheads, ff, drop=0.0, window=0, normkind="rms", ffn="swiglu", bias=False):
        super().__init__()
        self.n1 = norm(normkind, dim)
        self.attn = Attention(dim, qheads, kvheads, bias=bias, drop=drop)
        self.n2 = norm(normkind, dim)
        self.ffn = SwiGLU(dim, ff, bias) if ffn == "swiglu" else GeluMLP(dim, ff, bias)
        self.drop = nn.Dropout(drop)
        self.window = window
        self.causal = True

    def forward(self, x, ex, cache=None):
        x = x + self.drop(self.attn(self.n1(x), rot=ex.get("rot"), mask=ex.get("mask"), causal=self.causal,
                                    cache=cache, window=self.window))
        return x + self.drop(self.ffn(self.n2(x)))


def lmloss(logits, targets):
    return F.cross_entropy(up(logits).reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)


def shift(ids, pad=-100):
    """Next-token targets for ids: targets[t] = ids[t+1], last position ignored."""
    return torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], pad)], 1)


def initweights(model, std=0.02, layers=1):
    """N(0, std) for linear/embedding weights, std/sqrt(2L) for residual output projections
    (GPT-2 scheme), zero biases, unit norm gains."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, std / math.sqrt(2 * layers) if getattr(m, "resid", False) else std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, std)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def countparams(model):
    """Unique parameters (a tied embedding/head matrix is counted once)."""
    total = sum(p.numel() for p in model.parameters())
    emb = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Embedding))
    return {"total": total, "embedding": emb, "nonembedding": total - emb}


class CausalLM(nn.Module):
    family = "causal"

    def setup(self, vocab, dim, tie=True, drop=0.0, normkind="rms"):
        self.emb = nn.Embedding(vocab, dim)
        self.norm = norm(normkind, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.edrop = nn.Dropout(drop)
        self.tied = tie
        self.ckpt = False
        if tie:
            self.head.weight = self.emb.weight

    def embed(self, ids, pos):
        return self.emb(ids)

    def extras(self, T, pos, device, dtype):
        return {}

    def newcache(self):
        return {"pos": 0, "layers": [{} for _ in self.blocks]}

    def forward(self, ids, targets=None, cache=None, hiddens=False, last=False):
        T = ids.size(1)
        pos = cache["pos"] if cache is not None else 0
        x = self.edrop(self.embed(ids, pos))
        ex = self.extras(T, pos, ids.device, x.dtype)
        ex["pos"] = pos
        hs = [x] if hiddens else None
        for i, blk in enumerate(self.blocks):
            c = cache["layers"][i] if cache is not None else None
            if self.ckpt and self.training and c is None:
                x = checkpoint(blk, x, ex, use_reentrant=False)
            else:
                x = blk(x, ex, c)
            if hiddens:
                hs.append(x)
        h = self.norm(x)
        logits = self.head(h[:, -1:] if last else h)
        out = {"logits": logits, "hidden": h}
        if hiddens:
            out["hiddens"] = hs
        if targets is not None:
            out["loss"] = lmloss(logits, targets)
        if cache is not None:
            cache["pos"] = pos + T
        return out
