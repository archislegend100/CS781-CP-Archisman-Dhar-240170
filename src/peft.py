"""Parameter-efficient adaptation of the Phase-1 models.

LoRA   (Hu et al. 2021):                         h = W0 x + (alpha/r) B A x,         B = 0 at init
LoRMA+ (Bihany, Patel, Modi, Findings ACL 2025): h = ((alpha/r) B A + I) W0 x,       B = 0 at init
       computed in the paper's re-ordered form y + (alpha/r) B (A y) with y = W0 x, so the cost is O(d k b)
LoRMApi (same paper): h = I_pi(B A) W0 x, row i of B A cyclically rotated by i; B[:, 0] = 1, A[0, 0] = 1 (rest of A
       zero) makes I_pi(B A) = I at init. The d x d product is formed explicitly, so it is used on attention
       projections only (d <= 768 here).
Adapter (Houlsby et al. 2019): f(x) + U gelu(D f(x)) on each attention / FFN / mixer output, U = 0 at init
rsLoRA (Kalajdzievski 2023): LoRA with the rank-stabilised scale alpha/sqrt(r) instead of alpha/r.
QLoRA (Dettmers et al., NeurIPS 2023): the frozen W0 is stored as 4-bit NormalFloat (NF4) codes with blockwise (64)
       absmax scales, the scales themselves 8-bit quantised in blocks of 256 (double quantisation); W0 is dequantised
       on the fly in bf16 and a LoRA adapter is trained on top. Pure torch (no bitsandbytes), so the 4-bit storage is
       real but the matmul runs on the dequantised weight.
AdaLoRA (Zhang et al., ICLR 2023): h = W0 x + (alpha/r) P diag(lambda) Q x in SVD form with an orthogonality penalty on
       P and Q; a budget scheduler prunes singular values by smoothed sensitivity x uncertainty importance down to a
       target average rank.
BitFit (Ben Zaken et al., ACL 2022): only bias terms are trained. Our Phase-1 linears are bias-free, so a zero-initialised
       bias is added to every targeted projection (exact Phase-1 function at init) and the RMSNorm gains are trained.
IA3 (Liu et al., NeurIPS 2022): learned vectors rescale keys and values (output of kv / Hyena inp) and the FFN
       intermediate activation (input of down); initialised to ones.
All of them start as the exact Phase-1 function.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q", "kv", "o", "up", "down", "inp", "out")


class LoRA(nn.Module):
    def __init__(self, base, r=8, alpha=16, drop=0.0, rs=False):
        super().__init__()
        self.base, self.s = base, alpha / (math.sqrt(r) if rs else r)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.base(x) + self.s * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMA(nn.Module):
    def __init__(self, base, r=8, alpha=None):
        super().__init__()
        d = base.out_features
        self.base, self.s = base, (alpha or r) / r
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        return y + self.s * F.linear(F.linear(y, self.A), self.B)


class LoRMAPi(nn.Module):
    def __init__(self, base, r=8):
        super().__init__()
        d = base.out_features
        self.base = base
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.randn(d, r) * 0.02)
        with torch.no_grad():
            self.B[:, 0] = 1.0
            self.A[0, 0] = 1.0
        i = torch.arange(d)
        self.register_buffer("idx", (i[None] - i[:, None]) % d, persistent=False)

    def forward(self, x):
        y = self.base(x)
        M = (self.B @ self.A).gather(1, self.idx)
        return F.linear(y, M)


class IA3Out(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def addia3(model):
    a = wraplinears(model, IA3Out, ("kv", "inp"))
    b = wraplinears(model, IA3In, ("down",))
    return a + b


class Adapter(nn.Module):
    def __init__(self, inner, dim, bottleneck=64):
        super().__init__()
        self.inner = inner
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, 0.0, 0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, *a, **k):
        y = self.inner(*a, **k)
        return y + self.up(F.gelu(self.down(y)))


NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


class NF4Linear(nn.Module):
    """Frozen linear layer stored in 4-bit NormalFloat with double-quantised blockwise scales (QLoRA storage)."""

    def __init__(self, lin, block=64, block2=256):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        w = lin.weight.detach().float().reshape(-1)
        n = w.numel()
        pad = (-n) % block
        w = torch.cat([w, w.new_zeros(pad)]).view(-1, block)
        absmax = w.abs().amax(1).clamp(min=1e-8)
        mid = (NF4[1:] + NF4[:-1]) / 2
        codes = torch.bucketize((w / absmax[:, None]).clamp(-1, 1), mid).to(torch.uint8).view(-1)
        self.register_buffer("packed", codes[0::2] | (codes[1::2] << 4))      # two 4-bit codes per byte
        # double quantisation: absmax (fp32) -> int8 around its mean, one fp32 scale per block2 scales
        m = absmax.mean()
        a = absmax - m
        pad2 = (-a.numel()) % block2
        a = torch.cat([a, a.new_zeros(pad2)]).view(-1, block2)
        sc = a.abs().amax(1).clamp(min=1e-12) / 127
        self.register_buffer("qabs", torch.round(a / sc[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("sabs", sc)
        self.register_buffer("mabs", m.reshape(1))
        self.register_buffer("code", NF4.clone())
        self.n, self.nblocks, self.block = n, absmax.numel(), block
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    def weight_(self, dtype):
        codes = torch.stack([self.packed & 15, self.packed >> 4], 1).view(-1).long()
        absmax = ((self.qabs.float() * self.sabs[:, None]).view(-1)[:self.nblocks] + self.mabs)
        w = (self.code[codes].view(-1, self.block) * absmax[:, None]).view(-1)[:self.n]
        return w.view(self.out_features, self.in_features).to(dtype)

    def forward(self, x):
        return F.linear(x, self.weight_(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def qlora(model, targets, r=8, alpha=16, drop=0.0):
    """QLoRA: quantise each targeted frozen linear to NF4, then put a LoRA adapter on it."""
    n = 0
    for mod in list(model.modules()):
        if isinstance(mod, (LoRA, NF4Linear)):
            continue
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, LoRA(NF4Linear(child), r=r, alpha=alpha, drop=drop))
                n += 1
    return n


class SVDLinear(nn.Module):
    """AdaLoRA layer: W0 x + (alpha/r) P (lambda * mask) Q x."""

    def __init__(self, base, r=12, alpha=16):
        super().__init__()
        self.base, self.s, self.r = base, alpha / r, r
        self.P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.lam = nn.Parameter(torch.zeros(r))
        self.register_buffer("mask", torch.ones(r))

    def forward(self, x):
        return self.base(x) + self.s * F.linear(F.linear(x, self.Q) * (self.lam * self.mask), self.P)

    def orth(self):
        P, Q = self.P.float(), self.Q.float()
        I = torch.eye(self.r, device=P.device)
        return ((P.T @ P - I) ** 2).sum() + ((Q @ Q.T - I) ** 2).sum()


class AdaLoRAController:
    """Budget schedule of AdaLoRA: full rank r0 for `tinit` steps, cubic decrease of the total number of kept singular
    values to the target (average rank `target`) at step T - `tfinal`, then fixed. Importance of triplet i in a layer is
    s(P_:,i) + s(lambda_i) + s(Q_i,:), where s() = smoothed sensitivity |w * grad| times its smoothed uncertainty."""

    def __init__(self, model, target, total, tinit, tfinal, every=10, b1=0.85, b2=0.85):
        self.layers = [m for m in model.modules() if isinstance(m, SVDLinear)]
        self.r0 = self.layers[0].r if self.layers else 0
        self.b0, self.bT = self.r0 * len(self.layers), int(target * len(self.layers))
        self.total, self.tinit, self.tfinal, self.every, self.b1, self.b2 = total, tinit, tfinal, every, b1, b2
        self.ibar, self.ubar = {}, {}

    def budget(self, t):
        if t < self.tinit:
            return self.b0
        if t >= self.total - self.tfinal:
            return self.bT
        f = 1 - (t - self.tinit) / max(1, self.total - self.tfinal - self.tinit)
        return int(self.bT + (self.b0 - self.bT) * f ** 3)

    @torch.no_grad()
    def update(self, t):
        scores = []
        for k, m in enumerate(self.layers):
            parts = []
            for name, p in (("P", m.P), ("lam", m.lam), ("Q", m.Q)):
                if p.grad is None:
                    return
                i = (p * p.grad).abs().float()
                key = (k, name)
                if key not in self.ibar:
                    self.ibar[key], self.ubar[key] = i.clone(), torch.zeros_like(i)
                self.ibar[key] = self.b1 * self.ibar[key] + (1 - self.b1) * i
                self.ubar[key] = self.b2 * self.ubar[key] + (1 - self.b2) * (i - self.ibar[key]).abs()
                s = self.ibar[key] * self.ubar[key]
                parts.append(s.mean(0) if name == "P" else s if name == "lam" else s.mean(1))
            scores.append(sum(parts))
        if t % self.every or t < self.tinit:
            return
        b = self.budget(t)
        allsc = torch.cat(scores)
        if b >= allsc.numel():
            return
        thr = torch.kthvalue(allsc, allsc.numel() - b).values
        for m, sc in zip(self.layers, scores):
            m.mask.copy_((sc > thr).float())

    def penalty(self):
        return sum(m.orth() for m in self.layers) if self.layers else 0.0

    def ranks(self):
        return [int(m.mask.sum()) for m in self.layers]


class BiasAdd(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.b = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, x):
        return self.base(x) + self.b


def addbitfit(model, targets=TARGETS):
    n = wraplinears(model, BiasAdd, targets)
    g = 0
    for m in model.modules():
        if type(m).__name__ in ("RMSNorm", "LayerNorm"):
            for p in m.parameters():
                p.requires_grad = True
                g += p.numel()
    return n, g


def wraplinears(model, cls, targets=TARGETS, **kw):
    """Replace every nn.Linear child whose attribute name is in `targets` by cls(linear, **kw)."""
    n = 0
    for mod in list(model.modules()):
        if isinstance(mod, (LoRA, LoRMA, LoRMAPi, IA3Out, IA3In, SVDLinear, BiasAdd)):
            continue
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, cls(child, **kw))
                n += 1
    return n


def layers(backbone):
    """Transformer layers of a Phase-1 backbone (decoder blocks, or encoder layers of LegalSeq)."""
    return list(backbone.enc) if hasattr(backbone, "enc") else list(backbone.blocks)


def addadapters(backbone, dim, bottleneck=64, names=("attn", "mix", "ffn")):
    """Houlsby (after every attention / mixer / FFN sublayer) or, with names=("ffn",), Pfeiffer placement (after the
    FFN only; Pfeiffer et al., AdapterFusion, EACL 2021)."""
    n = 0
    for blk in layers(backbone):
        for name in names:
            if hasattr(blk, name):
                setattr(blk, name, Adapter(getattr(blk, name), dim, bottleneck))
                n += 1
    return n


def count(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr
