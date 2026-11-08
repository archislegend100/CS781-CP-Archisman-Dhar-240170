import torch
import torch.nn as nn

from .common import Block, CausalLM, Rope, alibi, alibislopes, initweights


class LegalGPT(CausalLM):
    """Decoder-only: RMSNorm pre-norm, RoPE or ALiBi, GQA/MHA, SwiGLU, tied embeddings."""

    def __init__(self, m):
        super().__init__()
        kv = m.get("kvheads", m.qheads)
        self.setup(m.vocab, m.dim, m.get("tie", True), m.get("dropout", 0.0))
        self.blocks = nn.ModuleList(Block(m.dim, m.qheads, kv, m.ff, m.get("dropout", 0.0)) for _ in range(m.layers))
        self.pos = m.get("pos", "rope")
        if self.pos == "rope":
            self.rope = Rope(m.dim // m.qheads, m.get("ropebase", 10000.0), m.get("ropescale", 1.0))
        elif self.pos == "alibi":
            self.register_buffer("slopes", alibislopes(m.qheads), persistent=False)
        else:
            raise ValueError(self.pos)
        initweights(self, m.get("init", 0.02), m.layers)

    def extras(self, T, pos, device, dtype):
        if self.pos == "rope":
            return {"rot": self.rope(pos, T, device)}
        return {"mask": alibi(self.slopes, pos, T, pos + T, device, dtype)}
