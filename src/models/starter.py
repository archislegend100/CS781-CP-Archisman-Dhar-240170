import torch
import torch.nn as nn

from .common import Block, CausalLM, initweights


class Starter(CausalLM):
    """GPT-2 layout as instantiated by the official starter (GPT2Config + GPT2LMHeadModel, random init):
    learned absolute positions, pre-LayerNorm, GELU(tanh) 4x MLP, biased MHA, dropout 0.1, tied head."""

    def __init__(self, m):
        super().__init__()
        drop = m.get("dropout", 0.1)
        self.setup(m.vocab, m.dim, True, drop, normkind="ln")
        self.wpe = nn.Embedding(m.ctx, m.dim)
        self.blocks = nn.ModuleList(Block(m.dim, m.qheads, m.qheads, 4 * m.dim, drop, normkind="ln", ffn="gelu", bias=True)
                                    for _ in range(m.layers))
        initweights(self, m.get("init", 0.02), m.layers)

    def embed(self, ids, pos):
        p = torch.arange(pos, pos + ids.size(1), device=ids.device)
        return self.emb(ids) + self.wpe(p)[None]
