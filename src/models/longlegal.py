import torch.nn as nn

from .common import Block, CausalLM, Rope, initweights


class LongLegal(CausalLM):
    """Local/global decoder. Most layers use causal sliding-window attention (window w,
    blockwise, O(T*w) memory); every `globalevery`-th layer is dense causal attention."""

    def __init__(self, m):
        super().__init__()
        kv = m.get("kvheads", m.qheads)
        self.setup(m.vocab, m.dim, m.get("tie", True), m.get("dropout", 0.0))
        g = m.globalevery
        self.kinds = ["global" if (i + 1) % g == 0 else "local" for i in range(m.layers)]
        self.blocks = nn.ModuleList(
            Block(m.dim, m.qheads, kv, m.ff, m.get("dropout", 0.0), window=0 if k == "global" else m.window)
            for k in self.kinds)
        self.rope = Rope(m.dim // m.qheads, m.get("ropebase", 10000.0), m.get("ropescale", 1.0))
        initweights(self, m.get("init", 0.02), m.layers)

    def extras(self, T, pos, device, dtype):
        return {"rot": self.rope(pos, T, device)}
