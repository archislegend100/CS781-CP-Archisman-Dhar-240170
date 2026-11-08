import math

import torch.nn as nn

from .common import RMSNorm, lmloss


class LegalGRU(nn.Module):
    """Recurrent baseline derived from Assignment 2 (Q4, nn.GRU path): stacked GRU LM, tied head."""
    family = "causal"

    def __init__(self, m):
        super().__init__()
        self.emb = nn.Embedding(m.vocab, m.dim)
        self.rnn = nn.GRU(m.dim, m.dim, m.layers, batch_first=True, dropout=m.get("dropout", 0.0))
        self.norm = RMSNorm(m.dim)
        self.head = nn.Linear(m.dim, m.vocab, bias=False)
        self.head.weight = self.emb.weight
        self.ckpt = False
        nn.init.normal_(self.emb.weight, 0.0, m.get("init", 0.02))
        for name, p in self.rnn.named_parameters():
            if name.startswith("weight_hh"):
                for g in p.data.chunk(3, 0):
                    nn.init.orthogonal_(g)
            elif name.startswith("weight_ih"):
                nn.init.normal_(p, 0.0, 1 / math.sqrt(m.dim))
            else:
                nn.init.zeros_(p)

    def newcache(self):
        return {"pos": 0, "h": None}

    def forward(self, ids, targets=None, cache=None, hiddens=False, last=False):
        x, hn = self.rnn(self.emb(ids), cache["h"] if cache is not None else None)
        h = self.norm(x)
        logits = self.head(h[:, -1:] if last else h)
        out = {"logits": logits, "hidden": h}
        if targets is not None:
            out["loss"] = lmloss(logits, targets)
        if cache is not None:
            cache["h"] = hn
            cache["pos"] += ids.size(1)
        return out
