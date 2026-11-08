from .common import countparams
from .legalgpt import LegalGPT
from .legalgru import LegalGRU
from .legalhybrid import LegalHybrid
from .legalseq import LegalSeq
from .longlegal import LongLegal
from .starter import Starter

MODELS = {"starter": Starter, "legalgpt": LegalGPT, "legalseq": LegalSeq, "longlegal": LongLegal,
          "legalhybrid": LegalHybrid, "legalgru": LegalGRU}


def build(m):
    return MODELS[m.name](m)
