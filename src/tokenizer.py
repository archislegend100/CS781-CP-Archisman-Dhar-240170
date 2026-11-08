"""Tokenizers.

`BPE` loads the Assignment-1 byte-level BPE (bpe_en.json: merges, itos, vocab_size).
Encoding follows the A1 BytePairEncoder exactly: whitespace pre-tokenisation, "▁" prepended
to each word, UTF-8 bytes as base symbols, lowest-rank merge applied repeatedly.
`HFTok` wraps the GPT-2 tokenizer used by the official starter (BASE-0 only).
"""
import heapq
import json
import shutil
import sys
from collections import Counter, defaultdict

from config import args, fingerprint, path

MARK = "▁"
BYTES = [bytes([b]).decode("latin-1") for b in range(256)]
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
INF = float("inf")


class BPE:
    kind = "a1bpe"

    def __init__(self, merges, itos, cachesize=2_000_000):
        self.merges = [tuple(m) for m in merges]
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.pad, self.unk, self.bos, self.eos = (self.stoi[s] for s in SPECIALS)
        self.vocab = len(self.itos)
        self.cache = {}
        self.cachesize = cachesize
        self.fp = fingerprint({"merges": [list(m) for m in self.merges], "itos": self.itos})

    @classmethod
    def load(cls, p):
        d = json.load(open(path(p), encoding="utf-8"))
        return cls(d["merges"], d["itos"])

    def save(self, p):
        with open(path(p), "w", encoding="utf-8") as f:
            json.dump({"merges": [list(m) for m in self.merges], "itos": self.itos,
                       "vocab_size": self.vocab}, f, ensure_ascii=False)

    def word(self, w):
        ids = self.cache.get(w)
        if ids is not None:
            return ids
        syms = [BYTES[b] for b in (MARK + w).encode("utf-8")]
        ranks = self.ranks
        while len(syms) > 1:
            best = min(zip(syms, syms[1:]), key=lambda p: ranks.get(p, INF))
            if best not in ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        ids = [self.stoi.get(s, self.unk) for s in syms]
        if len(self.cache) >= self.cachesize:
            self.cache.clear()
        self.cache[w] = ids
        return ids

    def encode(self, text):
        out = []
        for w in text.split():
            out.extend(self.word(w))
        return out

    def decode(self, ids):
        s = "".join(self.itos[i] for i in ids if i >= len(SPECIALS))
        return s.encode("latin-1").decode("utf-8", errors="replace").replace(MARK, " ").strip()


class HFTok:
    kind = "hf"

    def __init__(self, name=None, file=None, text=None):
        from tokenizers import Tokenizer
        if text is not None:
            self.tok = Tokenizer.from_str(text)
        elif file and path(file).exists():
            self.tok = Tokenizer.from_file(str(path(file)))
        else:
            self.tok = Tokenizer.from_pretrained(name)
        self.vocab = self.tok.get_vocab_size()
        eos = self.tok.token_to_id("<|endoftext|>")
        self.pad = self.unk = self.bos = self.eos = eos
        self.fp = fingerprint(self.tok.to_str())

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True)


def gettok(cfg):
    t = cfg.tokenizer
    if t.kind == "hf":
        return HFTok(t.name, t.get("path"))
    p = path(t.path)
    if not p.exists():
        raise SystemExit(f"tokenizer missing: {p}  (run: python tokenizer.py {cfg.cfgpath} import)")
    tok = BPE.load(p)
    if t.get("fp") and tok.fp != t.fp:
        raise SystemExit(f"tokenizer fingerprint {tok.fp} != expected {t.fp}")
    return tok


def trainbpe(freq, size, log=print):
    """Assignment-1 heap BPE on a word-frequency dictionary D_init = {t_i: f_i}.

    Returns merges M (|M| = M_F) and base vocabulary V_B (specials + 256 bytes + merged symbols).
    """
    words = list(freq)
    freqs = [freq[w] for w in words]
    splits = [[BYTES[b] for b in (MARK + w).encode("utf-8")] for w in words]
    pairs = Counter()
    where = defaultdict(set)
    for i, s in enumerate(splits):
        for p in zip(s, s[1:]):
            pairs[p] += freqs[i]
            where[p].add(i)
    itos = SPECIALS + BYTES
    merges = []
    heap = [(-c, p) for p, c in pairs.items()]
    heapq.heapify(heap)
    while len(itos) < size:
        best = None
        while heap:
            c, p = heapq.heappop(heap)
            if pairs.get(p, 0) == -c > 0:
                best = p
                break
        if best is None or pairs[best] < 2:
            break
        a, b = best
        merged = a + b
        merges.append([a, b])
        itos.append(merged)
        touched = set()
        for i in list(where.get(best, ())):
            s, f = splits[i], freqs[i]
            for p in zip(s, s[1:]):
                pairs[p] -= f
                touched.add(p)
                where[p].discard(i)
                if pairs[p] <= 0:
                    pairs.pop(p, None)
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(merged)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            splits[i] = out
            for p in zip(out, out[1:]):
                pairs[p] += f
                where[p].add(i)
                touched.add(p)
        pairs.pop(best, None)
        where.pop(best, None)
        touched.discard(best)
        for p in touched:
            if pairs.get(p, 0) > 0:
                heapq.heappush(heap, (-pairs[p], p))
        if len(heap) > max(1000, 4 * len(pairs)):
            heap = [(-c, p) for p, c in pairs.items()]
            heapq.heapify(heap)
        if len(merges) % 1000 == 0:
            log(f"merges {len(merges)}")
    return merges, itos


def selfcheck(tok, verbose=True):
    samples = [
        "The appellant was convicted under Section 302 of the Indian Penal Code, 1860.",
        "Hon'ble Mr. Justice A.K. Sikri - (2014) 5 SCC 1; Art. 226(1)(a) r/w Order XXI Rule 97 CPC.",
        "न्यायालय ने कहा",
        "  tabs\tand\nnewlines   collapse ",
        "",
    ]
    ok = True
    for s in samples:
        ids = tok.encode(s)
        back = tok.decode(ids)
        want = " ".join(s.split()) if tok.kind == "a1bpe" else s
        good = back == want and all(0 <= i < tok.vocab for i in ids)
        ok &= good
        if verbose:
            print(f"{'ok ' if good else 'BAD'} {len(s.split()):3d} words {len(ids):3d} tok  {back[:60]!r}")
    return ok


def main():
    cfg, modes = args()
    mode = modes[0] if modes else "check"
    t = cfg.tokenizer
    if mode == "import":
        src, dst = path(t.origin), path(t.path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
        a, b = BPE.load(src), BPE.load(dst)
        assert a.fp == b.fp, "copied tokenizer differs from origin"
        print(f"tokenizer {dst} fp {b.fp}")
    elif mode == "check":
        tok = gettok(cfg)
        print(f"tok: {tok.kind} vocab {tok.vocab} fp {tok.fp}")
        print(f"specials pad {tok.pad} unk {tok.unk} bos {tok.bos} eos {tok.eos}")
        if tok.kind == "a1bpe":
            assert tok.itos[:4] == SPECIALS
            assert tok.itos[4:260] == BYTES
            assert all(tok.itos[260 + i] == a + b for i, (a, b) in enumerate(tok.merges))
            assert len(set(tok.itos)) == len(tok.itos)
            assert BPE.load(t.path).fp == tok.fp
            print(f"|V_B| {tok.vocab}  M_F {len(tok.merges)}  bytes 256  specials {len(SPECIALS)}")
        ok = selfcheck(tok)
        print("tokenizer ok" if ok else "tokenizer FAILED")
        sys.exit(0 if ok else 1)
    elif mode == "train":
        from data import getdocs, loadsplit
        size = int(t.size)
        split = loadsplit(cfg)
        freq = Counter()
        words = 0
        quota = t.get("trainwords", 200_000_000) // len(cfg.data.sources)
        for src in cfg.data.sources:
            n = 0
            for _, _, text in getdocs(cfg, src, keys=split["train"][src]):
                ws = text.split()
                freq.update(ws)
                n += len(ws)
                if n >= quota:
                    break
            words += n
        if t.get("minfreq", 1) > 1:
            freq = Counter({w: c for w, c in freq.items() if c >= t.minfreq})
        print(f"D_init: {len(freq)} word types, {words} words")
        merges, itos = trainbpe(freq, size)
        BPE(merges, itos).save(t.path)
        print(f"saved {t.path}: |V_B| {len(itos)} M_F {len(merges)}")
    elif mode == "stats":
        from data import getdocs
        tok = gettok(cfg)
        for src in cfg.data.sources:
            w = n = 0
            for i, (_, _, text) in enumerate(getdocs(cfg, src)):
                w += len(text.split())
                n += len(tok.encode(text))
                if i >= 200:
                    break
            print(f"{src:14s} {n / max(w, 1):.3f} tok/word over {w} words")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
