"""Qwen2 byte-level BPE tokenizer built straight from the GGUF metadata (stdlib re only)."""
import re

# stdlib `re` has no \p{L}/\p{N}: letter ~ [^\W\d_], digit ~ \d, "not letter/digit" ~ [\W_]. Close to Qwen2's pattern.
_PRETOK = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:(?![\r\n])[\W_])?[^\W\d_]+"
    r"|\d"
    r"| ?(?:(?!\s)[\W_])+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    def __init__(self, meta):
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.types = meta["tokenizer.ggml.token_type"]
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(" ", 1)): i for i, m in enumerate(meta["tokenizer.ggml.merges"])}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.eos = meta["tokenizer.ggml.eos_token_id"]
        self.special = sorted((t for t, ty in zip(self.tokens, self.types) if ty == 3), key=len, reverse=True)
        self._split = re.compile("(" + "|".join(re.escape(s) for s in self.special) + ")")
        self._cache = {}

    def _bpe(self, word):
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((parts[i], parts[i + 1]), 1 << 60), i) for i in range(len(parts) - 1)]
            r, i = min(pairs)
            if r == 1 << 60:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        self._cache[word] = parts
        return parts

    def encode(self, text):
        ids = []
        for chunk in self._split.split(text):
            if not chunk:
                continue
            if chunk in self.id_of and self.types[self.id_of[chunk]] == 3:
                ids.append(self.id_of[chunk]); continue
            for piece in _PRETOK.findall(chunk):
                mapped = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                ids.extend(self.id_of[p] for p in self._bpe(mapped))
        return ids

    def token_bytes(self, i):
        t = self.tokens[i]
        if self.types[i] == 3:
            return t.encode("utf-8")
        return bytes(self.u2b[c] for c in t)

    def decode(self, ids):
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")
