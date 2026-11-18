"""Meaning angles for Frost: every known word carries 64 whole-number angles (micro-degrees). Two words mean alike when their angles line up.
    sim(a, b)                 -1..1, mean of cos(angle difference) over the 64 angles
    nearest(word, vocab, k)   the k words of vocab whose angles line up best with word's, with their sim
Reads only the finished table. How the angles were made stays in the maker."""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); PATH = os.path.join(HERE, "..", "model", "meaning", "meaning.npz")


class Meaning:
    def __init__(self, path=None):
        z = np.load(path or os.environ.get("MEANING_TABLE") or PATH); self.words = [str(w) for w in z["words"]]; self.ix = {w: i for i, w in enumerate(self.words)}
        self.th = (z["angles"].astype(np.float32) / (360 * 10 ** 6)) * 2 * np.pi; self.cos = np.cos(self.th); self.sin = np.sin(self.th); self.n = self.th.shape[1]

    def has(self, w): return w in self.ix

    def sim(self, a, b):
        if a not in self.ix or b not in self.ix: return 0.0
        i, j = self.ix[a], self.ix[b]; return float(np.mean(self.cos[i] * self.cos[j] + self.sin[i] * self.sin[j]))

    def nearest(self, w, vocab, k=3, floor=0.0):
        """vocab: iterable of words. Returns [(word, sim)] best first, words without angles are skipped"""
        if w not in self.ix: return []
        cand = [v for v in vocab if v in self.ix and v != w]
        if not cand: return []
        idx = np.fromiter((self.ix[v] for v in cand), dtype=np.int64, count=len(cand)); i = self.ix[w]
        s = (self.cos[idx] * self.cos[i] + self.sin[idx] * self.sin[i]).mean(1); o = np.argsort(-s)[:k]
        return [(cand[j], float(s[j])) for j in o if s[j] >= floor]
