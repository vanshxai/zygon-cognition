"""Neuron Path Locker.

Every reply Frost writes is a PATH: the exact token sequence plus the neurons it fired in every layer.
 * Approved replies are LOCKED.  Asking the same thing again returns the locked path from disk: no layer is run, a few milliseconds.
 * A changed ask (e.g. "make the button red" instead of "blue") BRANCHES: the model follows the locked path, verifying up to K of its
   tokens per pass (one batched forward instead of K single steps).  Where the new ask needs something else the follower falls off the
   path, the model generates the new part itself, and the follower REJOINS the old path as soon as the text lines up again.
   The result is exactly what the model would have written by itself (every token is sampled from the model's own output).
 * Every discovered path is indexed in model/path_locker/index.json (+ one .npz per path with tokens and neuron fingerprint).
   Rarely used / old paths are erased first when the index gets too big.
 * Code: code_index.py indexes the common and unique functions of the 10 programs with the neurons they fire; those function
   bodies are also used as drafts when Frost writes code.
numpy + stdlib only.
"""
import hashlib, json, os, re, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DIR = os.path.join(ROOT, "model", "path_locker")
FP_K = 48                                   # neurons kept per layer in a path fingerprint

STOP = set("a an the of to for in on at is are was were be been am do does did please can could would will you me my i we us it its this that "
           "what whats which who tell give show write explain describe about with and or so just now then also hey hi hello thanks thank".split())


def content_words(text):
    """ordered content words of an ask: lower case, punctuation and filler words removed ('What is the capital of France?' == 'capital of france' -> 'capital france')"""
    w = re.findall(r"[a-z0-9#+_\.]+", text.lower().replace("'s", ""))
    return [x.strip(".") for x in w if x.strip(".") and x.strip(".") not in STOP]


def ask_key(text):
    k = " ".join(content_words(text))
    return k or re.sub(r"\W+", " ", text.lower()).strip()          # an ask made only of filler words ("hi") keeps its own words


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def ctx_hash(prev_assistant):
    return hashlib.sha1(prev_assistant.strip().encode("utf-8")).hexdigest()[:12] if prev_assistant else ""


def ask_vector(model, text):
    """cheap meaning vector for an ask: mean of its token embeddings minus the average embedding, unit length"""
    E = model.W["token_embd.weight"]
    if not hasattr(model, "_emb_mean"):
        model._emb_mean = E.mean(0)
    ids = model.tok.encode(text)
    v = E[ids].mean(0) - model._emb_mean
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float16)


class Locker:
    def __init__(self, folder=DIR, max_records=400, max_mb=150):
        self.dir, self.max_records, self.max_mb = folder, max_records, max_mb
        os.makedirs(os.path.join(folder, "paths"), exist_ok=True)
        self.idx_file = os.path.join(folder, "index.json")
        self.lock = threading.RLock()
        self.rec, self.stats = {}, {"lock_hits": 0, "branch_runs": 0, "new_runs": 0, "tokens_replayed": 0, "tokens_drafted_ok": 0, "evicted": 0}
        self.tokens, self.vecs = {}, {}
        self._ngram = None
        if os.path.exists(self.idx_file):
            d = json.load(open(self.idx_file, encoding="utf-8"))
            self.rec, self.stats = d["records"], {**self.stats, **d.get("stats", {})}
        for rid in list(self.rec):
            try:
                z = np.load(self._npz(rid))
                self.tokens[rid], self.vecs[rid] = z["tokens"], (z["vec"] if "vec" in z.files else None)
            except Exception:
                self.rec.pop(rid)
        self.code = None                    # code function library (tokens), loaded lazily
        self.code_tokens = {}

    # ---------------------------------------------------------------- files
    def _npz(self, rid):
        return os.path.join(self.dir, "paths", rid + ".npz")

    def save_index(self):
        with self.lock:
            tmp = self.idx_file + ".tmp"
            json.dump({"records": self.rec, "stats": self.stats}, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            os.replace(tmp, self.idx_file)

    def _write_npz(self, rid, **arrs):
        old = {}
        if os.path.exists(self._npz(rid)):
            z = np.load(self._npz(rid)); old = {k: z[k] for k in z.files}
        old.update(arrs)
        np.savez_compressed(self._npz(rid), **old)

    # ---------------------------------------------------------------- lookup
    def find_exact(self, key, ch):
        best = None
        for rid, r in self.rec.items():
            if r["key"] == key and r["ctx"] == ch and r["status"] == "locked":
                if best is None or r["last_used"] > self.rec[best]["last_used"]:
                    best = rid
        return best

    def near(self, key, vec, ch, topn=3, min_sim=0.40):
        """locked or candidate paths for a similar (not identical) ask, best first: [(sim, id)]"""
        words, out = key.split(), []
        for rid, r in self.rec.items():
            if r["key"] == key and r["ctx"] == ch and r["status"] == "locked":
                continue
            c = 0.0
            v = self.vecs.get(rid)
            if vec is not None and v is not None:
                c = float(np.dot(vec.astype(np.float32), v.astype(np.float32)))
            sim = 0.5 * jaccard(words, r["key"].split()) + 0.5 * max(c, 0.0)
            if r["ctx"] != ch:
                sim *= 0.6
            if sim >= min_sim:
                out.append((sim, rid))
        out.sort(reverse=True)
        return out[:topn]

    # ---------------------------------------------------------------- write
    def add(self, ask, ch, tokens, text, vec, parent=None, branches=None, stats=None, prompt_ids=None):
        """store a freshly generated path as a candidate (replaces an older candidate of the same ask)"""
        with self.lock:
            key = ask_key(ask)
            for rid, r in list(self.rec.items()):
                if r["key"] == key and r["ctx"] == ch and r["status"] == "candidate":
                    self.delete(rid, save=False)
            rid = time.strftime("%Y%m%d%H%M%S") + "%03d" % (int(time.time() * 1000) % 1000)
            now = time.time()
            self.rec[rid] = {"id": rid, "ask": ask.strip()[:300], "key": key, "ctx": ch, "status": "candidate", "kind": "code" if "```" in text or "{" in text and ";" in text else "text",
                             "created": now, "last_used": now, "hits": 0, "locked_at": None, "n_tokens": len(tokens), "reply": text, "parent": parent,
                             "branches": branches or [], "fingerprint": False, "stats": stats or {}}
            self.tokens[rid], self.vecs[rid] = np.asarray(tokens, np.int32), vec
            self._write_npz(rid, tokens=np.asarray(tokens, np.int32), vec=vec if vec is not None else np.zeros(1, np.float16), prompt=np.asarray(prompt_ids if prompt_ids is not None else [], np.int32))
            self._ngram = None
            self.prune(keep=rid, save=False)
            self.save_index()
            return rid

    def approve(self, rid):
        with self.lock:
            if rid not in self.rec: return False
            r = self.rec[rid]
            for o, ro in list(self.rec.items()):                         # one locked path per ask: the newest approval wins
                if o != rid and ro["key"] == r["key"] and ro["ctx"] == r["ctx"] and ro["status"] == "locked":
                    ro["status"] = "candidate"
            r["status"], r["locked_at"] = "locked", time.time()
            self.save_index()
            return True

    def unlock(self, rid):
        with self.lock:
            if rid in self.rec:
                self.rec[rid]["status"] = "candidate"; self.save_index()

    def delete(self, rid, save=True):
        with self.lock:
            self.rec.pop(rid, None); self.tokens.pop(rid, None); self.vecs.pop(rid, None); self._ngram = None
            try: os.remove(self._npz(rid))
            except OSError: pass
            if save: self.save_index()

    def touch(self, rid, replayed=0):
        with self.lock:
            r = self.rec[rid]; r["hits"] += 1; r["last_used"] = time.time()
            self.stats["lock_hits"] += 1; self.stats["tokens_replayed"] += replayed
            self.save_index()

    def score(self, r, now=None):
        """keep-score: used often and recently = high.  Locked paths get a bonus, never-used candidates are the first to go."""
        now = now or time.time()
        age_days = max((now - r["last_used"]) / 86400.0, 0.0)
        return ((r["hits"] * 2.0) + (3.0 if r["status"] == "locked" else 0.0) + 1.0) / (1.0 + age_days)

    def disk_mb(self):
        return sum(os.path.getsize(os.path.join(self.dir, "paths", f)) for f in os.listdir(os.path.join(self.dir, "paths"))) / 2 ** 20

    def prune(self, keep=None, save=True):
        """erase the least-used, oldest paths until the index is within its limits; returns the erased ids"""
        gone = []
        with self.lock:
            while len(self.rec) > self.max_records or self.disk_mb() > self.max_mb:
                cand = [(self.score(r), rid) for rid, r in self.rec.items() if rid != keep]
                if not cand: break
                _, rid = min(cand)
                self.delete(rid, save=False); gone.append(rid)
            self.stats["evicted"] += len(gone)
            if save and gone: self.save_index()
        return gone

    # ---------------------------------------------------------------- drafting corpus
    def load_code(self):
        if self.code is not None: return
        self.code = {}
        f = os.path.join(self.dir, "code_index.npz"); j = os.path.join(self.dir, "code_index.json")
        if os.path.exists(f) and os.path.exists(j):
            z = np.load(f); off = z["offsets"]; tok = z["tokens"]
            for i, fn in enumerate(json.load(open(j, encoding="utf-8"))["functions"]):
                self.code["fn:%s@%s" % (fn["name"], fn["program"])] = tok[off[i]:off[i + 1]]

    def corpus(self, primary):
        """ordered {id: tokens}: primary paths first, then every other stored path, then the code function library"""
        self.load_code()
        seqs = {rid: self.tokens[rid] for rid in primary if rid in self.tokens}
        for rid, t in self.tokens.items():
            seqs.setdefault(rid, t)
        for k, t in self.code.items():
            seqs.setdefault(k, t)
        return seqs

    # ---------------------------------------------------------------- fingerprint ("which neurons this path fires")
    def fingerprint(self, model, prompt_ids, rid, progress=None):
        """one teacher-forced pass over prompt+reply: for every layer keep the FP_K neurons triggered on most reply tokens
        (triggered = |activation| >= 10% of the loudest neuron of that layer at that token).  Branch tokens get their own fingerprint."""
        r = self.rec[rid]; toks = list(map(int, self.tokens[rid]))
        s0 = len(prompt_ids); ids = list(prompt_ids) + toks
        NL, F = model.n_layers, model.W["blk.0.ffn_up.weight"].shape[0]
        bmask = np.zeros(len(toks), bool)
        for b in r.get("branches", []):
            bmask[b["start"]:b["end"]] = True
        all_c = np.zeros((NL, F), np.uint16); br_c = np.zeros((NL, F), np.uint16)
        st = {"c0": 0}

        def hook(l, pt, a):
            if pt != "act": return
            ab = np.abs(a); mk = ab >= 0.1 * ab.max(1, keepdims=True)
            g = st["c0"] + np.arange(len(a)) - s0
            ok = g >= 0
            if not ok.any(): return
            all_c[l] += mk[ok].sum(0).astype(np.uint16)
            bm = ok & (bmask[np.clip(g, 0, len(bmask) - 1)] if len(bmask) else False)
            if bm.any(): br_c[l] += mk[bm].sum(0).astype(np.uint16)
        cache = model.new_cache(len(ids) + 8)
        for c0 in range(0, len(ids), 256):
            st["c0"] = c0
            model.forward(ids[c0:c0 + 256], cache, hook)
            if progress: progress(min(c0 + 256, len(ids)), len(ids))

        def top(c):
            ii = np.argsort(-c.astype(np.int32), axis=1)[:, :FP_K]
            return ii.astype(np.uint16), np.take_along_axis(c, ii, 1).astype(np.uint16)
        a_i, a_c = top(all_c)
        arrs = {"fp_ids": a_i, "fp_cnt": a_c}
        if bmask.any():
            b_i, b_c = top(br_c); arrs.update(br_ids=b_i, br_cnt=b_c)
        with self.lock:
            self._write_npz(rid, **arrs)
            r["fingerprint"] = True
            r["neurons_triggered_ever"] = int((all_c > 0).sum())
            self.save_index()

    def prompt(self, rid):
        return list(map(int, np.load(self._npz(rid))["prompt"]))

    def fp_overlap(self, a, b):
        """share of path a's top neurons that path b also uses (same layer)"""
        za, zb = np.load(self._npz(a)), np.load(self._npz(b))
        if "fp_ids" not in za.files or "fp_ids" not in zb.files: return None
        return float(np.mean([len(set(za["fp_ids"][l]) & set(zb["fp_ids"][l])) / FP_K for l in range(za["fp_ids"].shape[0])]))


# ====================================================================== following a locked path while the model verifies it
class PathFollower:
    """Proposes draft tokens by walking along stored paths.  Falls off at a divergence (branch), re-finds the path with an n-gram match (rejoin)."""
    N = 3

    def __init__(self, seqs, K=12):
        self.seqs, self.K, self.k = seqs, K, K
        self.hist, self.track = [], None            # track = [sid, ptr]
        self.lastptr = {}
        self.index = {}
        for sid, s in seqs.items():
            for i in range(self.N, len(s)):
                self.index.setdefault((int(s[i - 3]), int(s[i - 2]), int(s[i - 1])), []).append((sid, i))
        first = next(iter(seqs), None)
        if first is not None and not first.startswith("fn:"):
            self.track = [first, 0]

    def advance(self, tok):
        """a token was emitted; True if it was the next token of the followed path"""
        self.hist.append(int(tok)); on = False
        if self.track:
            sid, p = self.track; s = self.seqs[sid]
            if p < len(s) and int(s[p]) == int(tok):
                self.track[1] += 1; self.lastptr[sid] = p + 1; on = True
            else:
                self.track = None
        return on

    def propose(self, room):
        k = max(0, min(self.k, room))
        if k == 0: return []
        if not self.track and len(self.hist) >= self.N:
            cand = self.index.get(tuple(self.hist[-self.N:]))
            if cand:
                best = cand[0]
                for sid, pos in cand:                       # prefer the earliest stored path, at/after where we left it
                    if pos >= self.lastptr.get(sid, 0):
                        best = (sid, pos); break
                self.track = [best[0], best[1]]
        if not self.track: return []
        sid, p = self.track; s = self.seqs[sid]
        d = [int(x) for x in s[p:p + k]]
        if not d: self.track = None
        return d

    def report(self, drafted, accepted):
        if drafted == 0: return
        self.k = min(self.K, self.k + 4) if accepted >= drafted * 0.75 else max(3, self.k // 2) if accepted < drafted * 0.4 else self.k


def follow_generate(m, logits, cache, follower, max_new, sample_kw, should_stop=lambda: False):
    """Generator over (token, on_path).  Same sampling as normal decoding; drafts only decide how many tokens one forward pass checks.
    Ends at the end-of-turn token, max_new, a stop request or a full cache.  `follower.passes / .fed / .accepted` hold the counters."""
    end, eos = m.tok.id_of["<|im_end|>"], m.tok.eos
    follower.passes = follower.fed = follower.accepted = 0
    n, t = 0, m.sample(logits, **sample_kw)
    cap = cache.k.shape[2]
    while True:
        if t == end or t == eos:
            m.stop_reason = "end-of-turn token"; return
        if n >= max_new or should_stop() or cache.len >= cap - 2:
            m.stop_reason = "max new tokens reached"; return
        on = follower.advance(t)
        yield t, on
        n += 1
        if n >= max_new: m.stop_reason = "max new tokens reached"; return
        room = min(cap - cache.len - 2, max_new - n)
        drafts = follower.propose(room)
        feed = [t] + drafts
        p0 = cache.len
        L = m.forward(feed, cache, all_logits=True)
        follower.passes += 1; follower.fed += len(feed)
        acc, nxt = 0, None
        for j in range(len(feed)):
            s = m.sample(L[j], **sample_kw)
            if j < len(drafts) and s == drafts[j]:
                follower.advance(s); acc += 1; follower.accepted += 1
                yield s, True
                n += 1
                if n >= max_new: break
            else:
                nxt = s; break
        cache.len = p0 + 1 + acc
        follower.report(len(drafts), acc)
        if nxt is None:
            m.stop_reason = "max new tokens reached"; return
        t = nxt
