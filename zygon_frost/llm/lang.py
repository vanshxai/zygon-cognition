"""Frost Lang (Frost 3.0): an engine that always has something to say.

 store      sentences (text + where it came from), a word index (word -> sentences), a symbol index (. ? ! : ( " ... -> sentences), everyday-English lessons, saved answers
 reading    split into sentences, add each to the indexes; nothing else
 replying   1 lesson match (greetings, small talk, grammar)  2 saved answer  3 association: the sentences that contain the rare words of what was typed, the shortest complete one wins
            4 nothing known about the typed word: ask where to read about it
 typing mistakes are mapped to the nearest known word by letter triplets
Persistence: append-only log, replayed on start. No matrices.
"""
import json, math, os, random, re, threading, time
from collections import defaultdict

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SYM_ASK = {".": "a plain statement", "?": "a question", "!": "an exclamation", ":": "a colon", ";": "a semicolon", ",": "a comma", "-": "a dash", "(": "brackets", ")": "brackets", '"': "quotation marks", "'": "an apostrophe"}


FILLER = frozenset("write function program code example that for in i want how do use python c a an the to of me show please my give is it with using make".split())


def stem(w):
    """a light word-ending remover so adds / added / adding all read as add (applied to lessons and to what is typed against them)"""
    if len(w) > 3:
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3: w = w[:-len(suf)]; break
        if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou": w = w[:-1]
    return w


def lwords(text): return [stem(w) for w in WORD.findall(text.lower().replace("'", ""))]


def tris(text):
    s = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def edit_distance(a, b, limit):
    """Damerau-Levenshtein distance, giving up once it is certainly above `limit`"""
    if abs(len(a) - len(b)) > limit: return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1; cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]: cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit: return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class Lang:
    def __init__(self, folder):
        self.dir = folder; os.makedirs(folder, exist_ok=True); self.lock = threading.RLock(); self.log = os.path.join(folder, "learned.jsonl")
        self.ctx, self.last_end, self.said, self.ctx_on = {}, None, set(), True
        self.ctxp = {"gain": 0.3, "pron_gain": 0.6, "kappa": 1.2, "rho": 5.0}      # the chat so far: word -> weight (fades every turn), where the last reply stopped, the sentences already said
        self.pend = set()                                               # sentences that ended a paragraph in the book
        self.sents, self.src, self.dead = [], [], set()                 # sentence text, source index, deleted ids
        self.sources, self.src_of = [], {}                              # source names
        self.post, self.sym = defaultdict(list), defaultdict(list)      # word -> sentence ids, symbol -> sentence ids
        self.groups, self.next_gid, self.wd, self.wdidx, self.gidx = {}, 1, [], defaultdict(set), defaultdict(set)   # lessons: wording ids per word, lesson ids per word
        self.saved, self.next_sid = {}, 1
        self._vt, self._vt_n = defaultdict(set), 0                      # vocabulary trigram index (typing mistakes)
        self.agg, self.t_start = {}, time.time()
        self.mp = {"on": False, "tau": 0.55}; self._mn = None; self._mstem = None; self._mcand = {}       # meaning angles: a word never seen is read as the known word whose 64 angles line up best (if they line up at least tau)
        self.dirp = {"mode": "dir", "tau": 1.0, "theta": 0.25, "maxw": 90, "back": False, "beta": 0.4, "para": True}
        if os.path.exists(self.log):
            for ln in open(self.log, encoding="utf-8"):
                if ln.strip(): self._apply(json.loads(ln), replay=True)

    # ------------------------------------------------------------ metrics
    def _agg(self, key, val):
        a = self.agg.setdefault(key, [0, 0.0, float("inf"), 0.0]); a[0] += 1; a[1] += val; a[2] = min(a[2], val); a[3] = max(a[3], val)

    def reply_stats(self):
        a = self.agg.get("reply"); c = self.agg.get("reply_words"); nf = self.agg.get("reply_ask")
        return {"n": a[0] if a else 0, "not_found": nf[0] if nf else 0, "mean_ms": (a[1] / a[0]) if a else None, "fastest_ms": a[2] if a else None, "slowest_ms": a[3] if a else None, "units_per_s": (c[1] / (a[1] / 1000)) if a and c and a[1] else None, "unit": "words"}

    # ------------------------------------------------------------ learning
    def _source(self, name):
        if name not in self.src_of: self.src_of[name] = len(self.sources); self.sources.append(name)
        return self.src_of[name]

    def _add(self, text, source):
        sid = len(self.sents); self.sents.append(text); self.src.append(self._source(source))
        for w in set(WORD.findall(text.lower())): self.post[w].append(sid)
        for c in set(ch for ch in text if not ch.isalnum() and not ch.isspace() and ch != "'"): self.sym[c].append(sid)
        return sid

    def _apply(self, ev, replay=False):
        t = ev["t"]
        if t == "sent": self._add(ev["text"], ev["source"])
        elif t == "doc":
            ends = set(ev.get("ends", ()))
            for k, s in enumerate(ev["sentences"]):
                sid = self._add(s, ev["source"])
                if k in ends: self.pend.add(sid)
        elif t == "lesson":
            gid = self.next_gid; self.next_gid += 1; g = {"gid": gid, "cat": ev["cat"], "utts": ev["utts"], "replies": ev["replies"]}; self.groups[gid] = g
            for u in ev["utts"]:
                wid = len(self.wd); ws = frozenset(lwords(u)); self.wd.append((gid, u, ws))
                for w in ws: self.wdidx[w].add(wid); self.gidx[w].add(gid)
        elif t == "saved":
            sid = self.next_sid; self.next_sid += 1; self.saved[sid] = {"id": sid, "q": ev["q"], "text": ev["text"], "source": ev["source"], "when": ev.get("time", 0), "sent_from": len(self.sents)}
            n0 = len(self.sents)
            for s in ev.get("sentences", []): self._add(s, "web: " + ev["source"])
            self.saved[sid]["sent_to"] = len(self.sents)
        elif t == "del":
            kind, key = ev["kind"], ev["key"]
            if kind == "doc": self.dead |= {i for i, s in enumerate(self.src) if self.sources[s] == key}
            elif kind == "saved":
                sv = self.saved.pop(key, None)
                if sv: self.dead |= set(range(sv["sent_from"], sv["sent_to"]))
            elif kind == "lesson":
                g = self.groups.pop(key, None)
                if g:
                    gone = {i for i, (gid, _, _) in enumerate(self.wd) if gid == key}
                    for w, ids in self.wdidx.items(): ids -= gone
                    for ids in self.gidx.values(): ids.discard(key)
        if not replay:
            with self.lock: open(self.log, "a", encoding="utf-8").write(json.dumps(ev, ensure_ascii=False) + "\n")

    def add_sentence(self, text, source):
        t0 = time.perf_counter(); self._apply({"t": "sent", "text": text, "source": source}); self._agg("learn", (time.perf_counter() - t0) * 1000)

    def learn_lesson(self, cat, utts, replies):
        t0 = time.perf_counter(); self._apply({"t": "lesson", "cat": cat, "utts": list(utts), "replies": list(replies)}); self._agg("learn", (time.perf_counter() - t0) * 1000)

    def learn_text(self, text, source):
        sents, ends = [], []; t0 = time.perf_counter()
        for par in re.split(r"\n\s*\n", text):                  # keep where each paragraph ends
            ss = split_sentences(par)
            if ss: sents += ss; ends.append(len(sents) - 1)
        if sents: self._apply({"t": "doc", "sentences": sents, "source": source, "ends": ends})
        self._agg("learn", (time.perf_counter() - t0) * 1000); return sum(len(s.split()) for s in sents)

    def remember(self, q, text, source):
        sents = split_sentences(text); self._apply({"t": "saved", "q": q, "text": text, "source": source, "sentences": sents, "time": time.time()}); return f"s{self.next_sid - 1}"

    # ------------------------------------------------------------ helpers
    def _idf(self, w): return math.log((len(self.sents) + 1) / (len(self.post.get(w, ())) + 0.5))

    def _meaning(self):
        if self._mn is None:
            try:
                import meaning
                self._mn = meaning.Meaning(); self._mstem = defaultdict(list)
                for x in self._mn.words: self._mstem[stem(x)].append(x)
            except Exception: self._mn = False
        return self._mn or None

    def _mean_alt(self, raw, vocab, floor=None):
        """the vocabulary word (stemmed) whose meaning angles line up best with the typed word, or None"""
        M = self._meaning()
        if not M or not M.has(raw): return None
        key = (id(vocab), len(vocab))
        if key not in self._mcand: self._mcand = {key: [x for st in vocab if st in self._mstem for x in self._mstem[st]]}
        near = M.nearest(raw, self._mcand[key], 1, self.mp["tau"] if floor is None else floor)
        return stem(near[0][0]) if near else None

    def _closest(self, w):
        """the known word closest to a mistyped one: letter triplets find the candidates, edit distance (one slip for short words, two for long) picks the winner"""
        if len(w) < 3: return None
        if self._vt_n != len(self.post):
            self._vt = defaultdict(set); self._vt_n = len(self.post)
            for v in self.post:
                for g in tris(v): self._vt[g].add(v)
        cand = defaultdict(int)
        for g in tris(w):
            for v in self._vt.get(g, ()): cand[v] += 1
        limit = 1 if len(w) < 8 else 2; best, bd = None, 99
        for v, k in cand.items():
            if abs(len(v) - len(w)) > limit or k < 1: continue
            if len(w) <= 5 and v[0] != w[0]: continue                 # a short word keeps its first letter when mistyped: "hvac" is not "vac"
            d = edit_distance(w, v, limit)
            if d <= limit and (d < bd or (d == bd and best and len(self.post[v]) > len(self.post[best]))): best, bd = v, d
        return best

    # ------------------------------------------------------------ replying
    def lesson(self, q, prefix=None):
        qs = frozenset(lwords(q))
        if prefix: qs = frozenset(w for w in qs if w not in FILLER) or qs       # once the language is known, words like "write", "program", "python" carry no meaning
        if not qs or not self.wd: return None
        N = max(1, len(self.groups)); df = lambda w: len(self.gidx.get(w, ())); idf = lambda w: math.log((N + 1) / (df(w) + 0.5)); unknown = [w for w in qs if w not in self.wdidx]
        if unknown and self.mp["on"]:
            raws = WORD.findall(q.lower().replace("'", ""))
            for w in list(unknown):
                raw = next((r for r in raws if stem(r) == w), w); alt = self._mean_alt(raw, self.wdidx)
                if alt: qs = frozenset((qs - {w}) | {alt}); unknown.remove(w)
        if len(unknown) > 1 or (unknown and len(qs) < 2): return None
        cand = set()
        for w in qs: cand |= self.wdidx.get(w, set())
        known = [w for w in qs if w in self.wdidx]; kt = (sum(idf(w) for w in known) + len(unknown) * math.log(2 * N + 2)) or 1.0; sc = []       # a word never seen in any lesson counts as the rarest word
        for wid in cand:
            gid, u, ws = self.wd[wid]
            if gid not in self.groups or (prefix and not self.groups[gid]["cat"].startswith(prefix)): continue
            if prefix: ws = (ws - FILLER) or ws
            tot = sum(idf(w) for w in ws) or 1.0; sh = sum(idf(w) for w in ws & qs); cov, prec = sh / tot, sh / kt
            if cov >= 0.75 and prec >= (0.5 if unknown and len(qs) >= 4 else 0.35) and sh >= 1.2: sc.append((cov * (0.5 + 0.5 * prec), gid, u))      # at least one real word in common, not just "the"
        if not sc: return None
        sc.sort(key=lambda x: -x[0]); top = sc[0][0]; s, gid, u = random.choice([x for x in sc if x[0] >= top - 0.02]); g = self.groups[gid]
        return {"text": random.choice(g["replies"]), "cat": g["cat"], "matched": u}

    def saved_answer(self, q):
        qs = set(WORD.findall(q.lower()))
        if not qs: return None
        best, bs = None, 0.0
        for sv in self.saved.values():
            ws = set(WORD.findall(sv["q"].lower())); sh = qs & ws
            if not sh: continue
            sc = sum(self._idf(w) for w in sh) / (sum(self._idf(w) for w in qs | ws) or 1.0)
            if sc > bs: best, bs = sv, sc
        return best if best and bs >= 0.6 else None

    def associate(self, q):
        """the sentence most associated with what was typed; for one word, the shortest complete sentence that has it"""
        words = WORD.findall(q.lower()); known, fixed = [], []
        for w in words:
            if self.post.get(w): known.append(w)
            else:
                alt = self._closest(w)
                if not alt and self.mp["on"]: alt = self._mean_alt(w, self.post)
                if alt: known.append(alt); fixed.append((w, alt))
        fixed_from = {t for t, _ in fixed}; unknown = [w for w in words if w not in known and w not in fixed_from]
        if not known:
            syms = [c for c in q if not c.isalnum() and not c.isspace() and c != "'"]
            for c in syms:
                ids = [i for i in self.sym.get(c, ()) if i not in self.dead and 4 <= len(self.sents[i].split()) <= 30 and not re.search(r"\d{3,}|@|www|\w \w \w", self.sents[i])]
                if ids:
                    want = (lambda t: t.rstrip().endswith(c)) if c in ".?!" else (lambda t: True)
                    pick = sorted((i for i in ids if want(self.sents[i])), key=lambda i: len(self.sents[i])) or sorted(ids, key=lambda i: len(self.sents[i]))
                    return {"text": self.sents[pick[0]], "ids": [pick[0]], "cover": 1.0, "topic": q.strip(), "ask": False, "kind_of": "symbol"}
            return {"text": None, "topic": " ".join(unknown) or q.strip(), "ask": True, "kind_of": "unknown"}
        content = [w for w in set(known) if len(self.post[w]) <= 0.08 * len(self.sents)]              # words that are in more than 8% of all sentences (what, is, the ...) never decide the answer when a rarer word is there
        typed = [w for w in content if w not in self.FOLLOW]                      # words that say what the question is about (not "tell", "more", "else" ...)
        pron = any(w in self.PRON for w in words) or words[:1] in (["and"], ["also"], ["then"])
        if self.ctx_on and self.last_end is not None and not typed:               # "tell me more", "and then?": go on from where the last reply stopped
            a = self.last_end + 1
            if a < len(self.sents) and self.src[a] == self.src[self.last_end] and a not in self.dead and 3 <= len(self.sents[a].split()) <= 60:
                ckw = sorted(self.ctx, key=lambda w: -self.ctx[w])[:4]; cidf = {w: self._idf(w) * self.ctx[w] for w in ckw}
                ids = self.direct(a, ckw, cidf, sum(cidf.values()) or 1.0, 3); self._turn([], ids, a)
                return {"anchor": a, "text": " ".join(self.sents[i] for i in ids).replace("\ufffd", ""), "ids": ids, "cover": 1.0, "topic": " ".join(ckw), "ask": False, "kind_of": "continue", "fixed": fixed}
        kw = sorted(content or set(known), key=lambda w: len(self.post[w]))[:4]; idf = {w: self._idf(w) for w in kw}
        follow = bool(self.ctx_on and self.ctx and (pron or len(typed) <= 1)); gain = self.ctxp["pron_gain"] if pron else self.ctxp["gain"]
        if follow:                                                                  # add the fading chat words, weaker than the typed ones: weight = idf * how recent
            for w in sorted(self.ctx, key=lambda w: -self.ctx[w] * self._idf(w)):
                if w not in idf and self.post.get(w) and self.ctx[w] * gain >= 0.15 and len(kw) < 7: kw.append(w); idf[w] = self._idf(w) * self.ctx[w] * gain
        tot = sum(idf.values()) or 1.0
        if unknown and tot < 0.8 * math.log(2 * len(self.sents) + 2) * len(unknown): return {"text": None, "topic": " ".join(unknown), "ask": True, "kind_of": "unknown"}     # the typed word is new and the rest is too common to say anything about it
        cands = set()
        for w in kw[:5]: cands.update(self.post[w][:6000])
        best, bsc, bcov = None, -9.0, 0.0
        for i in cands:
            if i in self.dead: continue
            t = self.sents[i]; sw = set(WORD.findall(t.lower())); n = len(t.split())
            if n < 3 or n > 60: continue
            if typed and follow and not any(w in sw for w in typed): continue      # the chat can pick between sentences, but the sentence must still have a word that was just typed
            cov = sum(idf[w] for w in kw if w in sw) / tot
            sc = cov - 0.012 * n + (0.05 if t[:1].isupper() and t.rstrip()[-1:] in ".?!\"" else 0) + (0.04 if kw[0] in WORD.findall(t.lower())[:3] else 0) - (0.5 if (follow and i in self.said) else 0) + (self.ctxp['kappa'] * math.exp(-abs(i - self.last_end) / self.ctxp['rho']) if (follow and self.last_end is not None and self.src[i] == self.src[self.last_end]) else 0)
            if sc > bsc: best, bsc, bcov = i, sc, cov
        if best is None: return {"text": None, "topic": " ".join(unknown) or q.strip(), "ask": True, "kind_of": "unknown"}
        ids = self.direct(best, kw, idf, tot, len(words)); self._turn(typed, ids, best)
        text = " ".join(self.sents[i] for i in ids).replace("�", "")
        low = bcov < 0.6 or bool(unknown)
        return {"anchor": best, "text": text, "ids": ids, "cover": round(bcov, 3), "topic": " ".join(unknown) or " ".join(known), "ask": low, "kind_of": "words", "fixed": fixed}

    # ------------------------------------------------------------ the chat as a thread
    FOLLOW = frozenset("tell more else another continue go on further then next also and about what say explain me please".split())
    PRON = frozenset("he she it they them his her its their this that these those there him".split())

    def _turn(self, typed, ids, anchor):
        """after a reply: every old chat word fades (x0.6), the typed words come in at 1.0, the reply's own rare words at 0.6"""
        lim = 0.08 * len(self.sents); self.ctx = {w: v * 0.6 for w, v in self.ctx.items() if v * 0.6 >= 0.1}
        for w in typed: self.ctx[w] = 1.0
        rare = sorted({w for i in ids for w in WORD.findall(self.sents[i].lower()) if self.post.get(w) and len(self.post[w]) <= lim and w not in self.FOLLOW and w not in self.PRON}, key=lambda w: len(self.post[w]))[:4]
        for w in rare: self.ctx[w] = max(self.ctx.get(w, 0), 0.6)
        if len(self.ctx) > 12: self.ctx = dict(sorted(self.ctx.items(), key=lambda kv: -kv[1])[:12])
        self.last_end = ids[-1]; self.said = set(ids)

    def reset_chat(self): self.ctx, self.last_end, self.said = {}, None, set()

    # ------------------------------------------------------------ directional gate
    ANAPHORA = frozenset("he she it they them his her its their this that these those such however thus therefore but and so also then".split())

    def direct(self, a, kw, idf, tot, nq):
        """how far to walk along the path of sentences from the best one: a multiplication gate, not a probability.
        g_j = exp(-(j-1)/tau) * (max(r_j, c_j) + beta)    r_j = idf share of the typed words in sentence a+j, c_j = idf share of the best sentence's own rare words in it
        keep sentence a+j while g_j >= theta and the reply stays under maxw words. A reply that starts with 'it', 'this', 'however' also takes the sentence before it."""
        P = self.dirp; mode = P["mode"]
        if mode == "single" or (mode == "pair" and nq < 3):
            return [a]
        if mode == "pair":
            n = a + 1
            return [a, n] if n < len(self.sents) and self.src[n] == self.src[a] and n not in self.dead and len(self.sents[n].split()) <= 30 else [a]
        N = len(self.sents); lim = 0.08 * N; sw0 = set(WORD.findall(self.sents[a].lower())); C = {w for w in sw0 if len(self.post.get(w, ())) <= lim}; idfC = {w: self._idf(w) for w in C}; tc = sum(idfC.values()) or 1.0
        ids = [a]; words = len(self.sents[a].split())
        if P["para"] and a in self.pend: return ids
        first = WORD.findall(self.sents[a].lower())[:1]
        if P["back"] and first and first[0] in self.ANAPHORA and a - 1 >= 0 and self.src[a - 1] == self.src[a] and a - 1 not in self.dead and words + len(self.sents[a - 1].split()) <= P["maxw"]:
            ids.insert(0, a - 1); words += len(self.sents[a - 1].split())
        j = 1
        while a + j < N and self.src[a + j] == self.src[a] and (a + j) not in self.dead:
            t = self.sents[a + j]; n = len(t.split())
            if words + n > P["maxw"]: break
            sw = set(WORD.findall(t.lower())); r = sum(idf[w] for w in kw if w in sw) / tot; c = sum(idfC[w] for w in C if w in sw) / tc
            if math.exp(-(j - 1) / P["tau"]) * (max(r, c) + P["beta"]) < P["theta"]: break
            ids.append(a + j); words += n; j += 1
            if P["para"] and (a + j - 1) in self.pend: break                   # a paragraph ended here: the thought is finished
        return ids

    def chat(self, q):
        t0 = time.perf_counter(); r = self.lesson(q)
        if r:
            ms = (time.perf_counter() - t0) * 1000; self._agg("reply", ms); self._agg("reply_words", len(r["text"].split())); return {"kind": "chat", "text": r["text"], "ms": ms, "cat": r["cat"], "matched": r["matched"]}
        sv = self.saved_answer(q)
        if sv:
            ms = (time.perf_counter() - t0) * 1000; self._agg("reply", ms); self._agg("reply_words", len(sv["text"].split())); return {"kind": "saved", "text": sv["text"], "from": sv["source"], "ms": ms}
        a = self.associate(q); ms = (time.perf_counter() - t0) * 1000
        if a["text"] is None: self._agg("reply_ask", ms); return {"kind": "ask", "ms": ms, "topic": a["topic"], "ask": True, "text": ""}
        self._agg("reply", ms); self._agg("reply_words", len(a["text"].split()))
        return {"kind": "lang", "text": a["text"], "ms": ms, "from": [self.sources[self.src[i]] for i in a["ids"]], "ask": a["ask"], "topic": a["topic"], "cover": a["cover"], "fixed": a.get("fixed")}

    # ------------------------------------------------------------ what it knows
    def memory(self):
        txt = sum(len(s) for i, s in enumerate(self.sents) if i not in self.dead); post = sum(len(v) for v in self.post.values()); sym = sum(len(v) for v in self.sym.values())
        parts = {"sentence text": txt, "word index": post * 8, "symbol index": sym * 8, "vocabulary": sum(len(w) + 24 for w in self.post), "lessons and saved answers": sum(len(u) + 40 for _, u, _ in self.wd) + sum(len(x["text"]) for x in self.saved.values())}
        return {"total": sum(parts.values()), "parts": parts, "sentences": len(self.sents) - len(self.dead), "words": len(self.post), "postings": post}

    def learned_items(self, query="", limit=300):
        out, docs = [], defaultdict(int)
        for i, s in enumerate(self.src):
            if i not in self.dead and not self.sources[s].startswith("web: ") and self.sources[s] != "reading": docs[self.sources[s]] += 1
        for name, n in docs.items(): out.append({"id": "d:" + name, "kind": "document", "title": name, "preview": f"{n:,} sentences", "source": "document", "when": 0})
        for sv in self.saved.values(): out.append({"id": f"s{sv['id']}", "kind": "saved answer", "title": sv["q"], "preview": sv["text"][:220], "source": sv["source"], "when": sv["when"]})
        for g in self.groups.values(): out.append({"id": f"g{g['gid']}", "kind": "everyday English", "title": g["utts"][0], "preview": g["replies"][0][:220], "source": g["cat"], "extra": f"{len(g['utts'])} ways of asking", "when": 0})
        q = query.strip().lower()
        if q: out = [o for o in out if q in (o["title"] + " " + o["preview"] + " " + o["source"] + " " + o["kind"]).lower()]
        order = {"saved answer": 0, "document": 1, "everyday English": 2}; out.sort(key=lambda o: (order.get(o["kind"], 3), -o["when"], o["title"]))
        return {"total": len(out), "items": out[:limit]}

    def delete_item(self, item_id):
        if item_id.startswith("d:"): self._apply({"t": "del", "kind": "doc", "key": item_id[2:]}); return 1
        if item_id.startswith("s") and item_id[1:].isdigit() and int(item_id[1:]) in self.saved: self._apply({"t": "del", "kind": "saved", "key": int(item_id[1:])}); return 1
        if item_id.startswith("g") and item_id[1:].isdigit() and int(item_id[1:]) in self.groups: self._apply({"t": "del", "kind": "lesson", "key": int(item_id[1:])}); return 1
        return 0


ABBR = ("mr", "mrs", "ms", "dr", "st", "prof", "vs", "etc", "no", "fig", "vol", "ch", "cf", "viz", "i.e", "e.g", "lat", "gen", "col", "capt", "lieut", "rev", "hon", "messrs", "sir", "mt", "co")


def split_sentences(text):
    """sentences of plain prose: joins broken lines, drops table-like and very short or very long pieces"""
    text = re.sub(r"\[[^\]]{0,200}\]", " ", text); text = re.sub(r"_([^_]{1,200})_", r"\1", text); text = re.sub(r"\s+", " ", text)
    out, buf = [], ""
    for piece in re.split(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z])", text):
        buf = (buf + " " + piece).strip() if buf else piece
        last = buf.rstrip(".!?\"'”’)").split(" ")[-1].lower().rstrip(".")
        if buf.endswith(".") and last in ABBR: continue
        n = len(buf.split()); letters = sum(c.isalpha() or c == " " for c in buf)
        if 4 <= n <= 70 and letters >= 0.8 * len(buf) and not re.search(r"@|www\.|https?:|\.com|\.org", buf): out.append(buf)
        buf = ""
    return out
