"""SI core: tokens with exponent identities, angle wheels, and hooks that attach every learned reply to the words that asked for it.

 * identity   token id -> e^x, x = ln(id+1)  (text, decodes back exactly)
 * wheel      one per context (last 0-3 tokens): a 360 degree circle, one sector per next token, sector width = how often it followed
 * hook       one per word of a question: sectors pointing at the replies that were learned for questions containing that word
 * learning   every answer that is not rejected is learned at once, the same way: wheels get sectors, the question's words get a hook to the reply
 * memory     grows with what is learned; a trigger notifies when it crosses the threshold you set
Nothing here uses a matrix.  Persistence = an append-only text log that is replayed on start.
"""
import json, math, os, random, re, threading, time
from collections import Counter
import path_locker as PL

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
DIR = os.path.join(ROOT, "model", "angle_ai")
END, BOS = 151645, -1
MB = 2 ** 20


def chat_words(text):
    """every word of an utterance, lower case, no filler removed (greetings are made of filler words)"""
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", "").replace(chr(0x2019), ""))


def wsim(a, b, df, n):
    """word overlap where rare words count more: sum of idf over shared words / sum of idf over all words (a word never seen counts as the rarest)"""
    a, b = set(a), set(b)
    if not a or not b: return 0.0
    idf = lambda w: math.log((n + 1) / (df(w) + 0.5))
    return sum(idf(w) for w in a & b) / sum(idf(w) for w in a | b)


def exp_text(t):
    return f"e^{math.log(t + 1):.6f}"


def exp_decode(s):
    return round(math.exp(float(s[2:]))) - 1


class AngleAI:
    K = 3

    def __init__(self, tok, folder=DIR):
        self.tok, self.dir, self.lock = tok, folder, threading.RLock()
        os.makedirs(folder, exist_ok=True)
        self.log_file, self.set_file, self.alert_file = (os.path.join(folder, n) for n in ("learned.jsonl", "settings.json", "alerts.log"))
        self.wheels, self.pw, self.pw_n, self.paths, self.hooks = {}, {}, {}, {}, {}
        self.window, self.series, self.alerts = [], [], []
        self.next_id = 1
        self.groups, self.dhooks, self.next_gid = {}, {}, 1             # lessons: a group = several wordings hooked to one or more replies
        self.cell_of = {}                                  # token id -> cell number (first-seen order) in the 10x10x10 box; the box is only for looking at it, never used to compute
        self.agg = {}                                      # metric aggregates: key -> [count, sum, min, max]
        self.t_start = time.time()
        self.met_file = os.path.join(folder, "metrics.jsonl")
        self.stats = {"memory": 0, "wheels": 0, "teacher": 0, "none": 0, "rejected": 0, "taught_tokens": 0}
        self.settings = {"memory_threshold_mb": 50.0, "confidence": 0.9, "replay_similarity": 0.5, "wheel_min_similarity": 0.34, "alert_active": False}
        if os.path.exists(self.set_file):
            self.settings.update(json.load(open(self.set_file, encoding="utf-8")))
        if os.path.exists(self.log_file):
            for ln in open(self.log_file, encoding="utf-8"):
                if ln.strip(): self._apply(json.loads(ln), replay=True)
        if os.path.exists(self.met_file):
            for ln in open(self.met_file, encoding="utf-8"):
                if ln.strip():
                    m = json.loads(ln); self._agg(m["k"], m["v"], persist=False)

    # ------------------------------------------------------------------ learning
    def _bump(self, ids, sign):
        hist = [BOS] * self.K
        for t in ids:
            if sign > 0 and t not in self.cell_of: self.cell_of[t] = len(self.cell_of)
            for k in range(self.K + 1):
                ctx = tuple(hist[len(hist) - k:]) if k else ()
                w = self.wheels.setdefault(ctx, {}); w[t] = w.get(t, 0) + sign
                if w[t] <= 0:
                    del w[t]
                    if not w: del self.wheels[ctx]
            hist.append(t)

    def _pull(self, words, ids, sign):
        for w in set(words):
            self.pw_n[w] = self.pw_n.get(w, 0) + sign; d = self.pw.setdefault(w, {})
            for t in set(ids):
                d[t] = d.get(t, 0) + sign
                if d[t] <= 0: del d[t]
            if self.pw_n[w] <= 0: self.pw_n.pop(w, None); self.pw.pop(w, None)

    def _apply(self, ev, replay=False):
        t = ev["t"]
        if t == "qa":
            pid = ev["id"]; self.next_id = max(self.next_id, pid + 1); words = PL.content_words(ev["prompt"])
            self.paths[pid] = {"id": pid, "prompt": ev["prompt"], "words": words, "reply_ids": ev["reply_ids"], "reply": self.dec(ev["reply_ids"]), "source": ev["source"], "created": ev["time"], "last_used": ev["time"], "hits": 0, "full": ev["full"], "pull": ev.get("pull", True)}
            for w in set(words): self.hooks.setdefault(w, {})[pid] = 1.0
            if ev["full"]:
                self._bump(ev["reply_ids"], +1)
                if ev.get("pull", True): self._pull(words, ev["reply_ids"], +1)          # word -> reply-token pulls are for answering questions; a document does not need them
        elif t == "lesson":
            gid = ev["gid"]; self.next_gid = max(self.next_gid, gid + 1)
            g = {"gid": gid, "cat": ev["cat"], "utts": ev["utts"], "words": [chat_words(u) for u in ev["utts"]], "replies": [{"text": r, "ids": ids} for r, ids in ev["replies"]], "source": ev.get("source", "lesson")}
            self.groups[gid] = g
            for ws in g["words"]:
                for w in set(ws): self.dhooks.setdefault(w, set()).add(gid)
            for r in g["replies"]: self._bump(r["ids"], +1)
        elif t == "lesson_del":
            g = self.groups.pop(ev["gid"], None)
            if g:
                for r in g["replies"]: self._bump(r["ids"], -1)
                for ws in g["words"]:
                    for w in set(ws):
                        if w in self.dhooks:
                            self.dhooks[w].discard(ev["gid"])
                            if not self.dhooks[w]: del self.dhooks[w]
        elif t == "hook":                                                   # a new wording attached to an already learned reply
            p = self.paths.get(ev["id"])
            if p:
                for w in set(PL.content_words(ev["prompt"])): self.hooks.setdefault(w, {})[ev["id"]] = 1.0
        elif t == "text":
            self._bump(ev["ids"], +1); self.stats["taught_tokens"] += len(ev["ids"])
        elif t == "reject":
            p = self.paths.pop(ev["id"], None)
            if p:
                for w in set(p["words"]):
                    self.hooks.get(w, {}).pop(p["id"], None)
                    if w in self.hooks and not self.hooks[w]: del self.hooks[w]
                if p["full"]:
                    self._bump(p["reply_ids"], -1)
                    if p.get("pull", True): self._pull(p["words"], p["reply_ids"], -1)
                self.stats["rejected"] += 1
        if not replay: self._note(ev)

    def _note(self, ev):
        with self.lock:
            open(self.log_file, "a", encoding="utf-8").write(json.dumps(ev, ensure_ascii=False) + "\n")
            nw = 0
            if ev["t"] == "qa":
                msg = f"learned a reply from {ev['source']}: hooked the words {PL.content_words(ev['prompt'])} to it" + ("" if ev["full"] else " (hook only, wheels unchanged)") + f"; {len(ev['reply_ids'])} tokens"
            elif ev["t"] == "lesson_del":
                msg = f"removed lesson #{ev['gid']}"
            elif ev["t"] == "lesson":
                msg = f"learned a lesson ({ev['cat']}): {len(ev['utts'])} wordings hooked to {len(ev['replies'])} repl{'y' if len(ev['replies']) == 1 else 'ies'}"
            elif ev["t"] == "hook":
                msg = f"hooked the new wording {PL.content_words(ev['prompt'])} to the reply learned for path #{ev['id']} (same reply, no wheel change)"
            elif ev["t"] == "text":
                msg = f"learned {len(ev['ids'])} tokens of plain text ({ev.get('label', '')})"
            else:
                msg = f"rejected path #{ev['id']}: its hook and its wheel counts were removed"
            self.window.append({"time": ev["time"], "text": msg, "sectors_total": nw}); self.window = self.window[-200:]
        now = time.time()
        if now - getattr(self, "_last_mem_check", 0.0) >= 0.5:                      # the memory trigger looks every half second, not after every sentence (counting all wheels each time is slow)
            self._last_mem_check = now; self.check_memory()

    def learn_qa(self, prompt, reply_ids, source, full=True, ended=True, pull=True):
        ids = [t for t in reply_ids if t != END] + ([END] if ended else [])
        t0 = time.perf_counter()
        with self.lock:
            pid = self.next_id; ev = {"t": "qa", "id": pid, "prompt": prompt, "reply_ids": ids, "source": source, "time": time.time(), "full": full, "pull": pull}
            self._apply(ev)
        self._agg("learn:qa", (time.perf_counter() - t0) * 1000); self._agg("learn_tokens", len(ids))
        return pid

    def learn_text(self, text, label=""):
        ids = self.tok.encode(text); t0 = time.perf_counter()
        for i in range(0, len(ids), 400):
            with self.lock: self._apply({"t": "text", "ids": ids[i:i + 400] + [END], "time": time.time(), "label": label})
        self._agg("learn:text", (time.perf_counter() - t0) * 1000); self._agg("learn_tokens", len(ids))
        return len(ids)

    def reject(self, pid):
        with self.lock:
            if pid in self.paths: self._apply({"t": "reject", "id": pid, "time": time.time()})

    # ------------------------------------------------------------------ answering
    def dec(self, ids):
        return self.tok.decode([t for t in ids if t != END])

    def _turn(self, hist, words):
        cand = {}
        for k in range(self.K, -1, -1):
            ctx = tuple(hist[len(hist) - k:]) if k else (); w = self.wheels.get(ctx)
            if w:
                tot = sum(w.values()); f = 0.4 ** (self.K - k)
                for u, c in w.items(): cand[u] = max(cand.get(u, 0.0), f * c / tot)
        if not cand: return cand
        known = [w for w in words if w in self.pw_n]
        for u in cand:
            pulls = sorted((self.pw[w].get(u, 0) / self.pw_n[w] for w in known), reverse=True)[:2]
            cand[u] *= 0.1 + (sum(pulls) / len(pulls) if pulls else 0.0)
        return cand

    def wheel_answer(self, prompt, max_new=60):
        words = PL.content_words(prompt); hist = [BOS] * self.K; out, confs = [], []
        for _ in range(max_new):
            cand = self._turn(hist, words)
            if not cand: break
            u = max(cand, key=cand.get); tot = sum(cand.values()); confs.append(cand[u] / tot if tot else 0.0)
            if u == END: break
            out.append(u); hist.append(u)
        known = sum(w in self.pw_n for w in words)
        return out, (sum(confs) / len(confs) if confs and known else 0.0)

    def find_hook(self, prompt):
        """follow the word hooks to the learned reply whose question has the same content words"""
        words = PL.content_words(prompt); N = max(1, len(self.paths)); sc = {}
        for w in set(words):
            h = self.hooks.get(w, {}); idf = math.log((N + 1) / (len(h) + 0.5))
            for pid, wt in h.items(): sc[pid] = sc.get(pid, 0.0) + idf * wt
        best, bs = None, 0.0
        for pid in sorted(sc, key=lambda p: -sc[p])[:5]:
            s = wsim(words, self.paths[pid]["words"], lambda w: len(self.hooks.get(w, {})), N)
            if s > bs: best, bs = pid, s
        return best, bs

    def ask(self, prompt):
        t0 = time.time(); pid, sim = self.find_hook(prompt)
        if pid is not None and sim >= 0.99:
            p = self.paths[pid]; p["hits"] += 1; p["last_used"] = time.time(); self.stats["memory"] += 1
            return self._res(prompt, p["reply"], "memory", 1.0, pid, t0, hooked=pid)
        words = PL.content_words(prompt); unknown = [w for w in words if w not in self.pw_n and w not in self.hooks]
        if unknown:                                                            # a word never seen before = something new: only the 3B model can answer
            return {"needs_teacher": True, "draft": "", "confidence": 0.0, "similar": (pid, sim), "reason": f"new word(s) the AI has never seen: {', '.join(unknown)}"}
        N = max(1, len(self.paths)); key = max(words, key=lambda w: math.log((N + 1) / (len(self.hooks.get(w, {})) + 0.5))) if words else None   # the rarest word, usually the name or the thing asked about
        covered = pid is not None and (key is None or key in self.paths[pid]["words"])
        if covered and sim >= self.settings["replay_similarity"]:                   # close wording of something learned: follow that path and hook this wording to it too
            p = self.paths[pid]; p["hits"] += 1; p["last_used"] = time.time(); self.stats["memory"] += 1
            th = time.perf_counter()
            with self.lock: self._apply({"t": "hook", "id": pid, "prompt": prompt, "time": time.time()})
            self._agg("learn:hook", (time.perf_counter() - th) * 1000)
            return self._res(prompt, p["reply"], "memory", sim, pid, t0, hooked=pid, similar_wording=True)
        out, conf = self.wheel_answer(prompt)
        if covered and out and conf >= self.settings["confidence"] and sim >= self.settings["wheel_min_similarity"] and len(out) >= 3:
            self.stats["wheels"] += 1
            new = self.learn_qa(prompt, out, "student wheels", full=False)             # the answer is hooked to its question; wheel counts are not inflated
            return self._res(prompt, self.dec(out), "wheels", conf, new, t0, similar=(pid, sim))
        return {"needs_teacher": True, "draft": self.dec(out), "confidence": conf, "similar": (pid, sim), "reason": "no learned question covers the main word of this question closely enough"}

    def _res(self, prompt, reply, source, conf, pid, t0, **kw):
        r = {"reply": reply, "source": source, "confidence": conf, "path_id": pid, "ms": round((time.time() - t0) * 1000, 3)}; r.update(kw)
        self._agg("ask:" + source, (time.time() - t0) * 1000)
        self.series.append(source); self.series = self.series[-80:]
        return r

    def learned_from_teacher(self, prompt, reply_text, t0, ended=True):
        ids = self.tok.encode(reply_text); pid = self.learn_qa(prompt, ids, "3B model", ended=ended); self.stats["teacher"] += 1
        return self._res(prompt, reply_text, "3B model", 1.0, pid, t0)

    # ------------------------------------------------------------------ search, completion, metrics, cube, store export
    def saved_answer(self, query, min_sim=0.5):
        """an answer that was approved from the web earlier: followed when its question has the same rare words"""
        words = PL.content_words(query)
        if not words or any(w not in self.hooks for w in words): return None
        N = max(1, len(self.paths)); df = lambda w: len(self.hooks.get(w, {})); key = max(words, key=lambda w: math.log((N + 1) / (df(w) + 0.5)))
        best, bs = None, 0.0
        for pid in set().union(*[set(self.hooks.get(w, {})) for w in words]):
            p = self.paths.get(pid)
            if not p or not p["source"].startswith("web:") or key not in p["words"]: continue
            sc = wsim(words, p["words"], df, N)
            if sc > bs: best, bs = pid, sc
        if best is None or bs < min_sim: return None
        p = self.paths[best]; p["hits"] += 1; p["last_used"] = time.time()
        return {"text": p["reply"], "source": p["source"][4:].strip(), "question": p["prompt"], "pid": best, "score": round(bs, 3)}

    def learned_items(self, query="", limit=300):
        """everything that has been taught, as a list a person can read and delete from"""
        out, docs = [], {}
        for pid, p in self.paths.items():
            src = p["source"]
            if src.startswith("web:"):
                out.append({"id": f"p{pid}", "kind": "saved answer", "title": p["prompt"], "preview": p["reply"][:220], "source": src[4:].strip(), "when": p["created"]})
            elif not src.startswith("student"):
                name = "bitcoin.pdf" if src == "pdf" else src.split(" p.")[0]; d = docs.setdefault(name, {"n": 0, "pages": set(), "when": p["created"]}); d["n"] += 1
                if " p." in src: d["pages"].add(src.split(" p.")[1])
        for name, d in docs.items():
            out.append({"id": "d:" + name, "kind": "document", "title": name, "preview": f"{d['n']} sentences" + (f" from {len(d['pages'])} pages" if d["pages"] else ""), "source": "document", "when": d["when"]})
        for gid, g in self.groups.items():
            out.append({"id": f"g{gid}", "kind": "everyday English", "title": g["utts"][0], "preview": g["replies"][0]["text"][:220], "source": g["cat"], "extra": f"{len(g['utts'])} ways of asking", "when": 0})
        q = query.strip().lower()
        if q: out = [o for o in out if q in (o["title"] + " " + o["preview"] + " " + o["source"] + " " + o["kind"]).lower()]
        kind_order = {"saved answer": 0, "document": 1, "everyday English": 2}
        out.sort(key=lambda o: (kind_order.get(o["kind"], 3), -o["when"], o["title"]))
        return {"total": len(out), "items": out[:limit]}

    def delete_item(self, item_id):
        if item_id.startswith("p") and item_id[1:].isdigit(): self.reject(int(item_id[1:])); return 1
        if item_id.startswith("d:"):
            name = item_id[2:]; pids = [pid for pid, p in self.paths.items() if not p["source"].startswith(("web:", "student")) and ("bitcoin.pdf" if p["source"] == "pdf" else p["source"].split(" p.")[0]) == name]
            for pid in pids: self.reject(pid)
            return len(pids)
        if item_id.startswith("g") and item_id[1:].isdigit():
            gid = int(item_id[1:])
            if gid in self.groups:
                with self.lock: self._apply({"t": "lesson_del", "gid": gid, "time": time.time()})
                return 1
        return 0

    def learn_lesson(self, cat, utts, replies, source="lesson"):
        """several ways of saying something hooked to the reply (or replies) to give; the replies' words also go into the wheels"""
        t0 = time.perf_counter(); reps = [[r, self.tok.encode(r) + [END]] for r in replies]
        with self.lock:
            gid = self.next_gid; self._apply({"t": "lesson", "gid": gid, "cat": cat, "utts": list(utts), "replies": reps, "source": source, "time": time.time()})
        self._agg("learn:lesson", (time.perf_counter() - t0) * 1000); self._agg("learn_tokens", sum(len(r[1]) for r in reps))
        return gid

    def converse(self, text, min_cover=0.75, min_prec=0.45):
        """answer from learned conversation lines: the rare words of a learned wording must all be in what was said (one unknown extra word is allowed)"""
        t0 = time.perf_counter(); words = chat_words(text)
        if not words or not self.groups: return None
        N = max(1, len(self.groups)); df = lambda w: len(self.dhooks.get(w, ())); idf = lambda w: math.log((N + 1) / (df(w) + 0.5))
        qs = set(words); unknown = [w for w in qs if w not in self.dhooks]
        if len(unknown) > 1 or (len(unknown) == 1 and len(qs) < 2): return None
        cand = set()
        for w in qs: cand |= self.dhooks.get(w, set())
        best, top = None, 0.0; scored = []
        for gid in cand:
            g = self.groups[gid]
            for ws, u in zip(g["words"], g["utts"]):
                us = set(ws); tot = sum(idf(w) for w in us) or 1.0; cover = sum(idf(w) for w in us & qs) / tot
                known_q = [w for w in qs if w in self.dhooks]; prec = sum(idf(w) for w in us & qs) / (sum(idf(w) for w in known_q) or 1.0)
                sc = cover * (0.5 + 0.5 * prec)
                if cover >= min_cover and prec >= min_prec: scored.append((sc, gid, u))
        if not scored: return None
        scored.sort(key=lambda x: -x[0]); top = scored[0][0]; ties = [x for x in scored if x[0] >= top - 0.02]
        sc, gid, u = random.choice(ties); g = self.groups[gid]; rep = random.choice(g["replies"]); ms = (time.perf_counter() - t0) * 1000
        self._agg("reply", ms); self._agg("reply_tokens", len(rep["ids"]) - 1)
        return {"found": True, "kind": "dialogue", "text": rep["text"], "cat": g["cat"], "matched": u, "score": round(sc, 3), "tokens": len(rep["ids"]) - 1, "ms": ms, "tokens_per_s": (len(rep["ids"]) - 1) / (ms / 1000) if ms else None,
                "wordings": len(g["utts"]), "replies_known": len(g["replies"])}

    def search(self, query, top=5):
        """follow the word hooks to the learned sentences/replies that cover the most rare words of the query: [(coverage, path id)], best first"""
        words = set(PL.content_words(query)); N = max(1, len(self.paths)); cand = set()
        for w in words: cand |= set(self.hooks.get(w, {}))
        idf = lambda w: math.log((N + 1) / (len(self.hooks.get(w, {})) + 0.5)); total = sum(idf(w) for w in words) or 1.0
        res = []
        for pid in cand:
            if pid not in self.paths: continue
            pw = set(self.paths[pid]["words"]); cov = sum(idf(w) for w in words & pw) / total
            res.append((cov + 0.001 * wsim(words, pw, lambda w: len(self.hooks.get(w, {})), N), pid))      # coverage first, tighter sentences win ties
        res.sort(reverse=True)
        return res[:top]

    def complete(self, text, max_new=40):
        """continue a sentence by turning wheels only (no prompt words, no teacher); returns the new token ids"""
        hist = [BOS] * self.K + self.tok.encode(text); out = []
        for _ in range(max_new):
            cand = self._turn(hist, [])
            if not cand: break
            u = max(cand, key=cand.get)
            if u == END: break
            out.append(u); hist.append(u)
        return out

    def compose(self, query, max_new=90, max_sent=2, min_cover=0.34):
        """Reply with no teacher. 1) the hooks of the query's words pick the learned sentences that cover them; 2) the angle wheels join word to word, steered towards
        the words of those sentences; when a sentence ends, the stored link to the sentence that followed it carries on. Every token is labelled with the sentence it came from."""
        t0 = time.perf_counter(); hits = [(sc, pid) for sc, pid in self.search(query, 6) if sc >= min_cover]; t_search = (time.perf_counter() - t0) * 1000
        if not hits:
            self._agg("reply_none", (time.perf_counter() - t0) * 1000)
            return {"found": False, "ms": (time.perf_counter() - t0) * 1000, "search_ms": t_search, "query_words": PL.content_words(query), "known_words": [w for w in PL.content_words(query) if w in self.hooks]}
        steer, pool = {}, {}
        for rank, (sc, pid) in enumerate(hits):
            for t in self.paths[pid]["reply_ids"]: steer[t] = max(steer.get(t, 0.0), sc / (1 + rank))
            pool[pid] = self.paths[pid]["reply_ids"]
        hist = [BOS] * self.K; out, orders, shares, seen4, sentences, cur = [], [], [], set(), 0, hits[0][1]
        for _ in range(max_new):
            best, bsc, bord, tot = None, -1.0, 0, 0.0; cand = {}
            for k in range(self.K, -1, -1):
                if k == 0 and cand: break                                                           # the 0-token wheel holds every token: only used when nothing longer is known
                ctx = tuple(hist[len(hist) - k:]) if k else (); w = self.wheels.get(ctx)
                if w:
                    tw = sum(w.values()); f = 0.4 ** (self.K - k)
                    for u, c in w.items():
                        v = f * c / tw
                        if u not in cand or v > cand[u][0]: cand[u] = (v, k)
            for u, (v, k) in cand.items():
                sc2 = v * (0.05 + steer.get(u, 0.0)); tot += sc2
                if len(out) >= 3 and tuple(out[-3:] + [u]) in seen4: sc2 *= 0.01                   # do not go round in circles
                if sc2 > bsc: best, bsc, bord = u, sc2, k
            if best is None: break
            if best == END:
                sentences += 1
                nxt = self.paths.get(cur + 1) if cur + 1 in self.paths else None
                if sentences >= max_sent or nxt is None: break
                cur += 1; pool[cur] = nxt["reply_ids"]; steer = {t: 1.0 for t in nxt["reply_ids"]}; hist = [BOS] * self.K; out.append(-2)   # -2 marks the hop to the next sentence
                continue
            out.append(best); orders.append(bord); shares.append(bsc / tot if tot else 0.0); hist.append(best)
            if len(out) >= 4 and -2 not in out[-4:]: seen4.add(tuple(out[-4:]))
        toks = [t for t in out if t != -2]
        # label every token with the learned sentence it belongs to (a sentence whose 3-grams contain the last three tokens; stay with the current one when possible)
        grams = {pid: {tuple(ids[i:i + 3]) for i in range(len(ids) - 2)} for pid, ids in pool.items()}; labels, last, i, hops, hop = [], None, 0, [], False
        for idx, t in enumerate(out):
            if t == -2: last = None; hop = True; continue
            tri = tuple(x for x in out[max(0, idx - 2):idx + 1] if x != -2); lab = last if (last in grams and tri in grams[last]) else next((pid for pid in pool if tri in grams[pid]), last if last is not None else hits[0][1])
            labels.append(lab); hops.append(hop); hop = False; last = lab
        segs = []
        for t, lab, hp in zip(toks, labels, hops):
            if segs and segs[-1]["pid"] == lab and not hp: segs[-1]["ids"].append(t)
            else: segs.append({"pid": lab, "ids": [t], "hop": hp})
        for sg in segs:
            pth = self.paths.get(sg["pid"], {}); sg["text"] = self.dec(sg["ids"]); sg["source"] = pth.get("source", ""); sg["sentence"] = pth.get("prompt", "")[:90]; sg["tokens"] = len(sg["ids"]); del sg["ids"]
        for a_, b_ in zip(segs, segs[1:]):                                                         # a space where one sentence ends and the next begins
            if (b_.get("hop") or a_["text"].rstrip()[-1:] in ".!?") and not b_["text"].startswith((" ", "\n")): b_["text"] = " " + b_["text"]
        ms = (time.perf_counter() - t0) * 1000; self._agg("reply", ms); self._agg("reply_tokens", len(toks))
        trace = [{"piece": self.dec([t]), "identity": exp_text(t), "order": o, "share": round(sh, 3)} for t, o, sh in zip(toks, orders, shares)]
        return {"found": True, "text": "".join(sg["text"] for sg in segs), "segments": segs, "tokens": len(toks), "ms": ms, "search_ms": t_search, "ms_per_token": ms / max(1, len(toks)), "tokens_per_s": len(toks) / (ms / 1000) if ms else None,
                "joins": max(0, len(segs) - 1), "matched": [{"pid": pid, "cover": round(sc, 3), "source": self.paths[pid].get("source", ""), "text": self.paths[pid]["prompt"][:110]} for sc, pid in hits[:4]], "trace": trace,
                "wheel_orders": {str(k): orders.count(k) for k in range(self.K + 1) if orders.count(k)}}

    def _agg(self, key, val, persist=True):
        a = self.agg.setdefault(key, [0, 0.0, float("inf"), 0.0]); a[0] += 1; a[1] += val; a[2] = min(a[2], val); a[3] = max(a[3], val)
        if persist:
            with self.lock: open(self.met_file, "a", encoding="utf-8").write(json.dumps({"k": key, "v": round(val, 4), "t": round(time.time(), 1)}) + "\n")

    def disk_bytes(self):
        return sum(os.path.getsize(os.path.join(self.dir, f)) for f in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, f)))

    def metrics(self):
        ask, learn = {}, {}
        for k, a in self.agg.items():
            if k.startswith("ask:"): ask[k[4:]] = {"n": a[0], "mean_ms": a[1] / a[0], "min_ms": a[2], "max_ms": a[3]}
            elif k.startswith("learn:"): learn[k[6:]] = {"n": a[0], "mean_ms": a[1] / a[0], "total_ms": a[1]}
        toks = self.agg.get("learn_tokens", [0, 0.0])[1]; total_ms = sum(v["total_ms"] for v in learn.values())
        m = self.memory(); disk = self.disk_bytes(); t3 = ask.get("3B model")
        fast = [ask[k] for k in ("memory", "wheels") if k in ask]; saved = None
        if t3 and fast: saved = sum(f["n"] * (t3["mean_ms"] - f["mean_ms"]) for f in fast)
        return {"ask": ask, "learn": learn, "tokens_learned": toks, "learn_seconds": total_ms / 1000, "learn_tokens_per_s": (toks / (total_ms / 1000)) if total_ms > 0 else None,
                "memory_bytes": m["total"], "disk_bytes": disk, "bytes_per_token": (m["total"] / toks) if toks else None, "bytes_per_learned_reply": (m["total"] / m["paths"]) if m["paths"] else None,
                "paths": m["paths"], "uptime_s": time.time() - self.t_start, "time_saved_s": (saved / 1000) if saved is not None else None}

    def cube(self, limit=1000):
        """the 10 x 10 x 10 box: every token's exponent in its own cell (first-seen order). Only for looking at; no part of answering uses it."""
        inv = {c: t for t, c in self.cell_of.items() if c < limit}; uni = self.wheels.get((), {}); cells = []
        for c, t in sorted(inv.items()):
            nxt = self.wheels.get((t,), {}); tot = sum(nxt.values()) or 1
            top = sorted(nxt.items(), key=lambda kv: -kv[1])[:5]
            cells.append({"c": c, "x": c % 10, "y": (c // 10) % 10, "z": c // 100, "piece": "[end]" if t == END else self.dec([t]), "id": t, "ex": exp_text(t), "n": uni.get(t, 0),
                          "next": [[self.cell_of.get(u, -1), round(cnt / tot, 3), "[end]" if u == END else self.dec([u])] for u, cnt in top]})
        return {"cells": cells, "tokens_known": len(self.cell_of), "shown": len(cells)}

    def export_store_text(self, path):
        lines = ["# FROST SI STORE.  token identity: id -> e^x with x = ln(id+1); line: context tokens => next token @ sector start-end (degrees of the 360 degree wheel)"]
        for ctx, w in sorted(self.wheels.items(), key=lambda kv: -sum(kv[1].values())):
            tot = sum(w.values()); a0 = 0.0; secs = []
            for u, c in sorted(w.items(), key=lambda kv: -kv[1]):
                a1 = a0 + 360.0 * c / tot; secs.append(f"{exp_text(u)}@{a0:.2f}-{a1:.2f}"); a0 = a1
            lines.append((",".join(exp_text(t) if t != BOS else "START" for t in ctx) or "ANY") + " => " + " ".join(secs))
        txt = "\n".join(lines); open(path, "w", encoding="utf-8").write(txt)
        return len(txt.encode("utf-8"))

    # ------------------------------------------------------------------ memory trigger + dashboard data
    def memory(self):
        sectors = sum(len(w) for w in self.wheels.values()); pulls = sum(len(d) for d in self.pw.values()); hooks = sum(len(h) for h in self.hooks.values())
        pbytes = sum(4 * len(p["reply_ids"]) + 64 for p in self.paths.values())
        dh = sum(len(v) for v in self.dhooks.values()); gb = sum(sum(4 * len(r["ids"]) + 48 for r in g["replies"]) + 16 * len(g["utts"]) for g in self.groups.values())
        parts = {"wheel sectors": sectors * 8, "prompt-word pulls": pulls * 8, "hooks": hooks * 12, "stored replies": pbytes}
        if self.groups: parts["lesson hooks and replies"] = dh * 12 + gb
        return {"parts": parts, "total": sum(parts.values()), "wheels": len(self.wheels), "sectors": sectors, "pulls": pulls, "hooks": hooks, "paths": len(self.paths)}

    def check_memory(self):
        m = self.memory(); thr = self.settings["memory_threshold_mb"] * MB
        if m["total"] >= thr and not self.settings["alert_active"]:
            self.settings["alert_active"] = True
            a = {"time": time.time(), "text": f"MEMORY THRESHOLD CROSSED: {m['total'] / MB:.2f} MB used, threshold {thr / MB:.2f} MB. Raise the threshold or decide what to do next.", "bytes": m["total"]}
            self.alerts.append(a); open(self.alert_file, "a", encoding="utf-8").write(json.dumps(a) + "\n")
        elif m["total"] < thr and self.settings["alert_active"]:
            self.settings["alert_active"] = False
        self.save_settings()

    def save_settings(self):
        json.dump(self.settings, open(self.set_file, "w", encoding="utf-8"))

    def state(self):
        m = self.memory(); thr = self.settings["memory_threshold_mb"] * MB
        return {"memory": m, "threshold_bytes": thr, "alert_active": self.settings["alert_active"], "alerts": self.alerts[-5:], "settings": self.settings, "stats": self.stats, "series": self.series,
                "window": self.window[-40:][::-1], "tokens_known": len(self.cell_of), "metrics": self.metrics(), "paths_list": [{"id": p["id"], "prompt": p["prompt"], "source": p["source"], "hits": p["hits"]} for p in sorted(self.paths.values(), key=lambda p: -p["created"])[:30]]}

    def wheel_view(self, text):
        """the wheel for the last tokens of `text` (with back-off) and its sectors"""
        ids = self.tok.encode(text) if text.strip() else []; hist = [BOS] * self.K + ids
        for k in range(self.K, -1, -1):
            ctx = tuple(hist[len(hist) - k:]) if k else (); w = self.wheels.get(ctx)
            if w:
                tot = sum(w.values()); a0 = 0.0; sec = []
                for u, c in sorted(w.items(), key=lambda kv: -kv[1])[:10]:
                    a1 = a0 + 360.0 * c / tot; sec.append({"token": "[end]" if u == END else self.dec([u]), "identity": exp_text(u), "share": c / tot, "from": round(a0, 2), "to": round(a1, 2)}); a0 = a1
                return {"context": [("[start]" if t == BOS else self.dec([t])) for t in ctx] or ["(any)"], "context_identities": [exp_text(t) for t in ctx if t != BOS], "n": tot, "order": k, "sectors": sec}
        return {"context": [], "n": 0, "order": -1, "sectors": []}

    def identities(self, text):
        ids = self.tok.encode(text)
        return [{"piece": self.dec([t]), "id": t, "identity": exp_text(t), "decoded": exp_decode(exp_text(t))} for t in ids]
