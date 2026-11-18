"""ZYGON 2.0 (experiment): letters instead of word-pieces.

 * ring      every character sits at one fixed angle of one ring (a to z first, then capitals, digits, punctuation, space)
 * thread    everything it reads is ONE long thread that winds round the ring millions of times: a spiral whose position is an exponent and whose angle is the letter
 * learning  appending text to the thread (plus a small index of 6-letter and 3-letter stretches); nothing else
 * next letter  find the longest stretch of thread that matches what was just written and continue it; stretches that belong to sentences the question hooked into count triple
 * finding   questions and sentences are compared by letter triplets, so typing mistakes and new words still match
No matrices, no word list, no tokenizer. Persistence = an append-only log that is replayed on start.
"""
import bisect, json, math, os, random, re, threading, time
from collections import defaultdict

RING = [chr(c) for c in range(32, 127)]                    # 95 characters: the angle of a character is its place on the ring
ANGLE = {c: i * 360.0 / len(RING) for i, c in enumerate(RING)}
CAP = 160                                                  # stretches remembered per index key (the most recent)
SEP = 10


def clean(t):
    out = []
    for c in t:
        o = ord(c)
        out.append(c if 32 <= o < 127 else "'" if c in "‘’" else '"' if c in "“”" else "-" if c in "–—" else " " if c in "\t\r\n" else "?" if o > 127 else " ")
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def tris(text):
    s = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


class SI2:
    def __init__(self, folder):
        self.dir = folder; os.makedirs(folder, exist_ok=True); self.lock = threading.RLock()
        self.items, self.next_id, self.groups, self.next_gid = {}, 1, {}, 1
        self.thread, self.idx6, self.idx3, self.starts, self.start_ids, self.pos_of = bytearray(), defaultdict(list), defaultdict(list), [], [], {}
        self.tri_s, self.tri_w, self.wordings, self.next_wid = defaultdict(set), defaultdict(set), {}, 1
        self.agg, self.dirty, self._tg = {}, False, {}; self.log = os.path.join(folder, "learned.jsonl"); self.t_start = time.time()
        if os.path.exists(self.log):
            for ln in open(self.log, encoding="utf-8"):
                if ln.strip(): self._apply(json.loads(ln), replay=True)

    # ------------------------------------------------------------ metrics
    def _agg(self, key, val):
        a = self.agg.setdefault(key, [0, 0.0, float("inf"), 0.0]); a[0] += 1; a[1] += val; a[2] = min(a[2], val); a[3] = max(a[3], val)

    def reply_stats(self):
        a = self.agg.get("reply"); c = self.agg.get("reply_chars"); nf = self.agg.get("reply_none")
        return {"n": a[0] if a else 0, "not_found": nf[0] if nf else 0, "mean_ms": (a[1] / a[0]) if a else None, "fastest_ms": a[2] if a else None, "slowest_ms": a[3] if a else None, "units_per_s": (c[1] / (a[1] / 1000)) if a and c and a[1] else None, "unit": "letters"}

    # ------------------------------------------------------------ thread
    def _thread_add(self, text, iid):
        t = clean(text).encode("ascii", "replace");
        if not t: return
        start = len(self.thread); self.pos_of[iid] = len(self.starts); self.starts.append(start); self.start_ids.append(iid); self.thread.extend(t); self.thread.append(SEP); end = len(self.thread)
        th = self.thread
        for i in range(start + 5, end - 1):
            for n, idx in ((6, self.idx6), (3, self.idx3)):
                if n == 3 and i < start + 2: continue
                lst = idx[bytes(th[i - n + 1:i + 1])]; lst.append(i)
                if len(lst) > 2 * CAP: del lst[:CAP]
        for i in range(start + 2, min(start + 5, end - 1)):
            lst = self.idx3[bytes(th[i - 2:i + 1])]; lst.append(i)

    def _rebuild(self):
        with self.lock:
            self.thread = bytearray(); self.idx6, self.idx3 = defaultdict(list), defaultdict(list); self.starts, self.start_ids, self.pos_of = [], [], {}
            self.tri_s, self.tri_w = defaultdict(set), defaultdict(set)
            for iid, it in self.items.items():
                if not it["alive"]: continue
                self._thread_add(it["text"], iid)
                if it["kind"] in ("document", "saved answer"):
                    for g in tris(it.get("q") or it["text"]): self.tri_s[g].add(iid)
            for wid, w in self.wordings.items():
                if self.groups.get(w[0], {}).get("alive"):
                    for g in w[2]: self.tri_w[g].add(wid)
            self.dirty = False

    def tg(self, iid, field):
        k = (iid, field)
        if k not in self._tg: self._tg[k] = tris(self.items[iid][field] or "")
        return self._tg[k]

    def _item_at(self, p):
        return self.start_ids[bisect.bisect_right(self.starts, p) - 1] if self.starts else None

    # ------------------------------------------------------------ learning (all events are logged and replayed)
    def _apply(self, ev, replay=False):
        t = ev["t"]
        if t in ("doc", "saved", "read", "reply"):
            iid = self.next_id; self.next_id += 1
            kind = {"doc": "document", "saved": "saved answer", "read": "reading", "reply": "lesson reply"}[t]
            it = {"id": iid, "kind": kind, "text": ev["text"], "q": ev.get("q"), "source": ev.get("source", ""), "name": ev.get("name", ""), "gid": ev.get("gid"), "created": ev.get("time", 0), "alive": True}
            self.items[iid] = it; self._thread_add(ev["text"], iid)
            if kind in ("document", "saved answer"):
                for g in tris(it.get("q") or it["text"]): self.tri_s[g].add(iid)
        elif t == "lesson":
            gid = self.next_gid; self.next_gid += 1; self.groups[gid] = {"gid": gid, "cat": ev["cat"], "utts": ev["utts"], "replies": ev["replies"], "alive": True}
            for u in ev["utts"]:
                wid = self.next_wid; self.next_wid += 1; tr = tris(u); self.wordings[wid] = (gid, u, tr)
                for g in tr: self.tri_w[g].add(wid)
            for r in ev["replies"]: self._apply({"t": "reply", "text": r, "gid": gid, "source": "lesson: " + ev["cat"], "time": ev.get("time", 0)}, replay=True)
            return gid if not replay else None
        elif t == "del":
            for iid in ev["ids"]:
                if iid in self.items: self.items[iid]["alive"] = False
            for gid in ev.get("gids", []):
                if gid in self.groups: self.groups[gid]["alive"] = False
            self.dirty = True
        if not replay and t != "lesson_reply_internal" and ev.get("_log", True):
            with self.lock: open(self.log, "a", encoding="utf-8").write(json.dumps({k: v for k, v in ev.items() if k != "_log"}, ensure_ascii=False) + "\n")

    def learn_sentence(self, text, source, name=""):
        t0 = time.perf_counter(); self._apply({"t": "doc", "text": text, "source": source, "name": name, "time": time.time()}); self._agg("learn:doc", (time.perf_counter() - t0) * 1000)

    def learn_text(self, text, label=""):
        t0 = time.perf_counter(); self._apply({"t": "read", "text": text, "source": label, "time": time.time()}); self._agg("learn:read", (time.perf_counter() - t0) * 1000); return len(clean(text))

    def learn_lesson(self, cat, utts, replies, source="lesson"):
        t0 = time.perf_counter()
        with self.lock:
            ev = {"t": "lesson", "cat": cat, "utts": list(utts), "replies": list(replies), "time": time.time()}
            gid = self._apply(ev, replay=True)                             # build in memory first ...
            with open(self.log, "a", encoding="utf-8") as f: f.write(json.dumps(ev, ensure_ascii=False) + "\n")      # ... then log the lesson once (its replies are re-created from it on replay)
        self._agg("learn:lesson", (time.perf_counter() - t0) * 1000); return gid

    def remember(self, question, answer, source):
        t0 = time.perf_counter(); self._apply({"t": "saved", "q": question, "text": answer, "source": source, "time": time.time()}); self._agg("learn:saved", (time.perf_counter() - t0) * 1000)
        return f"s{self.next_id - 1}"

    # ------------------------------------------------------------ finding by letter triplets
    @staticmethod
    def _cands(qt, index, k=10):
        """candidates are looked up from the rarest letter triplets of the question only (the common ones match nearly everything)"""
        rare = sorted((g for g in qt if g in index), key=lambda g: len(index[g]))[:k]; out = set()
        for g in rare: out |= index[g]
        return out

    def _score(self, qt, index, universe_n, cands, getter):
        df = lambda g: len(index.get(g, ()))
        idf = lambda g: math.log((universe_n + 1) / (df(g) + 0.5)); qtot = sum(idf(g) for g in qt) or 1.0; out = []
        for cid in cands:
            ct = getter(cid); shared = qt & ct; sh = sum(idf(g) for g in shared)
            out.append((sh / (sum(idf(g) for g in ct) or 1.0), sh / qtot, cid))
        return out

    def converse(self, text, min_cover=0.4, min_prec=0.4):
        t0 = time.perf_counter()
        if self.dirty: self._rebuild()
        qt = tris(text)
        if len(qt) < 2 or not self.wordings: return None
        cands = self._cands(qt, self.tri_w)
        sc = [(cov * (0.5 + 0.5 * prec), cov, prec, wid) for cov, prec, wid in self._score(qt, self.tri_w, len(self.wordings), cands, lambda w: self.wordings[w][2]) if cov >= min_cover and prec >= min_prec]
        if not sc: return None
        sc.sort(reverse=True); top = sc[0][0]; s, cov, prec, wid = random.choice([x for x in sc if x[0] >= top - 0.02]); gid, u, _ = self.wordings[wid]; g = self.groups[gid]
        rep = random.choice(g["replies"]); ms = (time.perf_counter() - t0) * 1000; self._agg("reply", ms); self._agg("reply_chars", len(rep))
        return {"found": True, "kind": "dialogue", "text": rep, "cat": g["cat"], "matched": u, "score": round(s, 3), "ms": ms}

    def saved_answer(self, q, min_cover=0.7, min_prec=0.7):
        if self.dirty: self._rebuild()
        qt = tris(q); cands = self._cands(qt, self.tri_s)
        best = None
        for cov, prec, iid in self._score(qt, self.tri_s, len(self.items), [c for c in cands if self.items[c]["kind"] == "saved answer"], lambda i: self.tg(i, "q")):
            if cov >= min_cover and prec >= min_prec and (best is None or cov * prec > best[0]): best = (cov * prec, iid)
        if not best: return None
        it = self.items[best[1]]; return {"text": it["text"], "source": it["source"].replace("web:", "").strip(), "question": it["q"], "iid": it["id"]}

    def search(self, q, top=6, min_cover=0.45):
        if self.dirty: self._rebuild()
        qt = tris(q); cands = self._cands(qt, self.tri_s)
        sc = [(prec + 0.05 * cov, iid) for cov, prec, iid in self._score(qt, self.tri_s, len(self.items), [c for c in cands if self.items[c]["kind"] == "document"], lambda i: self.tg(i, "text")) if prec >= min_cover]
        sc.sort(reverse=True); return sc[:top]

    # ------------------------------------------------------------ next letter = continue the longest matching stretch of thread
    def _next(self, ctx, steer):
        """candidates for the next letter: first the sentences the question hooked into (read directly, so they are never forgotten), then the index of the whole thread"""
        th = self.thread; cand, best = defaultdict(float), {}
        def vote(p, boost):
            m = 0
            while m < 48 and m < len(ctx) and p - m >= 0 and th[p - m] == ctx[-1 - m]: m += 1
            if p + 1 >= len(th): return
            ch = th[p + 1]; w = float(m * m) * boost; cand[ch] += w
            if w > best.get(ch, (0, 0))[0]: best[ch] = (w, p)
        if len(ctx) >= 4:
            key = bytes(ctx[-4:])
            for iid in steer:
                k = self.pos_of.get(iid)
                if k is None: continue
                s0 = self.starts[k]; e0 = self.starts[k + 1] if k + 1 < len(self.starts) else len(th); seg = bytes(th[s0:e0]); j = seg.find(key)
                while j != -1:
                    vote(s0 + j + 3, 20.0); j = seg.find(key, j + 1)
        if not cand:
            for n, idx in ((6, self.idx6), (3, self.idx3)):
                if len(ctx) < n: continue
                poss = idx.get(bytes(ctx[-n:]))
                if not poss: continue
                for p in poss[-60:]: vote(p, 1.0)
                break
        return cand, best

    def compose(self, q, max_chars=420, max_sent=2, min_cover=0.6):
        t0 = time.perf_counter()
        hits = self.search(q, min_cover=min_cover)
        if not hits:
            self._agg("reply_none", (time.perf_counter() - t0) * 1000)
            return {"found": False, "ms": (time.perf_counter() - t0) * 1000}
        steer = {iid for _, iid in hits}; cur = hits[0][1]; out = bytearray(clean(self.items[cur]["text"])[:30].encode()); labels = [cur] * len(out); seen = set(); sents = 0
        while len(out) < max_chars:
            cand, best = self._next(out[-48:], steer)
            if not cand: break
            for ch in sorted(cand, key=cand.get, reverse=True):
                tail = bytes(out[-9:]) + bytes([ch])
                if tail in seen: cand[ch] *= 0.02
            ch = max(cand, key=cand.get)
            if ch == SEP:
                sents += 1; nxt = cur + 1
                if sents >= max_sent or nxt not in self.items or not self.items[nxt]["alive"] or self.items[nxt]["kind"] != "document": break
                cur = nxt; steer.add(cur); out.append(32); labels.append(cur); seed = clean(self.items[cur]["text"])[:30].encode(); out.extend(seed); labels.extend([cur] * len(seed)); continue
            out.append(ch); seen.add(bytes(out[-10:])); labels.append(self._item_at(best[ch][1]) if ch in best else cur)
        segs = []
        for c, lab in zip(out, labels):
            if segs and segs[-1]["iid"] == lab: segs[-1]["text"] += chr(c)
            else: segs.append({"iid": lab, "text": chr(c)})
        for s in segs: it = self.items.get(s["iid"], {}); s["source"] = it.get("source", ""); s["sentence"] = it.get("text", "")[:90]
        ms = (time.perf_counter() - t0) * 1000; self._agg("reply", ms); self._agg("reply_chars", len(out))
        return {"found": True, "kind": "document", "text": out.decode(), "segments": segs, "ms": ms, "letters": len(out), "joins": max(0, len(segs) - 1),
                "matched": [{"iid": i, "cover": round(sc, 3), "source": self.items[i]["source"], "text": self.items[i]["text"][:110]} for sc, i in hits[:4]]}

    def complete(self, text, max_chars=300):
        steer = set(); out = bytearray(clean(text).encode()); n0 = len(out); seen = set()
        for _ in range(max_chars):
            cand, _b = self._next(out[-48:], steer)
            if not cand: break
            ch = max(cand, key=cand.get)
            if ch == SEP: break
            out.append(ch)
        return out[n0:].decode()

    # ------------------------------------------------------------ what it knows, and removing things
    def memory(self):
        if self.dirty: self._rebuild()
        idx = sum(len(v) for v in self.idx6.values()) + sum(len(v) for v in self.idx3.values()); tri = sum(len(v) for v in self.tri_s.values()) + sum(len(v) for v in self.tri_w.values())
        txt = sum(len(i["text"]) + len(i.get("q") or "") for i in self.items.values() if i["alive"])
        parts = {"letter thread": len(self.thread), "stretch index": idx * 8, "triplet index": tri * 12, "stored text": txt}
        return {"total": sum(parts.values()), "parts": parts, "letters": len(self.thread), "index_entries": idx, "triplets": tri, "items": sum(1 for i in self.items.values() if i["alive"]), "lessons": sum(1 for g in self.groups.values() if g["alive"])}

    def learned_items(self, query="", limit=300):
        out, docs = [], {}
        for it in self.items.values():
            if not it["alive"]: continue
            if it["kind"] == "saved answer": out.append({"id": f"s{it['id']}", "kind": "saved answer", "title": it["q"], "preview": it["text"][:220], "source": it["source"].replace("web:", "").strip(), "when": it["created"]})
            elif it["kind"] == "document":
                d = docs.setdefault(it["name"], {"n": 0, "pages": set(), "when": it["created"]}); d["n"] += 1
                if " p." in it["source"]: d["pages"].add(it["source"].split(" p.")[1])
        for name, d in docs.items(): out.append({"id": "d:" + name, "kind": "document", "title": name, "preview": f"{d['n']} sentences" + (f" from {len(d['pages'])} pages" if d["pages"] else ""), "source": "document", "when": d["when"]})
        for g in self.groups.values():
            if g["alive"]: out.append({"id": f"g{g['gid']}", "kind": "everyday English", "title": g["utts"][0], "preview": g["replies"][0][:220], "source": g["cat"], "extra": f"{len(g['utts'])} ways of asking", "when": 0})
        q = query.strip().lower()
        if q: out = [o for o in out if q in (o["title"] + " " + o["preview"] + " " + o["source"] + " " + o["kind"]).lower()]
        order = {"saved answer": 0, "document": 1, "everyday English": 2}; out.sort(key=lambda o: (order.get(o["kind"], 3), -o["when"], o["title"]))
        return {"total": len(out), "items": out[:limit]}

    def delete_item(self, item_id):
        ids, gids = [], []
        if item_id.startswith("s") and item_id[1:].isdigit(): ids = [int(item_id[1:])]
        elif item_id.startswith("d:"): ids = [i["id"] for i in self.items.values() if i["alive"] and i["kind"] == "document" and i["name"] == item_id[2:]]
        elif item_id.startswith("g") and item_id[1:].isdigit():
            gids = [int(item_id[1:])]; ids = [i["id"] for i in self.items.values() if i["gid"] == gids[0]]
        if not ids and not gids: return 0
        self._apply({"t": "del", "ids": ids, "gids": gids, "time": time.time()}); return len(ids) + len(gids)
