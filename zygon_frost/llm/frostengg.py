"""Frost Engg: structures from photos/video, learned as parts with measured sizes.
The shape finder is the read-only yanthros copy (yanthros_backup/vision/structure.py: planes and cylinders by RANSAC). Frost Engg stores what it finds
(parts, sizes, wireframe lines) in model/engg_solo, finds similar structures, and keeps every measurement with its source.
    learn_cloud(name, points)   find the parts, store them
    learn_files(name, structure_json, wireframe_json)   store a structure that was already analysed
    measure(id, scale, unit, density)   sizes in real units
    similar(id) / find(text)    look up
    examples() / score()        simple shapes with known sizes, to check the measurements"""
import hashlib, json, math, os, re, sys, time
sys.dont_write_bytecode = True
import threading
import numpy as np
LOCK = threading.Lock()

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
LINK = os.path.join(ROOT, "yanthros_backup"); ORIGINAL = os.path.join(os.path.expanduser("~"), "Desktop", "yanthros")
STORE = os.path.join(ROOT, "model", "engg_solo"); SRC = os.path.join("vision", "structure.py")
sys.path.insert(0, os.path.join(LINK, "vision"))
import structure as ST                                                   # read only: nothing in the copy is written
MATERIALS = {"steel": 7850, "aluminium": 2720, "stainless steel": 8000, "brass": 8500, "copper": 8940, "cast iron": 7200, "ABS plastic": 1060, "nylon": 1140, "concrete": 2400, "timber (pine)": 500}


def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


class Engg:
    def __init__(self, store=STORE):
        self.store = store; os.makedirs(store, exist_ok=True); self.S = {}; self.post = {}
        p = os.path.join(store, "structures.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"): r = json.loads(line); self.S[r["id"]] = r
        self._index()

    def _index(self):
        self.post = {}
        for r in self.S.values():
            for w in self._words(r): self.post.setdefault(w, set()).add(r["id"])

    @staticmethod
    def _words(r):
        ws = set(re.findall(r"[a-z0-9]+", r["name"].lower()))
        for p in r["st"]["primitives"]: ws.add({"plane": "plate", "cylinder": "cylinder"}[p["type"]]); ws.add(p["type"])
        if any(p["type"] == "plane" for p in r["st"]["primitives"]) and any(p["type"] == "cylinder" for p in r["st"]["primitives"]): ws.add("mixed")
        return ws

    def log(self, kind, metrics):
        with open(os.path.join(self.store, "runs.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **metrics}) + "\n")

    def link_status(self):
        src = os.path.join(LINK, SRC); o = os.path.join(ORIGINAL, SRC)
        return {"copy": LINK, "copy_matches_original": (sha(src) == sha(o)) if os.path.exists(o) else None}

    # ---------------------------------------------------------------- learning
    def _add(self, name, st, lines, source, truth=None):
        nid = max(self.S, default=0) + 1; allp = np.array([p for pl in lines for p in pl]) if lines else np.zeros((1, 3))
        r = {"id": nid, "name": name, "source": source, "st": st, "lines": lines, "bbox": [allp.min(0).tolist(), allp.max(0).tolist()], "truth": truth, "learned_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.S[nid] = r; open(os.path.join(self.store, "structures.jsonl"), "a", encoding="utf-8").write(json.dumps(r) + "\n"); self._index(); return r

    @staticmethod
    def tidy(P, st, lines):
        """drop a cylinder that is wider or longer than the whole object (it cannot be real). wireframe lines go with it: a plane has 1 line, a cylinder 3"""
        ext = P.max(0) - P.min(0); keep, out, k = [], [], 0
        for p in st["primitives"]:
            n = 1 if p["type"] == "plane" else 3; grp = lines[k:k + n]; k += n
            if p["type"] == "cylinder":
                ax = np.array(p["axis"]); along = float(np.abs(ax) @ ext); across = float(np.sqrt(max(ext @ ext - along * along, 0)))
                if 2 * p["radius"] > 1.1 * across / math.sqrt(2) or p["length"] > 1.1 * along: continue
            keep.append(p); out += grp
        return {**st, "primitives": keep}, out

    def analyse(self, P, seed=0, mode="best"):
        """mode raw = the yanthros finder as it is. best = also try flat faces only, keep the simpler reading when it explains the points as well, and drop impossible cylinders"""
        P = np.asarray(P, float)
        if mode == "raw": return ST.analyse(P, seed=seed) + ("raw",)
        with LOCK:
            mixed = ST.analyse(P, seed=seed); real = ST.ransac_cylinder; ST.ransac_cylinder = lambda *a, **k: None
            try: flat = ST.analyse(P, seed=seed)
            finally: ST.ransac_cylinder = real
        cands = []
        for tag, (st, lines) in (("mixed", mixed), ("flat faces only", flat)):
            st, lines = self.tidy(P, st, lines); cands.append((st["unexplained_points"] / max(st["points_used"], 1) + 0.02 * len(st["primitives"]), tag, st, lines))
        sc, tag, st, lines = min(cands, key=lambda c: c[0]); return st, lines, tag

    def learn_cloud(self, name, P, source="point cloud", truth=None, seed=0, mode="best"):
        t0 = time.perf_counter(); st, lines, tag = self.analyse(P, seed, mode); st = {**st, "reading": tag}; secs = time.perf_counter() - t0
        r = self._add(name, st, lines, source, truth); r["seconds"] = round(secs, 2); return r

    def learn_files(self, name, structure_text, wireframe_text=None, source="structure file"):
        st = json.loads(structure_text); lines = json.loads(wireframe_text) if wireframe_text else []
        if "primitives" not in st: raise ValueError("this file has no 'primitives' list")
        return self._add(name, st, lines, source)

    def forget(self, nid):
        self.S.pop(nid, None); open(os.path.join(self.store, "structures.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in self.S.values())); self._index()

    # ---------------------------------------------------------------- looking up
    def listing(self): return [{"id": r["id"], "name": r["name"], "parts": [p["type"] for p in r["st"]["primitives"]], "source": r["source"]} for r in self.S.values()]

    def find(self, text, k=8):
        toks = [t for t in re.findall(r"[a-z0-9]+", text.lower())]; N = len(self.S) or 1; sc = {}
        for t in toks:
            hit = self.post.get(t, set())
            if hit:
                for i in hit: sc[i] = sc.get(i, 0) + math.log((N + 1) / (len(hit) + 0.5))
        return [{"id": i, "name": self.S[i]["name"], "score": round(s, 2)} for i, s in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]

    @staticmethod
    def _sizes(r):
        out = []
        for p in r["st"]["primitives"]: out.append(math.log(p["area"]) if p["type"] == "plane" else math.log(max(p["volume"], 1e-9)) / 1.5)
        return sorted(out)

    def similar(self, nid, k=5):
        a = self.S[nid]; sa = self._sizes(a); out = []
        for i, b in self.S.items():
            if i == nid: continue
            sb = self._sizes(b); n = max(len(sa), len(sb)); x = sa + [-9] * (n - len(sa)); y = sb + [-9] * (n - len(sb))
            ta = sorted(p["type"] for p in a["st"]["primitives"]); tb = sorted(p["type"] for p in b["st"]["primitives"])
            d = sum(abs(u - v) for u, v in zip(x, y)) / n + 0.5 * abs(ta.count("cylinder") - tb.count("cylinder")) + 0.2 * abs(len(ta) - len(tb))
            out.append({"id": i, "name": b["name"], "distance": round(d, 3)})
        return sorted(out, key=lambda x: x["distance"])[:k]

    # ---------------------------------------------------------------- measuring (real units)
    def measure(self, nid, scale=1.0, unit="m", density=None):
        r = self.S[nid]; s = float(scale); parts = []; area = 0.0
        for i, p in enumerate(r["st"]["primitives"]):
            if p["type"] == "plane":
                a = p["area"] * s * s; ext = sorted((e * s for e in p["extent"]), reverse=True)[:2]; area += a
                parts.append({"n": i + 1, "type": "flat face", "area": a, "size": f"about {ext[0]:.4g} x {ext[1]:.4g} {unit}", "points": p["points"]})
            else:
                rad, L = p["radius"] * s, p["length"] * s; v = math.pi * rad * rad * L; lat = 2 * math.pi * rad * L; area += lat
                parts.append({"n": i + 1, "type": "cylinder", "radius": rad, "length": L, "diameter": 2 * rad, "volume": v, "side_area": lat, "mass": (v * density) if density else None, "points": p["points"]})
        lo, hi = np.array(r["bbox"][0]) * s, np.array(r["bbox"][1]) * s
        return {"id": nid, "name": r["name"], "unit": unit, "scale": s, "parts": parts, "surface_area": area, "box": (hi - lo).tolist(), "unexplained_points": r["st"]["unexplained_points"], "points_used": r["st"]["points_used"],
                "note": "sizes are correct up to the scale you give; flat faces use the outline of their points, so a face with a hole or a notch reads larger than it is"}


# ---------------------------------------------------------------- simple shapes with known sizes
def _box(lo, hi, n, rng):
    lo, hi = np.array(lo, float), np.array(hi, float); d = hi - lo; faces = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]; a = d[o[0]] * d[o[1]]
        for side in (lo[ax], hi[ax]): faces.append((ax, o, side, a))
    tot = sum(f[3] for f in faces); pts = []
    for ax, o, side, a in faces:
        m = max(int(n * a / tot), 20); P = np.zeros((m, 3)); P[:, ax] = side
        P[:, o[0]] = rng.uniform(lo[o[0]], hi[o[0]], m); P[:, o[1]] = rng.uniform(lo[o[1]], hi[o[1]], m); pts.append(P)
    return np.vstack(pts)


def _cyl(c, r, L, n, rng, caps=(True, True)):
    lat = 2 * math.pi * r * L; cap = math.pi * r * r; tot = lat + cap * sum(caps); m = int(n * lat / tot); th = rng.uniform(0, 2 * math.pi, m)
    pts = [np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th), c[2] + rng.uniform(0, L, m)]]
    for z, on in zip((c[2], c[2] + L), caps):
        if on:
            k = int(n * cap / tot); rr = r * np.sqrt(rng.uniform(0, 1, k)); t = rng.uniform(0, 2 * math.pi, k); pts.append(np.c_[c[0] + rr * np.cos(t), c[1] + rr * np.sin(t), np.full(k, z)])
    return np.vstack(pts)


def _inside(P, lo, hi, m=1e-6): return np.all((P > np.array(lo) + m) & (P < np.array(hi) - m), axis=1)


def example(kind, seed=0, n=8000, noise=0.003, outliers=0.003):
    """returns (points, truth). truth = true parts: flat faces by area, cylinders by radius and length"""
    rng = np.random.default_rng(seed); T = []
    if kind == "box 2 x 1 x 1":
        P = _box((0, 0, 0), (2, 1, 1), n, rng); T = [("plane", 2.0)] * 4 + [("plane", 1.0)] * 2
    elif kind == "cube 1":
        P = _box((0, 0, 0), (1, 1, 1), n, rng); T = [("plane", 1.0)] * 6
    elif kind == "cylinder r0.3 L1.4":
        P = _cyl((0, 0, 0), 0.3, 1.4, n, rng); T = [("cylinder", 0.3, 1.4), ("plane", math.pi * 0.09), ("plane", math.pi * 0.09)]
    elif kind == "plate with post":
        B = _box((0, 0, 0), (1.2, 1.2, 0.2), n // 2, rng); C = _cyl((0.6, 0.6, 0.2), 0.25, 1.0, n // 2, rng, caps=(False, True))
        B = B[~(np.hypot(B[:, 0] - 0.6, B[:, 1] - 0.6) < 0.25) | (B[:, 2] < 0.1)]; P = np.vstack([B, C])
        T = [("plane", 1.44), ("plane", 1.44 - math.pi * 0.0625)] + [("plane", 0.24)] * 4 + [("cylinder", 0.25, 1.0), ("plane", math.pi * 0.0625)]
    elif kind == "L bracket":
        a, b = ((0, 0, 0), (2, 1, 0.2)), ((0, 0, 0.2), (0.2, 1, 1.5)); A = _box(*a, n // 2, rng); Bx = _box(*b, n // 2, rng)
        A = A[~_inside(A, *b)]; Bx = Bx[~_inside(Bx, *a)]; A = A[~((A[:, 2] > 0.2 - 1e-6) & (A[:, 0] < 0.2 + 1e-6) & (A[:, 2] < 0.2 + 1e-6))]; Bx = Bx[Bx[:, 2] > 0.2 + 1e-6] if False else Bx
        P = np.vstack([A, Bx]); T = [("plane", 2.0), ("plane", 1.8), ("plane", 1.5), ("plane", 1.3), ("plane", 0.66), ("plane", 0.66), ("plane", 0.2), ("plane", 0.2)]
    else: raise ValueError(kind)
    P = P + rng.normal(0, noise, P.shape); lo, hi = P.min(0), P.max(0); k = int(outliers * len(P)); P = np.vstack([P, rng.uniform(lo - 0.3, hi + 0.3, (k, 3))])
    truth = [{"type": t[0], "area": t[1]} if t[0] == "plane" else {"type": "cylinder", "radius": t[1], "length": t[2]} for t in T]
    return P, truth


EXAMPLES = ["cube 1", "box 2 x 1 x 1", "cylinder r0.3 L1.4", "plate with post", "L bracket"]


def score(st, truth):
    """each true part is matched to the closest found part of the same kind (by size). Reports counts and relative errors."""
    found = [dict(p) for p in st["primitives"]]; used = set(); errs = []; missed = 0
    for t in truth:
        best, be = None, None
        for i, p in enumerate(found):
            if i in used or p["type"] != t["type"]: continue
            e = abs(p["area"] - t["area"]) / t["area"] if t["type"] == "plane" else max(abs(p["radius"] - t["radius"]) / t["radius"], abs(p["length"] - t["length"]) / t["length"])
            if be is None or e < be: best, be = i, e
        if best is None: missed += 1
        else: used.add(best); errs.append(be)
    return {"true_parts": len(truth), "found_parts": len(found), "matched": len(errs), "missed": missed, "extra": len(found) - len(used),
            "median_error": round(float(np.median(errs)), 4) if errs else None, "max_error": round(float(np.max(errs)), 4) if errs else None, "errors": [round(float(e), 4) for e in errs]}
