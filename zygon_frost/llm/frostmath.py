"""Frost Math: formulas as records, exact answers, every step shown.
The formulas come from a read-only copy of yanthros (yanthros_backup/engine/formulas.py). Learning copies them into model/math_solo;
the copy is never edited and its checksum is checked on every start.
    learn()                      read the copy, write the store, log a run
    find(text)                   formulas whose name, area or variables match the words (idf, like Frost Lang)
    solve_one(name, vals, tgt)   one formula, one unknown, exact numbers, with the check
    solve_chain(vals, tgt)       any route through the formulas from what you know to what you want
Numbers are exact (fractions) until the last step. Nothing is rounded before the answer is shown."""
import hashlib, json, math, os, re, sys, time
sys.dont_write_bytecode = True
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
LINK = os.path.join(ROOT, "yanthros_backup"); ORIGINAL = os.path.join(os.path.expanduser("~"), "Desktop", "yanthros")
STORE = os.path.join(ROOT, "model", "math_solo")
SRC_FILE = os.path.join("engine", "formulas.py")
CONSTANTS = {            # shown as a suggestion in the form, never filled in silently
    "g_grav": ("9.80665", "standard gravity, exact by definition (CGPM 1901)"), "c_light": ("299792458", "speed of light, exact by definition (SI 2019)"),
    "mu0": ("1.25663706212e-6", "CODATA 2018 (measured)"), "sigma_sb": ("5.670374419e-8", "Stefan-Boltzmann, CODATA 2018"),
}
NATIVE = {"RPM": "rpm", "N_pump": "rpm", "N_pump2": "rpm", "N_s": "rpm", "N_crit_shaft": "rpm", "N_spindle": "rpm", "N_rpm_stepper": "rpm", "D_workpiece": "mm", "Vc_cutting": "m/min", "theta_step": "deg", "T_batt_amb": "C", "VR_sync_pct": "pct", "err_pct": "pct"}
UNRESOLVED = {"air_gap_length_im", "chain_velocity"}      # a number inside the formula fixes its units and I could not tell which: plain numbers only
PREFERRED = ["V", "A", "ohm", "W", "Pa", "N", "J", "N.m", "m", "m2", "m3", "s", "kg", "Hz", "K", "m/s", "m3/s", "kg/s", "kg/m3", "m/min", "W/m2"]
FUNCS = {"sqrt": sp.sqrt, "exp": sp.exp, "ln": sp.log, "log": sp.log, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "pi": sp.pi, "e": sp.E}
NUMLIT = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
OKCHARS = re.compile(r"^[0-9A-Za-z+\-*/(). ]*$")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read())
    return h.hexdigest()


def parse_value(s):
    """text -> exact number.  '0.1' -> 1/10, '3/4', '1e-3', '2*pi', 'sqrt(2)', 'exp(-1/2)'.  Anything else is refused."""
    t = str(s).strip().replace(",", "").replace("^", "**")
    if not t or not OKCHARS.match(t): raise ValueError(f"'{s}' is not a number I can read exactly")
    words = re.findall(r"[A-Za-z]+", NUMLIT.sub("0", t))
    if any(w not in FUNCS for w in words): raise ValueError(f"'{s}' is not a number I can read exactly")
    try: v = sp.sympify(t, rational=True, locals=FUNCS)
    except Exception: raise ValueError(f"'{s}' is not a number I can read exactly")
    if not getattr(v, "is_number", False) or v.has(sp.zoo, sp.nan) or abs(sp.im(sp.N(v, 30))) > 1e-20: raise ValueError(f"'{s}' is not a real number")
    return v


def sig(v, digits=15):
    f = sp.N(v, digits)
    if f == 0: return "0"
    a = abs(float(f))
    if a >= 1e9 or a < 1e-4: return f"{float(f):.{digits - 1}e}".replace("e+0", "e+").replace("e-0", "e-")
    s = str(sp.N(v, digits))
    return s.rstrip("0").rstrip(".") if "." in s else s


class Math:
    def __init__(self, store=STORE):
        self.store = store; os.makedirs(store, exist_ok=True)
        self.F = {}; self.vars = {}; self.post = {}; self.meta = {}
        if os.path.exists(os.path.join(store, "formulas.jsonl")): self.load()

    # ---------------------------------------------------------------- learning
    def learn(self):
        t0 = time.perf_counter(); src = os.path.join(LINK, SRC_FILE)
        sys.path.insert(0, os.path.join(LINK, "engine"))
        import formulas as Y
        recs, vid = [], {}
        for i, (name, eq) in enumerate(Y.FORMULAS.items()):
            vs = sorted(str(s) for s in eq.free_symbols)
            for v in vs: vid.setdefault(v, len(vid))
            recs.append({"id": i, "name": name, "domain": Y.FORMULA_DOMAINS.get(name, "unknown"), "lhs": sp.srepr(eq.lhs), "rhs": sp.srepr(eq.rhs), "text": f"{eq.lhs} = {eq.rhs}", "vars": vs, "source": "yanthros formulas.py (builtin, no citation recorded)"})
        with open(os.path.join(self.store, "formulas.jsonl"), "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        vars_ = {v: {"id": i, "x": round(math.log(i + 1), 6)} for v, i in vid.items()}
        meta = {"linked_copy": src, "linked_sha256": sha(src), "original_sha256": sha(os.path.join(ORIGINAL, SRC_FILE)) if os.path.exists(os.path.join(ORIGINAL, SRC_FILE)) else None,
                "formulas": len(recs), "variables": len(vars_), "domains": len({r["domain"] for r in recs}), "learned_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(vars_, open(os.path.join(self.store, "vars.json"), "w")); json.dump(meta, open(os.path.join(self.store, "meta.json"), "w"), indent=1)
        self.load(); secs = time.perf_counter() - t0
        size = sum(os.path.getsize(os.path.join(self.store, n)) for n in ("formulas.jsonl", "vars.json", "meta.json"))
        self.log("learn", {**meta, "seconds": round(secs, 2), "store_bytes": size})
        return {**meta, "seconds": round(secs, 2), "store_bytes": size}

    def load(self):
        self.F = {}; self.post = {}
        lp = os.path.join(self.store, "labels.json"); self.labels = json.load(open(lp, encoding="utf-8")) if os.path.exists(lp) else {}
        for line in open(os.path.join(self.store, "formulas.jsonl"), encoding="utf-8"):
            r = json.loads(line); r["eq"] = sp.Eq(sp.sympify(r["lhs"]), sp.sympify(r["rhs"])); self.F[r["name"]] = r
        self.vars = json.load(open(os.path.join(self.store, "vars.json"))); self.meta = json.load(open(os.path.join(self.store, "meta.json")))
        up = os.path.join(self.store, "units.json"); self.udim = {}
        if os.path.exists(up):
            from fractions import Fraction
            self.udim = {v: tuple(Fraction(x) for x in e["dims"]) for v, e in json.load(open(up))["variables"].items()}
        self.by_var = {}
        for n, r in self.F.items():
            for v in r["vars"]: self.by_var.setdefault(v, []).append(n)
            for w in self._words(r): self.post.setdefault(w, set()).add(n)

    def log(self, kind, metrics):
        with open(os.path.join(self.store, "runs.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **metrics}) + "\n")

    def link_status(self):
        src = os.path.join(LINK, SRC_FILE); now = sha(src) if os.path.exists(src) else None
        orig = os.path.join(ORIGINAL, SRC_FILE); o = sha(orig) if os.path.exists(orig) else None
        return {"copy_unchanged": now == self.meta.get("linked_sha256"), "copy_matches_original": (now == o) if o else None, "copy": LINK}

    # ---------------------------------------------------------------- finding by words
    USE_LABELS = True                                   # plain names, aliases and variable labels count as words of a formula

    def _words(self, r):
        ws = set()
        lab = self.labels.get(r["name"]) if self.USE_LABELS else None
        if lab:
            for t in re.findall(r"[A-Za-z]{2,}", " ".join([lab["title"]] + lab["aliases"] + list(lab["vars"].values())).lower()):
                if t not in self.STOP: ws.add(t.rstrip("s") if len(t) > 3 else t)
        for part in re.split(r"[_\W]+", r["name"] + " " + r["domain"]):
            if part: ws.add(part.lower().rstrip("s") if len(part) > 3 else part.lower())
        for v in r["vars"]:
            ws.add(v.lower())
            for part in re.split(r"_", v):
                if len(part) > 2: ws.add(part.lower())
        return ws

    MP = {"on": False, "tau": 0.5, "lam": 0.6, "k": 3, "mode": "unmatched"}
    STOP = frozenset("the a an of to in on for and or is are be how what which when does do can with from that this it its at by as into than then there their about much many give find tell".split())

    def _meaning(self):
        if getattr(self, "_mn", None) is None:
            try:
                import meaning
                self._mn = meaning.Meaning(); self._mcand = [w for w in self.post if w in self._mn.ix]
            except Exception: self._mn = False
        return self._mn or None

    def find(self, text, k=8):
        raw = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]; toks = [t.rstrip("s") if len(t) > 3 else t for t in raw]
        N = len(self.F); score = {}
        if self.MP["on"] and self._meaning():
            for t, r in zip(toks, raw):
                if t in self.STOP or r in self.STOP or len(r) < 3 or (self.MP["mode"] == "unmatched" and t in self.post): continue
                for w, sm in self._mn.nearest(r, self._mcand, self.MP["k"], self.MP["tau"]):
                    hit = self.post.get(w, set())
                    if hit:
                        idf = math.log((N + 1) / (len(hit) + 0.5))
                        for n in hit: score[n] = score.get(n, 0) + self.MP["lam"] * sm * idf
        for t in toks:
            hit = self.post.get(t, set())
            if not hit: continue
            idf = math.log((N + 1) / (len(hit) + 0.5))
            for n in hit: score[n] = score.get(n, 0) + idf
        for t in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):                    # an exact variable name counts too
            if t in self.by_var:
                for n in self.by_var[t]: score[n] = score.get(n, 0) + 0.5
        out = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self.card(n) | {"score": round(s, 2)} for n, s in out]

    def card(self, name):
        r = self.F[name]; lab = self.labels.get(name) or {}
        return {"name": name, "title": lab.get("title") or name.replace("_", " "), "var_labels": lab.get("vars", {}), "domain": r["domain"], "equation": r["text"], "vars": r["vars"], "source": r["source"], "output": str(r["eq"].lhs)}

    def domains(self):
        d = {}
        for r in self.F.values(): d.setdefault(r["domain"], []).append(r["name"])
        return {k: sorted(v) for k, v in sorted(d.items())}

    def suggest_constants(self, vars_): return {v: {"value": CONSTANTS[v][0], "why": CONSTANTS[v][1]} for v in vars_ if v in CONSTANTS}

    # ---------------------------------------------------------------- units
    def label_of(self, v):
        """the unit the box takes when a plain number is typed"""
        if v in NATIVE: return NATIVE[v]
        dm = self.udim.get(v)
        if dm is None: return None
        import units
        for n in PREFERRED:
            if n in units.U and units.same_dims(units.U[n][1], dm): return n
        return units.dim_text(dm)

    def var_units(self, vs): return {v: {"label": self.label_of(v), "known": v in self.udim} for v in vs}

    def read_value(self, text, v, formula=None):
        """typed text -> exact number in the unit the formula uses. '5 ft' is converted; a plain number is taken as that unit already."""
        t = str(text).strip()
        try: return parse_value(t), None
        except ValueError: pass
        import units
        if v not in self.udim: raise ValueError(f"the unit of {v} is not known yet, so type a plain number")
        if formula in UNRESOLVED: raise ValueError(f"{formula} has a built-in number that fixes its units; type a plain number")
        si, note = units.to_si(t, self.udim[v], v)
        if v in NATIVE:
            fac, dm, off, lab = units.unit_expr(NATIVE[v]); pure = units.same_dims(dm, units.TEMP) and off != 0
            return (si - (off if pure else 0)) / fac, (note or "") + f" (box takes {lab})"
        return si, note

    def alt_units(self, val, v, k=6):
        """the answer in other units that fit"""
        import units
        dm = self.udim.get(v)
        if dm is None: return []
        if v in NATIVE:
            fac, d_, off, lab = units.unit_expr(NATIVE[v]); pure = units.same_dims(d_, units.TEMP) and off != 0; si = val * fac + (off if pure else 0)
        else: si = val
        out = []
        for n in units.compatible(dm, 12):
            try:
                x, lab = units.from_si(si, n, v); out.append({"unit": n, "label": lab, "value": sig(x, 12)})
            except Exception: pass
        return out[:k]

    # ---------------------------------------------------------------- one formula, one unknown
    def solve_one(self, name, vals, target, choice=None):
        if name not in self.F: return {"ok": False, "error": f"no formula called {name}"}
        r = self.F[name]; eq = r["eq"]; vs = r["vars"]
        if target not in vs: return {"ok": False, "error": f"{target} is not in {name} ({r['text']})"}
        need = [v for v in vs if v != target]; missing = [v for v in need if v not in vals or str(vals[v]).strip() == ""]
        if missing: return {"ok": False, "error": "missing values", "missing": missing, "need": need}
        ex, unit_notes = {}, []
        try:
            for v in need:
                ex[v], nt = self.read_value(vals[v], v, name)
                if nt: unit_notes.append(f"{v}: {nt}")
        except ValueError as e: return {"ok": False, "error": str(e)}
        t = sp.Symbol(target); sub = {sp.Symbol(v): x for v, x in ex.items()}
        import types
        eq_sub = types.SimpleNamespace(lhs=eq.lhs.subs(sub), rhs=eq.rhs.subs(sub))              # not sp.Eq: sympy turns an equal pair into True and loses the sides
        if eq_sub.lhs.has(sp.zoo, sp.nan) or eq_sub.rhs.has(sp.zoo, sp.nan): return {"ok": False, "error": "these values make the formula divide by zero"}
        if sp.simplify(eq_sub.lhs - eq_sub.rhs) == 0: return {"ok": False, "error": f"with these values the formula holds for any {target}, so {target} cannot be found"}
        numeric = False
        try: sols = sp.solve(eq_sub.lhs - eq_sub.rhs, t)
        except Exception as e:
            sols = self._numeric_roots(eq_sub.lhs - eq_sub.rhs, t); numeric = True
            if not sols: return {"ok": False, "error": f"no exact solution found ({type(e).__name__}) and no number found either"}
        if not sols and not numeric: sols = self._numeric_roots(eq_sub.lhs - eq_sub.rhs, t); numeric = True
        real, dropped = [], 0
        for s in sols:
            n = sp.N(s, 60)
            if n.has(sp.zoo, sp.nan) or abs(sp.im(n)) > sp.Float("1e-50", 60): dropped += 1; continue
            real.append(s)
        if not real: return {"ok": False, "error": "there is no real answer for these values" + (f" ({dropped} complex)" if dropped else "")}
        steps = {"formula": name, "domain": r["domain"], "source": r["source"], "equation": r["text"], "values": {v: str(ex[v]) for v in need}, "solving_for": target,
                 "units": {v: self.label_of(v) for v in vs}, "unit_notes": unit_notes}
        if numeric: steps["method"] = "no closed form exists for this unknown; the number was found by root search to 50 digits (shown as exact below), then put back into the formula"
        try:
            if len(real) == 1 or choice is not None: pass
            if sp.count_ops(eq.lhs - eq.rhs) <= 40:
                rs = sp.solve(eq.lhs - eq.rhs, t); steps["rearranged"] = " or ".join(f"{target} = {x}" for x in rs[:3]) if rs else None
        except Exception: steps["rearranged"] = None
        steps["substituted"] = f"{eq_sub.lhs} = {eq_sub.rhs}"
        periodic = any(t in f.free_symbols for f in eq.atoms(sp.sin, sp.cos, sp.tan))
        if periodic: steps["repeats"] = "this formula repeats (sin/cos/tan): adding full turns gives other values that also fit; the ones shown are the first of them"
        if len(real) > 1 and choice is None:
            return {"ok": True, "choice": True, "options": [{"exact": str(x), "decimal": sig(x)} for x in real], "trace": [steps], "periodic": periodic, "message": f"{len(real)} real answers; pick the one that fits your case"}
        val = real[choice or 0]
        res = (eq.lhs - eq.rhs).subs({**sub, t: val}); resn = sp.N(res, 60); mag = max(abs(sp.N(eq.lhs.subs({**sub, t: val}), 60)), abs(sp.N(eq.rhs.subs({**sub, t: val}), 60)), sp.Float("1e-300", 60))
        rel = abs(resn) / mag; ok = rel < sp.Float("1e-45", 60)
        steps.update({"exact": str(val), "decimal": sig(val), "check": {"put_back": f"left side minus right side = {sig(resn, 6) if resn != 0 else '0'}", "relative_error": "0" if resn == 0 else f"{float(rel):.1e}", "passed": bool(ok)}})
        if dropped: steps["note"] = f"{dropped} complex answer(s) left out"
        return {"ok": bool(ok), "value": {"exact": str(val), "decimal": sig(val), "unit": self.label_of(target)}, "also_in": self.alt_units(val, target), "target": target, "trace": [steps], "value_obj": val}

    @staticmethod
    def _numeric_roots(expr, t):
        out = []
        for g in ("0.1", "0.5", "1", "2", "5", "10", "50", "100", "1000"):
            try: x = sp.nsolve(expr, t, sp.Float(g, 60), prec=60, tol=1e-50, maxsteps=100)
            except Exception: continue
            x = sp.N(x, 60)
            if abs(sp.im(x)) > sp.Float("1e-40", 60) or sp.re(x) <= 0 or any(abs(sp.re(x) - y) < sp.Float("1e-30", 60) for y in out): continue
            out.append(sp.re(x))
        return sorted(out, key=lambda z: float(z))

    # ---------------------------------------------------------------- a route through several formulas
    def route(self, known, target, force=None):
        """which formulas, in which order. known is a set of variable names. returns (steps, None) or (None, closest); step = (formula, variable it gives)"""
        force = force or {}; have = {v: (None, ()) for v in known}              # variable -> (formula that gives it, all formulas it rests on)
        if target in have: return [], None
        for _ in range(40):
            new = {}
            for n, r in self.F.items():
                unk = [v for v in r["vars"] if v not in have]
                if len(unk) != 1: continue
                v = unk[0]
                if v in force and force[v] != n: continue
                used = {n}
                for u in r["vars"]:
                    if u != v: used.update(have[u][1])
                if v not in new or len(used) < len(new[v][1]): new[v] = (n, tuple(sorted(used)))
            if not new: break
            have.update(new)
            if target in have: break
        if target not in have: return None, self.closest(known, target, have)
        order, seen = [], set()

        def need(v):                                                          # walk back from the target, inputs first
            if v in known or v in seen: return
            seen.add(v); n = have[v][0]
            for u in self.F[n]["vars"]:
                if u != v: need(u)
            order.append((n, v))
        need(target)
        return order, None

    def closest(self, known, target, have):
        reach = set(have) | set(known); out = []
        for n in self.by_var.get(target, []):
            miss = [v for v in self.F[n]["vars"] if v != target and v not in reach]
            out.append({"formula": n, "domain": self.F[n]["domain"], "equation": self.F[n]["text"], "also_need": miss})
        out.sort(key=lambda x: len(x["also_need"])); return out[:5]

    def solve_chain(self, vals, target, force=None, choices=None):
        t0 = time.perf_counter(); choices = choices or {}
        given = {v: x for v, x in vals.items() if str(x).strip() != ""}
        if target in given: return {"ok": True, "value": {"exact": str(given[target]), "decimal": str(given[target])}, "target": target, "trace": [], "note": "you already gave this value"}
        order, close = self.route(set(given), target, force)
        if order is None: return {"ok": False, "error": f"cannot reach {target} from what is given", "closest": close}
        used = {n for n, _ in order}; ex = {}
        try:
            for v, x in given.items(): ex[v], _ = self.read_value(x, v, next((n for n in used if n in UNRESOLVED), None))
        except ValueError as e: return {"ok": False, "error": str(e)}
        known = {v: str(x) for v, x in ex.items()}; trace = []
        for n, v in order:
            r = self.F[n]; got = self.solve_one(n, known, v, choice=choices.get(v))
            if not got.get("ok") and not got.get("choice"): got["failed_at"] = {"formula": n, "wanted": v}; got["trace"] = trace + got.get("trace", []); return got
            if got.get("choice"): got["trace"] = trace + got["trace"]; got["choose_for"] = v; got["failed_at"] = None; return got
            known[v] = str(got["value_obj"]); trace += got["trace"]
        alt = [self.card(n) | {"gives": target} for n in self.by_var.get(target, []) if n != order[-1][0]][:6]
        return {"ok": all(s["check"]["passed"] for s in trace), "value": trace[-1] and {"exact": trace[-1]["exact"], "decimal": trace[-1]["decimal"], "unit": self.label_of(target)}, "also_in": self.alt_units(sp.sympify(trace[-1]["exact"]), target), "target": target, "trace": trace, "formulas_used": [n for n, _ in order],
                "crosses_areas": len({s["domain"] for s in trace}) > 1, "other_formulas_with_this_variable": alt, "ms": round((time.perf_counter() - t0) * 1000, 1)}


if __name__ == "__main__":
    m = Math()
    if len(sys.argv) > 1 and sys.argv[1] == "learn": print(json.dumps(m.learn(), indent=1))
    else: print(m.link_status(), m.meta)
