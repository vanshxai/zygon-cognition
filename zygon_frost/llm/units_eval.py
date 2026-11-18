"""Frost Math units test.  python units_eval.py [seed]  ->  model/units_eval.json and one row in model/math_solo/runs.jsonl
A  every unit reads its own name, and a value goes to SI and back to exactly the same number (fractions, no rounding)
B  reference conversions checked against exact definitions (NIST SP 811 / the SI definitions)
C  second opinion on the inferred units: for formulas that are plain products and powers, sympy's own dimension system must agree with Frost Math's
D  the hide-one-value test again, but every given value is typed in a random unit that fits (ft, psi, BTU/hr, gpm ...) and the answer must still match to 40 digits
E  a plain number into a box that takes rpm / mm / degrees / Celsius is treated as that unit; a wrong kind of unit is refused"""
import json, os, random, sys, time
import sympy as sp
import units as U
import frostmath as fm
sys.stdout.reconfigure(encoding="utf-8")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3; random.seed(SEED)
M = fm.Math(); F = M.F; out = {"seed": SEED}
close = lambda a, b: abs(sp.N(a - b, 60)) <= sp.Float("1e-40", 60) * max(1, abs(sp.N(b, 60)))

# ---------------------------------------------------------------- A
t0 = time.perf_counter(); bad = []; n = 0
for name, (fac, dm, lab, off) in U.U.items():
    ue = U.unit_expr(name)
    if not ue: bad.append((name, "does not read its own name")); continue
    for x in (sp.Rational(1), sp.Rational(7, 3), sp.Rational(-5, 2), sp.Rational(123456, 7)):
        if off == 0 and x < 0 and name not in ("m", "ft"): pass
        try:
            si, _ = U.to_si(f"{x} {name}", dm, None) if off == 0 else (x * fac + off, None)
            back, _ = U.from_si(si, name)
        except Exception as e: bad.append((name, str(e))); continue
        n += 1
        if sp.simplify(back - x) != 0: bad.append((name, f"{x} -> {si} -> {back}"))
out["A"] = {"units": len(U.U), "round_trips": n, "failures": len(bad), "detail": bad[:10], "seconds": round(time.perf_counter() - t0, 1)}; print("A", out["A"], flush=True)

# ---------------------------------------------------------------- B
REF = [("1 ft", "3048/10000", "m"), ("1 in", "254/10000", "m"), ("1 mi", "1609344/1000", "m"), ("1 lb", "45359237/100000000", "kg"), ("1 gal", "3785411784/1000000000000", "m3"),
       ("1 cfm", "28316846592/60000000000000", "m3/s"), ("1 BTU", "105505585262/100000000", "J"), ("1 kWh", "3600000", "J"), ("1 hp", "74569987158227022/100000000000000", "W"),
       ("1 psi", "6894757293168/1000000000", "Pa"), ("1 atm", "101325", "Pa"), ("1 bar", "100000", "Pa"), ("1 kcal", "4184", "J"), ("1 lbf", "44482216152605/10000000000000", "N"),
       ("1 rpm", "1/60", "Hz"), ("1 mph", "44704/100000", "m/s"), ("1 inwg", "24908891/100000", "Pa"), ("1 ton", "12000*105505585262/360000000000", "W"), ("1 BTU/hr", "105505585262/360000000000", "W"),
       ("32 F", "27315/100", "K"), ("212 F", "37315/100", "K"), ("0 C", "27315/100", "K"), ("100 C", "37315/100", "K"), ("1 gpm", "3785411784/60000000000000", "m3/s"), ("180 deg", "pi", "rad")]
badb = []
for t, exp, _ in REF:
    v, _ = U.to_si(t, None, None)
    if sp.simplify(v - sp.sympify(exp, rational=True)) != 0: badb.append((t, str(v), exp))
kw = sp.N(1000 / U.U["BTU/hr"][0], 12); tons = sp.N(U.U["ton"][0] / 1000, 12)
out["B"] = {"checked": len(REF), "failures": len(badb), "detail": badb, "1 kW in BTU/hr": str(kw), "1 ton in kW": str(tons)}; print("B", out["B"], flush=True)

# ---------------------------------------------------------------- C
base = {k: sp.Symbol("dim_" + k, positive=True) for k in U.DIMS}
def dexpr(dm):
    r = sp.Integer(1)
    for k, e in zip(U.DIMS, dm): r *= base[k] ** sp.Rational(e.numerator, e.denominator)
    return r
checked = agree = 0; disagree = []; skipped = 0
for name, r in F.items():
    eq = r["eq"]
    if name in fm.UNRESOLVED or not all(v in M.udim for v in r["vars"]): skipped += 1; continue
    if any(isinstance(a, (sp.Add, sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.Abs, sp.Max, sp.Min)) or (a.is_Pow and not a.exp.is_Number) for a in sp.preorder_traversal(eq.rhs)): skipped += 1; continue
    sub = {sp.Symbol(v): dexpr(M.udim[v]) for v in r["vars"]}
    ratio = sp.simplify(eq.lhs.subs(sub, simultaneous=True) / eq.rhs.subs(sub, simultaneous=True).subs({sp.pi: 1})); ratio = ratio.as_independent(*base.values())[1]
    checked += 1
    if ratio == 1: agree += 1
    else: disagree.append((name, str(ratio)))
out["C"] = {"formulas_checked": checked, "agree": agree, "disagree": disagree[:10], "skipped (sums, functions, unknown units)": skipped}; print("C", out["C"], flush=True)

# ---------------------------------------------------------------- D
NICE = [sp.Rational(a, b) for a in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25) for b in (1, 2, 4, 5)]
def native_to_si(v, x):
    if v in fm.NATIVE:
        fac, dm, off, lab = U.unit_expr(fm.NATIVE[v]); return x * fac + (off if (U.same_dims(dm, U.TEMP) and off != 0) else 0)
    return x
def si_to_native(v, si):
    if v in fm.NATIVE:
        fac, dm, off, lab = U.unit_expr(fm.NATIVE[v]); return (si - (off if (U.same_dims(dm, U.TEMP) and off != 0) else 0)) / fac
    return si
def typed(v, x):
    """x (in the unit the formula uses) written the way a person would: in some other fitting unit, exact"""
    dm = M.udim[v]; si = native_to_si(v, x)
    cands = [n for n in U.compatible(dm, 30) if U.U[n][3] == 0 and U.U[n][0].is_Rational and n not in ("ratio",)] if dm != U.Z else []
    if not cands or random.random() < 0.15: return str(x)
    n = random.choice(cands); y = si / U.U[n][0]
    if not y.is_Rational: return str(x)
    return f"{y} {n}"
def truth_for(name):
    r = F[name]; eq = r["eq"]; outv = str(eq.lhs)
    for _ in range(25):
        a = {v: random.choice(NICE) for v in r["vars"] if v != outv}
        try: t = eq.rhs.subs({sp.Symbol(k): x for k, x in a.items()})
        except Exception: continue
        nn = sp.N(t, 50)
        if t.has(sp.zoo, sp.nan) or not t.is_number or abs(sp.im(nn)) > 1e-30 or nn == 0 or abs(sp.re(nn)) > 1e12: continue
        a[outv] = t; return a
    return None
t0 = time.perf_counter(); D = {"formulas": 0, "cases": 0, "right": 0, "wrong": 0, "refused": 0, "multi_root_or_repeat": 0}; wrong = []
for name, r in F.items():
    if name in fm.UNRESOLVED or not all(v in M.udim for v in r["vars"]): continue
    tr = truth_for(name)
    if tr is None: continue
    D["formulas"] += 1
    for v in r["vars"]:
        vals = {k: typed(k, x) for k, x in tr.items() if k != v}; D["cases"] += 1
        res = M.solve_one(name, vals, v)
        if res.get("choice"):
            if any(close(sp.sympify(o["exact"]), tr[v]) for o in res["options"]): D["right"] += 1
            else: D["multi_root_or_repeat"] += 1
        elif res.get("ok"):
            if close(res["value_obj"], tr[v]): D["right"] += 1
            else: D["wrong"] += 1; wrong.append((name, v, vals, str(res["value_obj"]), str(tr[v])))
        else: D["refused"] += 1; wrong.append((name, v, vals, res.get("error"), ""))
D["seconds"] = round(time.perf_counter() - t0, 1); D["detail"] = wrong[:12]; out["D"] = D; print("D", {k: v for k, v in D.items() if k != "detail"}, flush=True)
for w in wrong[:8]: print("   ", w)

# ---------------------------------------------------------------- E
E = {"checks": 0, "failures": []}
def chk(label, cond):
    E["checks"] += 1
    if not cond: E["failures"].append(label)
v, _ = M.read_value("1800", "N_pump2"); chk("plain number into rpm box stays 1800", v == 1800)
v, _ = M.read_value("30 Hz", "N_pump2"); chk("30 Hz into an rpm box becomes 1800", v == 1800)
v, _ = M.read_value("212 F", "T_batt_amb"); chk("212 F into a Celsius box is 100", v == 100)
v, _ = M.read_value("100 C", "T_amb"); chk("100 C into a Kelvin box is 373.15", v == sp.Rational(37315, 100))
v, _ = M.read_value("10 dF", "dT_hvac"); chk("a 10 F difference is 50/9 K", v == sp.Rational(50, 9))
v, _ = M.read_value("1 inwg", "P_hyd"); chk("1 inwg is 249.08891 Pa", v == sp.Rational(24908891, 100000))
for bad_in, var in (("5 ft", "V"), ("100 psi", "I"), ("3 gpm", "T_amb")):
    try: M.read_value(bad_in, var); chk(f"{bad_in} into {var} must be refused", False)
    except ValueError: chk("", True)
try: M.read_value("5 furlongs", "L"); chk("unknown unit refused", False)
except ValueError: chk("", True)
E["failures"] = [f for f in E["failures"] if f]; out["E"] = E; print("E", E, flush=True)

json.dump(out, open(os.path.join(fm.ROOT, "model", "units_eval.json"), "w", encoding="utf-8"), indent=1, default=str)
inf = json.load(open(os.path.join(fm.STORE, "units.json")))
M.log("units", {"seed": SEED, "units_in_table": len(U.U), "variables": len(M.F) and len(M.by_var), "variables_with_unit": len(M.udim), "formulas_that_do_not_balance": inf["does_not_balance"], "A_failures": out["A"]["failures"], "B_failures": out["B"]["failures"],
                "C_checked": out["C"]["formulas_checked"], "C_agree": out["C"]["agree"], "D": {k: v for k, v in D.items() if k != "detail"}, "E_checks": E["checks"], "E_failures": len(E["failures"])})
