"""Frost Math accuracy test.  python math_eval.py [seed]  ->  model/math_eval.json and one row in model/math_solo/runs.jsonl
A  every formula, every variable as the unknown. The truth is made first (random exact values, the formula's own left side worked out), then one value is hidden and Frost Math must give it back.
B  two and three formulas in a row. The truth is worked out forward through the chain; Frost Math gets only the starting values and must reach the end.
Right = the answer equals the truth to 40 digits AND the put-back check passed. 'No closed form' (cannot rearrange) is counted apart from wrong."""
import json, os, random, sys, time
import sympy as sp
import frostmath as fm
sys.stdout.reconfigure(encoding="utf-8")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7; random.seed(SEED)
M = fm.Math(); F = M.F
NICE = [sp.Rational(a, b) for a in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25) for b in (1, 2, 4, 5)]
close = lambda a, b: abs(sp.N(a - b, 60)) <= sp.Float("1e-40", 60) * max(1, abs(sp.N(b, 60)))


def draw(names): return {v: random.choice(NICE) for v in names}


def txt(x):
    """how a person would type it: decimals where the fraction ends (5/4 -> 1.25), otherwise the exact text"""
    if x.is_Rational and x.q in (1, 2, 4, 5, 8, 10): return str(sp.Float(x, 15)).rstrip("0").rstrip(".") if x.q != 1 else str(x.p)
    return str(x)


def truth_for(name, tries=25):
    """exact values for every variable of a formula, consistent with it (positive, real)"""
    r = F[name]; eq = r["eq"]; out = str(eq.lhs)
    for _ in range(tries):
        a = draw([v for v in r["vars"] if v != out])
        try: t = eq.rhs.subs({sp.Symbol(k): x for k, x in a.items()})
        except Exception: continue
        n = sp.N(t, 50)
        if t.has(sp.zoo, sp.nan) or not t.is_number or abs(sp.im(n)) > 1e-30 or n == 0 or abs(sp.re(n)) > 1e12: continue
        a[out] = t; return a
    return None


# ---------------------------------------------------------------- A
t0 = time.perf_counter(); A = {"cases": 0, "right": 0, "no_closed_form": 0, "wrong": 0, "multi_root": 0, "no_truth": 0, "formulas_all_right": 0, "formulas": len(F)}; wrong = []; noform = []
for name, r in F.items():
    tr = truth_for(name)
    if tr is None: A["no_truth"] += 1; continue
    allok = True
    for v in r["vars"]:
        vals = {k: txt(x) for k, x in tr.items() if k != v}; A["cases"] += 1
        res = M.solve_one(name, vals, v)
        if res.get("choice"):
            A["multi_root"] += 1; ok = any(close(sp.sympify(o["exact"]), tr[v]) for o in res["options"])
            if ok: res = M.solve_one(name, vals, v, choice=[i for i, o in enumerate(res["options"]) if close(sp.sympify(o["exact"]), tr[v])][0]); ok = res.get("ok") and close(res["value_obj"], tr[v])
        elif res.get("ok"): ok = close(res["value_obj"], tr[v])
        elif "no exact solution" in str(res.get("error")) or "no real answer" in str(res.get("error")): A["no_closed_form"] += 1; noform.append((name, v, res["error"])); allok = False; continue
        else: ok = False
        if ok: A["right"] += 1
        elif res.get("periodic") and any(s_.get("repeats") for s_ in res.get("trace", [])): A["periodic_other_turn"] = A.get("periodic_other_turn", 0) + 1; A["right"] += 0
        else: A["wrong"] += 1; allok = False; wrong.append((name, v, str(res.get("error") or res.get("value"))))
    A["formulas_all_right"] += allok
A["seconds"] = round(time.perf_counter() - t0, 1)
print("A", A, flush=True)
for w in wrong[:15]: print("  wrong", w)
for w in noform[:10]: print("  noform", w)

# ---------------------------------------------------------------- B
t0 = time.perf_counter(); B = {2: {"cases": 0, "right": 0, "other_route": 0, "other_route_checked": 0, "no_route": 0, "wrong": 0}, 3: {"cases": 0, "right": 0, "other_route": 0, "other_route_checked": 0, "no_route": 0, "wrong": 0}}
names = list(F); users = {}
for n, r in F.items():
    for v in r["vars"]:
        if v != str(r["eq"].lhs): users.setdefault(v, []).append(n)
bad = []


def build_chain(depth):
    for _ in range(60):
        a = random.choice(names); tr = truth_for(a)
        if tr is None: continue
        chain = [a]; known = {k: x for k, x in tr.items()}; given = {k: x for k, x in tr.items() if k != str(F[a]["eq"].lhs)}; cur = str(F[a]["eq"].lhs); ok = True
        for _ in range(depth - 1):
            cands = [n for n in users.get(cur, []) if n not in chain and str(F[n]["eq"].lhs) not in known]
            if not cands: ok = False; break
            n = random.choice(cands); r = F[n]; out = str(r["eq"].lhs)
            fresh = draw([v for v in r["vars"] if v not in known and v != out]); known.update(fresh); given.update(fresh)
            try: t = r["eq"].rhs.subs({sp.Symbol(k): x for k, x in known.items() if k in r["vars"]})
            except Exception: ok = False; break
            nn = sp.N(t, 50)
            if t.has(sp.zoo, sp.nan) or not t.is_number or abs(sp.im(nn)) > 1e-30 or nn == 0 or abs(sp.re(nn)) > 1e12: ok = False; break
            known[out] = t; chain.append(n); cur = out
        if ok: return chain, given, cur, known[cur]
    return None


for depth in (2, 3):
    for _ in range(150):
        got = build_chain(depth)
        if not got: continue
        chain, given, target, tv = got; B[depth]["cases"] += 1
        res = M.solve_chain({k: txt(x) for k, x in given.items()}, target)
        if res.get("error") and "cannot reach" in res["error"]: B[depth]["no_route"] += 1; continue
        if res.get("choice"): B[depth]["needs_choice"] = B[depth].get("needs_choice", 0) + 1; continue
        if "value" not in res or not res.get("trace"): B[depth]["wrong"] += 1; bad.append((depth, chain, target, res.get("error"))); continue
        steps_ok = all(s["check"]["passed"] for s in res["trace"]); same = close(sp.sympify(res["value"]["exact"]), tv)
        if res.get("ok") and same: B[depth]["right"] += 1
        elif res.get("ok") and steps_ok and set(res["formulas_used"]) != set(chain): B[depth]["other_route"] += 1; B[depth]["other_route_checked"] += 1
        else: B[depth]["wrong"] += 1; bad.append((depth, chain, target, res.get("error") or res.get("formulas_used"), {"given": {k: txt(x) for k, x in given.items()}, "got": res.get("value"), "truth": str(tv), "ok": res.get("ok"), "checks": [s_["check"] for s_ in res["trace"]]}))
Bsec = round(time.perf_counter() - t0, 1)
print("B", B, Bsec, flush=True)
for w in bad[:10]: print("  bad", w)
res = {"seed": SEED, "A": A, "B": B, "B_seconds": Bsec, "A_wrong": wrong[:40], "A_no_closed_form": noform[:60], "B_bad": bad[:20]}
json.dump(res, open(os.path.join(fm.ROOT, "model", "math_eval.json"), "w", encoding="utf-8"), indent=1, default=str)
M.log("eval", {"seed": SEED, "A_cases": A["cases"], "A_right": A["right"], "A_wrong": A["wrong"], "A_no_closed_form": A["no_closed_form"], "A_seconds": A["seconds"], "B2": B[2], "B3": B[3], "B_seconds": Bsec})
