"""Exponent/angle QKV router (prototype).   python qkv_router.py "<question>"   |   python qkv_router.py --eval
Key   = every capability (333 formulas by plain name + aliases + variable names, 3 guided jobs, Engg, Lang) written as a sum of word angles (the meaning table, 64 angles per word).
Query = the same sum over the words typed. Score = how well the two angle sums line up (cos of the angle differences). Value = the function bound to the best key.
Only adding and comparing angles: no matrices, no tokenizer, no learned weights. Offline."""
import json, math, os, re, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meaning import Meaning
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = re.compile(r"[a-z]+")
STOP = set("the a an of to is are what how do does i my me we you it in on for and or with from this that be can if at as by".split())
JOBS = {
    "job:invoice": "service invoice bill quote quotation customer charge labour hours materials systems installed price tax discount total payment",
    "job:warehouse": "warehouse design layout racking pallets storage floor area aisles height levels building capacity positions",
    "job:hvac_size": "hvac size sizing tonnage tons cfm btu airflow cooling load air conditioner capacity system choose",
    "engg": "engineering shape structure measure point cloud scan plane cylinder pipe tank dimensions size of part from photo",
    "lang": "hello hi thanks help repair fix broken not working problem why noise leak smell noisy frozen ice tripping breaker thermostat filter dirty maintenance explain what is meaning",
}


class Router:
    def __init__(self):
        self.m = Meaning(); keys = {}
        for l in open(os.path.join(ROOT, "model", "math_solo", "labels_draft.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            if not r.get("ok"): continue
            keys["formula:" + r["name"]] = " ".join([r.get("title", ""), " ".join(r.get("aliases", [])), " ".join(r.get("vars", {}).values()), r["name"].replace("_", " ")])
        keys.update(JOBS); self.names = list(keys); toks = {n: self.words(t) for n, t in keys.items()}
        df = {}
        for n, ws in toks.items():
            for w in set(ws): df[w] = df.get(w, 0) + 1
        N = len(keys); self.idf = {w: math.log(1 + N / c) for w, c in df.items()}
        self.K = np.stack([self.vec(ws, True) for ws in toks.values()])

    def words(self, t): return [w for w in W.findall(t.lower()) if w not in STOP and self.m.has(w)]

    def vec(self, ws, fixed=False):
        v = np.zeros(2 * self.m.n, np.float32)
        for w in ws:
            i = self.m.ix[w]; wt = self.idf.get(w, math.log(1 + 400)) if not fixed else self.idf.get(w, 1.0); v += wt * np.concatenate([self.m.cos[i], self.m.sin[i]])
        n = np.linalg.norm(v); return v / n if n else v

    def route(self, q, k=3):
        ws = self.words(q)
        if not ws: return [("lang", 0.0)]
        s = self.K @ self.vec(ws); o = np.argsort(-s)[:k]; return [(self.names[i], round(float(s[i]), 3)) for i in o]


# ---- test set: question -> accepted answers
T = [("I want a service invoice for the AC installs I did", "job:invoice"), ("make a bill for the customer with labour and materials", "job:invoice"), ("quote for installing two heat pumps", "job:invoice"),
     ("how much do I charge for 3 systems and 40 hours", "job:invoice"), ("invoice total with tax and discount", "job:invoice"), ("prepare a quotation for a duct job", "job:invoice"),
     ("design a warehouse", "job:warehouse"), ("how many pallets fit in a 50 by 30 building", "job:warehouse"), ("racking layout for a storage hall", "job:warehouse"), ("plan aisles and levels for my depot", "job:warehouse"),
     ("storage capacity of my warehouse floor", "job:warehouse"), ("how many tons of cooling do I need for 36000 btu", "job:hvac_size"), ("size an air conditioner", "job:hvac_size"), ("what airflow in cfm for this load", "job:hvac_size"),
     ("choose the right capacity for a split system", "job:hvac_size"), ("convert btu to tons", "job:hvac_size"), ("measure the cylinder in this point cloud", "engg"), ("find planes and pipes in my scan", "engg"),
     ("dimensions of a tank from a photo", "engg"), ("what shapes are in this structure", "engg"),
     ("current through a resistor", "formula:ohms_law"), ("voltage across a resistor given current", "formula:ohms_law"), ("electrical power from voltage and current", "formula:power_electrical"),
     ("volume of a cylinder", "formula:cylinder_volume"), ("mass from volume and density", "formula:mass_from_volume"), ("coefficient of performance of a heat pump", "formula:heat_pump_cop"),
     ("ideal gas pressure and temperature", "formula:ideal_gas_law"), ("sensible heat transfer rate of air", "formula:sensible_heat_rate"), ("hoop stress in a thin pipe", "formula:thin_cylinder_hoop_stress"),
     ("pressure at depth in water", "formula:hydrostatic_pressure"), ("how much energy is stored in a battery in watt hours", "formula:battery_energy_wh"), ("three phase real power", "formula:three_phase_power"),
     ("heat exchanger effectiveness", "formula:heat_exchanger_effectiveness"), ("cutoff frequency of an rc filter", "formula:rc_cutoff_frequency"), ("natural frequency from stiffness and mass", "formula:natural_frequency"),
     ("power factor correction capacitor", "formula:power_factor_correction_capacitor"), ("wavelength from frequency", "formula:wavelength"), ("buoyancy force on a submerged volume", "formula:buoyancy_force"),
     ("hello there", "lang"), ("why is my air conditioner not cooling", "lang"), ("my furnace keeps tripping the breaker", "lang"), ("thanks a lot", "lang"), ("the outdoor unit is making a loud noise", "lang"),
     ("my evaporator coil is frozen with ice", "lang"), ("what is a thermostat", "lang"), ("how often should I change the filter", "lang"), ("there is a bad smell from the vents", "lang")]


def keyword_baseline(q):
    import frostmath
    global _M
    try: _M
    except NameError: _M = frostmath.Math()
    low = q.lower()
    for name, ws in (("job:invoice", ("invoice", "bill", "quote", "quotation")), ("job:warehouse", ("warehouse", "racking", "storage space", "pallet")), ("job:hvac_size", ("size", "sizing", "tonnage", "cfm", "btu", "airflow", "tons"))):
        if any(w in low for w in ws): return name
    r = _M.find(q, 1)
    return "formula:" + r[0]["name"] if r else "lang"


def evaluate():
    R = Router(); ok = {"qkv": 0, "base": 0}; by = {}; rows = []
    for q, truth in T:
        top = R.route(q, 1)[0]; got = top[0]; b = keyword_baseline(q); grp = truth.split(":")[0]
        a = by.setdefault(grp, [0, 0, 0]); a[0] += 1; ok["qkv"] += got == truth; ok["base"] += b == truth; a[1] += got == truth; a[2] += b == truth
        if got != truth: rows.append((q, truth, got, top[1]))
    n = len(T); print(f"questions {n}   angle-QKV {ok['qkv']}/{n} = {100 * ok['qkv'] / n:.0f}%   old keyword+find {ok['base']}/{n} = {100 * ok['base'] / n:.0f}%")
    for g, (c, a, b) in by.items(): print(f"  {g:8s} n={c:2d}  angle-QKV {a:2d}   old {b:2d}")
    print("angle-QKV misses:"); [print("  ", r) for r in rows]


if __name__ == "__main__":
    if "--eval" in sys.argv: evaluate()
    else: print(Router().route(" ".join(sys.argv[1:]), 3))
