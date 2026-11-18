"""Does the meaning table make Frost understand better?   python meaning_eval.py  ->  model/meaning_eval.json and one row in model/lang_solo/runs.jsonl + model/math_solo/runs.jsonl
Each test is run with the meaning table off and on. Settings are chosen on half of the questions (dev) and the result is reported on the other half (test), so nothing is tuned on what it is scored on.
T1  Frost Math: find the right formula from a plain-words question that does not use the formula's own name
T2  Frost Lang: read a different wording of a small-talk or grammar question as the right lesson, and keep quiet on engineering questions
T3  Frost Lang: find the right stored sentence in a small HVAC note when the question is worded differently"""
import itertools, json, os, random, shutil, sys, tempfile, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frostmath as fm, lang
sys.stdout.reconfigure(encoding="utf-8")
ROOT = fm.ROOT
T1 = {
 "ohms_law": ["how much current flows through a resistor at a given voltage", "relationship between volts, amps and ohms"],
 "darcy_weisbach_head_loss": ["friction loss in a long pipe", "how much pressure is lost along a run of piping"],
 "reynolds_number": ["is the flow turbulent or smooth", "dimensionless number that tells the flow regime"],
 "sensible_cooling_load": ["how much heat must the air conditioner take out of the room air", "cooling needed to lower the air temperature"],
 "humidity_ratio": ["how much moisture is in the air", "water vapour carried per kilogram of dry air"],
 "heat_pump_cop": ["how efficient is a heat pump", "heat delivered per unit of electricity for heating"],
 "carnot_cop_refrigeration": ["best possible efficiency of a refrigerator", "ideal cooling performance between two temperatures"],
 "hydrostatic_pressure": ["pressure at the bottom of a tank of liquid", "pressure under a column of water"],
 "hydraulic_power": ["power carried by oil flowing under pressure", "how much power does the hydraulic fluid deliver"],
 "torricellis_law": ["speed of water leaving a hole in a tank", "how fast does liquid squirt out of a small opening in the side"],
 "water_hammer_pressure": ["pressure spike when a valve slams shut", "shock wave in a pipe after sudden closing"],
 "orifice_flow": ["flow through a small opening", "how much fluid passes through a hole in a plate"],
 "valve_flow_coefficient": ["how big should the valve be for a given flow", "valve capacity rating"],
 "three_phase_power": ["power drawn by a three phase supply", "how many kilowatts does a 3 phase load use"],
 "power_factor_correction_capacitor": ["capacitor needed to improve the power factor", "how big a capacitor bank to fix a poor power factor"],
 "cable_voltage_drop": ["volts lost along a long wire", "how much voltage disappears over a long cable run"],
 "cable_ampacity": ["how much current can a wire carry safely", "maximum safe current for a cable"],
 "fault_current": ["short circuit current", "how big is the current when a short happens"],
 "beam_deflection_center_load": ["how much does a beam sag with a weight in the middle", "bending of a beam under a point load at the centre"],
 "udl_max_moment": ["bending moment of a beam carrying an evenly spread load", "largest moment from a uniformly distributed load"],
 "bending_stress": ["stress in a beam when it bends", "how much stress does bending put on a member"],
 "euler_buckling_load": ["load at which a long thin column collapses sideways", "when does a slender strut buckle"],
 "thin_cylinder_hoop_stress": ["stress in the wall of a pressurised pipe", "circumferential stress in a thin walled tube under pressure"],
 "thermal_expansion": ["how much does a rod grow when heated", "length change from a temperature rise"],
 "fouriers_law_conduction": ["heat flow through a wall by conduction", "rate of heat transfer through a solid layer"],
 "convection_heat_transfer": ["heat lost from a surface to moving air", "heat exchange between a surface and a flowing fluid"],
 "stefan_boltzmann_radiation": ["heat radiated by a hot surface", "thermal radiation from a glowing body"],
 "log_mean_temp_difference": ["average temperature difference across a heat exchanger", "effective temperature driving force in an exchanger"],
 "heat_exchanger_effectiveness": ["how well does the heat exchanger perform", "fraction of the best possible heat transfer achieved"],
 "gear_ratio": ["speed reduction between two gears", "how many turns of the input per turn of the output"],
 "belt_power_transmitted": ["power carried by a drive belt", "how much power can a belt drive transfer"],
 "torque": ["twisting force from a motor current", "turning effort produced for a given amperage"],
 "power_mechanical": ["shaft power from torque and speed", "mechanical output of a rotating machine"],
 "rpm_conversion": ["convert radians per second to revolutions per minute", "turn angular speed into rpm"],
 "synchronous_speed": ["speed of the rotating field of an ac motor", "how fast does the magnetic field of an induction motor turn"],
 "spring_rate": ["stiffness of a coil spring", "force needed per unit of compression of a spring"],
 "bearing_life_l10": ["how long will the bearing last", "expected life of a rolling bearing"],
 "buck_converter_ratio": ["step down dc voltage regulator output", "output of a switching regulator that lowers the voltage"],
 "voltage_divider": ["two resistors splitting a voltage", "how to scale a voltage down with resistors"],
 "rc_cutoff_frequency": ["frequency where a resistor capacitor filter starts cutting", "corner frequency of a simple low pass filter"],
 "kinetic_energy": ["energy stored in a spinning flywheel", "rotational energy of a turning mass"],
 "impeller_tip_speed": ["rim speed of a pump wheel", "how fast the outer edge of the impeller moves"],
 "affinity_law_flow": ["how does flow change when the pump speed changes", "pump output versus rpm"],
 "npsh_available": ["will the pump cavitate at the inlet", "suction head margin before the pump starves"],
 "cylinder_force": ["push force of a hydraulic ram", "how hard can a hydraulic cylinder push"],
 "continuity_equation": ["flow speed when the pipe gets narrower", "velocity change as the duct area shrinks"],
 "coefficient_of_performance": ["cooling effect per unit of power input", "how much cooling for each watt used"],
 "minor_head_loss": ["pressure drop through fittings and bends", "loss across elbows and valves in a line"],
 "pump_flow_rate": ["how much liquid does the pump deliver", "volume per second moved by a pump"],
 "shear_stress": ["stress from a force sliding the surfaces across each other", "cutting stress in a bolt"],
 "hookes_law": ["extension proportional to force", "how much a spring stretches under load"],
}
T2 = [("hey buddy", "greeting"), ("salutations", "greeting"), ("how is it going", "how are you"), ("how have you been", "how are you"), ("i feel unhappy", "user sad"), ("i am feeling down", "user sad"),
      ("i am exhausted", "user tired"), ("i feel sleepy", "user tired"), ("i am delighted", "user happy"), ("i am so glad", "user happy"), ("much obliged", "thanks"), ("i appreciate it", "thanks"),
      ("my apologies", "sorry"), ("i apologise", "sorry"), ("see you later", "farewell"), ("catch you tomorrow", "farewell"), ("sleep well", "good night"), ("tell me about yourself", "who are you"),
      ("what should i call you", "who are you"), ("who built you", "who made you"), ("who is your creator", "who made you"), ("what are your abilities", "what can you do"), ("what are you able to do", "what can you do"),
      ("i need assistance", "help"), ("can you assist me", "help"), ("are you able to learn", "can you learn"), ("can you pick up new things", "can you learn"), ("what is a noun", "grammar noun"),
      ("explain plurals", "grammar plural"), ("what is an adjective", "grammar adjective"), ("describe the adverb", "grammar adverb"), ("which words are pronouns", "grammar pronoun"),
      ("what does hello mean", "meaning hello"), ("what does goodbye mean", "meaning goodbye"), ("what is the opposite of a word called", "grammar antonym"), ("what is a synonym", "grammar synonym")]
OFF = ["design a hvac system", "calculate the duct size", "size an air conditioner for my house", "what is the sensible heat formula", "pipe sizing for a boiler", "estimate concrete for a slab",
       "how much does a tonne of steel cost", "wiring a three phase motor", "explain the darcy equation", "how do i bend copper tubing", "what is a heat pump", "why is my compressor short cycling",
       "how many bricks for a wall", "what size breaker for a 5 kw heater", "airflow through a rectangular duct", "pressure drop in a pipe elbow"]
SENT = [("A ton of refrigeration equals 12,000 BTU per hour.", "12,000 BTU", ["tonnage conversion to btu per hour", "how many btu are in one ton of air conditioning"]),
        ("Sensible heat carried by air in BTU per hour equals 1.08 times the airflow in CFM times the temperature difference in degrees Fahrenheit.", "1.08", ["how do i work out the heat air carries from a temperature change", "formula for the warming or cooling load from airflow"]),
        ("Latent heat of air equals 0.68 times CFM times the humidity ratio difference.", "0.68", ["how much moisture energy does air carry", "calculate the latent load from airflow and humidity"]),
        ("Total heat of air equals 4.5 times CFM times the enthalpy change.", "4.5", ["total energy exchange of air from enthalpy", "overall capacity of air from airflow and enthalpy change"]),
        ("Heat carried by water in BTU per hour equals 500 times the flow in GPM times the temperature difference.", "500", ["heat moved by water from flow and temperature rise", "how to compute hydronic heat transfer with gpm"]),
        ("Airflow in CFM equals duct velocity in feet per minute times the duct area in square feet.", "duct area", ["how do i get airflow from air speed and duct size", "volumetric flow from velocity and cross sectional area of the duct"]),
        ("Air changes per hour equal CFM times 60 divided by the room volume in cubic feet.", "Air changes", ["how many times is the room air replaced each hour", "air exchange rate from airflow and room size"]),
        ("Air velocity from a pitot tube equals 4005 times the square root of the velocity pressure in inches of water.", "4005", ["air speed measured with a pitot tube", "convert a velocity pressure reading to speed"]),
        ("Fan airflow is directly proportional to the fan speed.", "directly proportional", ["how does blower output change with its rpm", "fan flow versus speed"]),
        ("Fan static pressure changes with the square of the fan speed.", "square of", ["how does fan pressure change when speed goes up", "static pressure and fan speed relationship"]),
        ("Fan horsepower changes with the cube of the fan speed.", "cube of", ["how does fan power change with speed", "horsepower needed when the fan runs faster"]),
        ("A valve flow coefficient Cv equals the flow in GPM divided by the square root of the pressure drop in psi.", "flow coefficient", ["how do i size a valve from flow and pressure drop", "valve capacity coefficient"]),
        ("Superheat is the suction line temperature minus the saturation temperature.", "Superheat", ["how to find refrigerant superheat", "suction line temperature minus saturation"]),
        ("Subcooling is the saturation temperature minus the liquid line temperature.", "Subcooling", ["how to calculate subcooling", "liquid line temperature compared to saturation temperature"]),
        ("One horsepower is 746 watts.", "746", ["how many watts in a horsepower", "convert horsepower to watts"]),
        ("One kilowatt equals about 3,412 BTU per hour.", "3,412", ["how many btu per hour is a kilowatt", "convert kilowatts to btu"]),
        ("Motor speed in RPM equals 120 times the frequency in hertz divided by the number of poles.", "120 times", ["motor speed from frequency and pole count", "how do i get the rpm of an induction motor"])]


def dev_test(items): return items[0::2], items[1::2]


def acc1(m, names_q, k=5):
    top1 = top3 = 0
    for name, q in names_q:
        r = [c["name"] for c in m.find(q, k)]; top1 += bool(r) and r[0] == name; top3 += name in r[:3]
    return {"top1": top1 / len(names_q), "top3": top3 / len(names_q), "n": len(names_q)}


out = {}; t0 = time.perf_counter()
# ---------------------------------------------------------------- T1
m = fm.Math(); names = list(T1); dev_n, test_n = names[0::2], names[1::2]
mk = lambda ns: [(n, q) for n in ns for q in T1[n]]
fm.Math.MP["on"] = False; base = {"dev": acc1(m, mk(dev_n)), "test": acc1(m, mk(test_n))}; fm.Math.MP["on"] = True; grid = []
for tau, lam, mode in itertools.product((0.4, 0.5, 0.6), (0.4, 0.8, 1.5), ("unmatched", "all")):
    fm.Math.MP.update(tau=tau, lam=lam, mode=mode); grid.append(((tau, lam, mode), acc1(m, mk(dev_n))))
best = max(grid, key=lambda g: (g[1]["top3"], g[1]["top1"], -g[0][0])); fm.Math.MP.update(tau=best[0][0], lam=best[0][1], mode=best[0][2])
on = {"dev": best[1], "test": acc1(m, mk(test_n)), "chosen": {"tau": best[0][0], "lam": best[0][1], "mode": best[0][2]}}
out["T1"] = {"questions": sum(len(v) for v in T1.values()), "formulas": len(T1), "off": base, "on": on}
print("T1 Frost Math  off:", {k: (round(v["top1"], 3), round(v["top3"], 3)) for k, v in base.items()}, " on:", {k: (round(v["top1"], 3), round(v["top3"], 3)) for k, v in on.items() if k != "chosen"}, on["chosen"], flush=True)
fm.Math.MP["on"] = False

# ---------------------------------------------------------------- T2
random.seed(0); L = lang.Lang(os.path.join(ROOT, "model", "lang_solo")); known = {w for g in L.groups.values() for w in g.get("utts", [])}


def run_t2(items, off):
    ok = bad = none = 0; fp = 0
    for q, cat in items:
        r = L.lesson(q); ok += bool(r) and r["cat"] == cat; bad += bool(r) and r["cat"] != cat; none += not r
    for q in off: fp += bool(L.lesson(q))
    return {"right": ok / len(items), "wrong": bad / len(items), "no_answer": none / len(items), "n": len(items), "false_on_engineering": fp, "engineering_n": len(off)}


d2, t2 = dev_test(T2); d_off, t_off = dev_test(OFF)
L.mp["on"] = False; b2 = {"dev": run_t2(d2, d_off), "test": run_t2(t2, t_off)}; grid = []
for tau in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
    L.mp.update(on=True, tau=tau); r = run_t2(d2, d_off); grid.append((tau, r, r["right"] - 0.5 * r["wrong"] - 0.3 * r["false_on_engineering"]))
bt = max(grid, key=lambda g: g[2]); L.mp.update(on=True, tau=bt[0]); o2 = {"dev": bt[1], "test": run_t2(t2, t_off), "chosen": {"tau": bt[0]}}
out["T2"] = {"off": b2, "on": o2}; L.mp["on"] = False
f = lambda r: (round(r["right"], 3), round(r["wrong"], 3), r["false_on_engineering"])
print("T2 Frost Lang lessons (right, wrong, engineering false hits)  off:", {k: f(v) for k, v in b2.items()}, " on:", {k: f(v) for k, v in o2.items() if k != "chosen"}, o2["chosen"], flush=True)

# ---------------------------------------------------------------- T3
tmp = tempfile.mkdtemp(); shutil.copytree(os.path.join(ROOT, "model", "lang_solo"), os.path.join(tmp, "s")); H = lang.Lang(os.path.join(tmp, "s"))
for s, _, _ in SENT: H.add_sentence(s, "HVAC test note (formulas pasted by the owner from the Belimo guide, unverified)")
items3 = [(key, q) for s, key, qs in SENT for q in qs]
def run_t3(items):
    ok = wrong = ask = 0
    for key, q in items:
        H.reset_chat(); r = H.chat(q)
        if r.get("kind") == "ask": ask += 1
        elif key in (r.get("text") or ""): ok += 1
        else: wrong += 1
    return {"right": ok / len(items), "wrong": wrong / len(items), "asked_where_to_learn": ask / len(items), "n": len(items)}
d3, t3 = dev_test(items3); H.mp["on"] = False; b3 = {"dev": run_t3(d3), "test": run_t3(t3)}; grid = []
for tau in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
    H.mp.update(on=True, tau=tau); r = run_t3(d3); grid.append((tau, r, r["right"] - 0.3 * r["wrong"]))
bt = max(grid, key=lambda g: g[2]); H.mp.update(on=True, tau=bt[0]); o3 = {"dev": bt[1], "test": run_t3(t3), "chosen": {"tau": bt[0]}}
out["T3"] = {"sentences": len(SENT), "off": b3, "on": o3}; H.mp["on"] = False
g3 = lambda r: (round(r["right"], 3), round(r["wrong"], 3), round(r["asked_where_to_learn"], 3))
print("T3 Frost Lang HVAC note (right, wrong, asked)  off:", {k: g3(v) for k, v in b3.items()}, " on:", {k: g3(v) for k, v in o3.items() if k != "chosen"}, o3["chosen"], flush=True)

out["seconds"] = round(time.perf_counter() - t0, 1)
json.dump(out, open(os.path.join(ROOT, "model", "meaning_eval.json"), "w", encoding="utf-8"), indent=1)
row = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "meaning_eval", "table": json.load(open(os.path.join(ROOT, "model", "meaning", "meta.json"))), **{k: out[k] for k in ("T1", "T2", "T3", "seconds")}}
for d in ("lang_solo", "math_solo"): open(os.path.join(ROOT, "model", d, "runs.jsonl"), "a", encoding="utf-8").write(json.dumps(row) + "\n")
print("done", out["seconds"], "s")
