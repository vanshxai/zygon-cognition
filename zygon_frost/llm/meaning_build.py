"""MAKER SIDE. Builds the meaning table: every word gets 64 whole-number angles taken from the 3B model's embedding.   python meaning_build.py  ->  model/meaning/meaning.npz
The table (words + 64 angles each, micro-degrees, whole numbers) is what Frost reads. The directions the angles are measured along stay in model/meaning/maker_private.npz and never ship.
Words: the 20,000 commonest whole words of the model's vocabulary, plus every word of our own stores and an engineering word list (those average their pieces)."""
import json, os, re, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf
from tokenizer import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "model", "meaning"); os.makedirs(OUT, exist_ok=True)
NANG = int(os.environ.get("NANG", 128)); N_COMMON = int(os.environ.get("NCOMMON", 40000)); TAG = os.environ.get("TAG", "")
ACRONYMS = {                          # short technical words the model reads as junk fragments: their angles are taken from what they stand for
    "hvac": "heating ventilation air conditioning", "hvacr": "heating ventilation air conditioning refrigeration", "cfm": "cubic feet per minute airflow", "btu": "british thermal unit heat energy",
    "btuh": "british thermal unit per hour heat power", "gpm": "gallons per minute water flow", "psi": "pounds per square inch pressure", "psig": "gauge pressure pounds per square inch", "inwc": "inches water column pressure",
    "rpm": "revolutions per minute speed", "cop": "coefficient performance efficiency", "seer": "seasonal energy efficiency ratio cooling efficiency", "eer": "energy efficiency ratio cooling efficiency",
    "hspf": "heating seasonal performance factor heat pump efficiency", "ach": "air changes per hour ventilation", "vfd": "variable frequency drive motor speed controller", "ahu": "air handling unit",
    "rtu": "rooftop unit air conditioner", "vav": "variable air volume box", "erv": "energy recovery ventilator", "hrv": "heat recovery ventilator", "dx": "direct expansion refrigerant cooling",
    "txv": "thermostatic expansion valve refrigerant metering", "eev": "electronic expansion valve", "ptac": "packaged terminal air conditioner", "kw": "kilowatt power", "hp": "horsepower power",
    "awg": "american wire gauge wire size", "gfci": "ground fault circuit interrupter", "dc": "direct current electricity", "pvc": "polyvinyl chloride plastic pipe", "cpvc": "chlorinated polyvinyl chloride plastic pipe",
    "pex": "cross linked polyethylene plastic pipe", "osha": "occupational safety health administration", "nec": "national electrical code", "ashrae": "heating refrigerating air conditioning engineers society",
    "acca": "air conditioning contractors association", "superheat": "refrigerant vapor temperature above boiling point", "subcooling": "refrigerant liquid temperature below condensing point",
    "ductwork": "air duct pipes ventilation", "hydronic": "water heating cooling pipes", "psychrometric": "air moisture temperature chart", "enthalpy": "total heat energy content", "tonnage": "cooling capacity ton"}
ENG = """hvac heating ventilation ventilating cooling conditioning airconditioning aircon refrigeration refrigerant compressor condenser evaporator coil duct ductwork damper diffuser grille plenum register fan blower airflow cfm btu tonnage ton superheat subcooling
thermostat setpoint humidity dehumidifier humidifier enthalpy psychrometric chiller boiler furnace radiator heatpump insulation infiltration load residential commercial thermal valve actuator pump pipe piping tubing fitting elbow flange gasket
copper pvc steel fabrication welding weld brazing solder joint pressure flow velocity friction head vacuum leak fault diagnostics maintenance install installation commissioning balancing airflow static pitot manometer gauge sensor
voltage current resistance amp amps volt watt kilowatt horsepower motor capacitor contactor relay breaker wiring cable conduit transformer phase torque rpm bearing shaft belt pulley gear stress strain beam column load deflection
concrete rebar mortar masonry plaster drywall framing joist rafter truss roofing shingle gutter flooring tile paint carpentry cabinet excavator crane landscaping irrigation estimate quote invoice material quantity labor cost""".split()


def store_words():
    ws = set()
    for f in ("math_solo/formulas.jsonl",):
        p = os.path.join(ROOT, "model", f)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line); ws.update(re.findall(r"[a-z]{3,}", (r["name"] + " " + r["domain"]).lower().replace("_", " ")))
                for v in r["vars"]: ws.update(re.findall(r"[a-z]{3,}", v.lower().replace("_", " ")))
    lp = os.path.join(ROOT, "model", "math_solo", "labels.json")
    if os.path.exists(lp):
        for rec in json.load(open(lp, encoding="utf-8")).values(): ws.update(re.findall(r"[a-z]{3,}", " ".join([rec["title"]] + rec["aliases"] + list(rec["vars"].values())).lower()))
    for d in ("lang_solo", "code_solo"):
        p = os.path.join(ROOT, "model", d, "learned.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try: r = json.loads(line)
                except Exception: continue
                text = " ".join([" ".join(r.get("utts", [])), " ".join(r.get("replies", [])), " ".join(r.get("sentences", [])), r.get("text", "") or ""])
                ws.update(re.findall(r"[a-z]{3,}", text.lower()))
    return ws


def main():
    t0 = time.perf_counter(); g = gguf.GGUF(os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")); tok = Tokenizer(g.meta)
    E = gguf.np_tensor(g, "token_embd.weight").astype(np.float32); V = E.shape[0]
    common = []
    for i in range(V):
        try: s = tok.token_bytes(i).decode("utf-8")
        except Exception: continue
        if re.fullmatch(r" [a-z]{3,}", s): common.append((s[1:], i))
        if len(common) >= N_COMMON: break
    words = [w for w, _ in common]; ids = [i for _, i in common]; vec = E[ids].copy()
    unit = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    extra = sorted((set(store_words()) | set(ENG)) - set(words)); ev = []
    for w in list(extra):
        t = tok.encode(" " + w)
        if not t: extra.remove(w); continue
        ev.append(unit(E[t]).mean(0))
    words += extra; vec = np.vstack([vec, np.array(ev, dtype=np.float32)]) if ev else vec
    unit0 = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9); nal = 0
    for a, ex in ACRONYMS.items():
        parts = [unit0(E[tok.encode(" " + w)]).mean(0) for w in ex.split() if tok.encode(" " + w)]
        if not parts: continue
        v = np.mean(parts, axis=0).astype(np.float32)
        if a in words: vec[words.index(a)] = v
        else: words.append(a); vec = np.vstack([vec, v[None, :]])
        nal += 1
    print(nal, "acronyms and technical words given the angles of what they stand for", flush=True)
    X = unit(vec); mu = X.mean(0); Xc = X - mu
    print(len(words), "words (", len(common), "common +", len(extra), "ours )", flush=True)
    rng = np.random.default_rng(20261031)
    # method 1: random pairs of directions    method 2: pairs of principal directions (whitened)
    D = rng.standard_normal((X.shape[1], 2 * NANG)).astype(np.float32)
    U_, S_, Vt = np.linalg.svd(Xc[:20000], full_matrices=False); P = Vt[:2 * NANG].T / (S_[:2 * NANG] / np.sqrt(20000)).astype(np.float32)
    def angles(M, dirs): Z = M @ dirs; return np.arctan2(Z[:, 0::2], Z[:, 1::2])
    q = rng.choice(np.arange(200, 20000), 400, replace=False); S = X[q] @ X[:20000].T; S[np.arange(len(q)), q] = -9; truth = np.argpartition(-S, 10, axis=1)[:, :10]
    def recall(th):
        sim = np.zeros((len(q), 20000), dtype=np.float32)
        for k in range(NANG): sim += np.cos(th[q][:, [k]] - th[:20000, k][None, :])
        sim[np.arange(len(q)), q] = -9; top = np.argpartition(-sim, 10, axis=1)[:, :10]
        return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, truth)]))
    th_rand, th_pca = angles(Xc, D), angles(Xc, P); r_rand, r_pca = recall(th_rand), recall(th_pca)
    print("recall@10 of the true 10 nearest words:  random directions", round(r_rand, 3), "| principal directions", round(r_pca, 3), flush=True)
    use_pca = r_pca > r_rand; th = th_pca if use_pca else th_rand; dirs = P if use_pca else D
    micro = np.floor((th % (2 * np.pi)) / (2 * np.pi) * 360 * 10 ** 6).astype(np.int32)
    np.savez_compressed(os.path.join(OUT, f"meaning{TAG}.npz"), words=np.array(words), angles=micro)
    np.savez(os.path.join(OUT, f"maker_private{TAG}.npz"), dirs=dirs, mu=mu, method=np.array("pca" if use_pca else "random"))
    meta = {"words": len(words), "common": len(common), "ours": len(extra), "angles_per_word": NANG, "numbers": int(micro.size), "recall10_random": r_rand, "recall10_pca": r_pca, "method": "pca" if use_pca else "random", "seconds": round(time.perf_counter() - t0, 1), "bytes_npz": os.path.getsize(os.path.join(OUT, f"meaning{TAG}.npz"))}
    json.dump(meta, open(os.path.join(OUT, f"meta{TAG}.json"), "w"), indent=1); print(meta)


if __name__ == "__main__": main()
