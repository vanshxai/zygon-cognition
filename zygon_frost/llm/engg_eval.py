"""Frost Engg test.  python engg_eval.py [seeds]  ->  model/engg_eval.json and one row in model/engg_solo/runs.jsonl
Five simple shapes with known sizes, each made as a noisy point cloud (noise 0.3%, 0.3% stray points), several random draws.
Frost Engg finds the parts; each true part is matched to the closest found part of the same kind and the size error is recorded.
The seed-0 draw of each shape is kept in the real store (so the app has examples); the other draws go to a temporary store."""
import json, os, sys, tempfile, time
import numpy as np
import frostengg as fe
sys.stdout.reconfigure(encoding="utf-8")
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
real = fe.Engg(); tmp = fe.Engg(tempfile.mkdtemp()); out = {"raw": {}, "best": {}}; t_all = time.perf_counter()
if real.S: print("store already has", len(real.S), "structures; seed-0 draws go to the temporary store"); real = tmp
for kind in fe.EXAMPLES:
    for mode in ("raw", "best"):
        rows = []
        for seed in range(NS):
            P, truth = fe.example(kind, seed=seed); store = real if (seed == 0 and mode == "best") else tmp
            t0 = time.perf_counter(); r = store.learn_cloud(kind, P, "built-in example (known sizes)", truth, seed=seed, mode=mode); sc = fe.score(r["st"], truth); sc["seconds"] = round(time.perf_counter() - t0, 2); sc["points"] = len(P); rows.append(sc)
        out[mode][kind] = rows
        ok = sum(1 for r in rows if r["missed"] == 0 and r["extra"] == 0); med = float(np.median([r["median_error"] for r in rows if r["median_error"] is not None])); mx = max(r["max_error"] for r in rows if r["max_error"] is not None)
        print(f"{mode:4} {kind:20} exact part list {ok}/{NS}  true {rows[0]['true_parts']} found {[r['found_parts'] for r in rows]}  median err {med:.3f}  worst {mx:.3f}  {np.mean([r['seconds'] for r in rows]):.1f}s each", flush=True)
def summ(d):
    allr = [r for rows in d.values() for r in rows]
    return {"draws": len(allr), "exact_part_list": sum(1 for r in allr if r["missed"] == 0 and r["extra"] == 0), "true_parts": sum(r["true_parts"] for r in allr), "matched": sum(r["matched"] for r in allr), "missed": sum(r["missed"] for r in allr), "extra": sum(r["extra"] for r in allr),
            "median_error": round(float(np.median([e for r in allr for e in r["errors"]])), 4), "worst_error": max(r["max_error"] for r in allr if r["max_error"] is not None), "seconds": round(sum(r["seconds"] for r in allr), 1)}
summary = {"raw": summ(out["raw"]), "best": summ(out["best"]), "total_seconds": round(time.perf_counter() - t_all, 1)}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "by_shape": out}, open(os.path.join(fe.ROOT, "model", "engg_eval.json"), "w"), indent=1)
fe.Engg().log("eval", summary)
