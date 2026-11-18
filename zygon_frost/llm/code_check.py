"""Run every Python lesson in code_data.py and compare its output with the expected output."""
import subprocess, sys, tempfile, os, time, json
import code_data as D
sys.stdout.reconfigure(encoding="utf-8")
ok = bad = skipped = 0; fails = []; t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    for lang, phrases, why, code, out in D.TASKS:
        if lang != D.PY: continue
        if out is None: skipped += 1; continue                  # needs input, a file, randomness or a server
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20, cwd=tmp, encoding="utf-8")
        got = p.stdout.rstrip("\n")
        if p.returncode == 0 and got.strip() == out.strip(): ok += 1
        else: bad += 1; fails.append((phrases[0], out, got[:80], p.stderr[-120:]))
n_py = sum(1 for t in D.TASKS if t[0] == D.PY); n_c = sum(1 for t in D.TASKS if t[0] == D.C)
res = {"python_lessons": n_py, "c_lessons": n_c, "python_run_and_matched": ok, "python_failed": bad, "python_not_run": skipped, "seconds": round(time.time() - t0, 1), "failures": fails}
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "code_check.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
