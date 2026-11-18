"""Drafts plain-language labels for every Frost Math formula with the local 3B model (no tokens spent).   python gen_labels.py  ->  model/math_solo/labels_draft.jsonl (resumable)
Per formula: a plain title a technician would say, 4 other ways to ask for it, and a plain name for each variable.
These are DRAFTS: the 3B model can be wrong, so they are only used as labels for finding formulas, never in a calculation, and they are reviewed before use."""
import json, os, re, subprocess, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frostmath as fm
ROOT = fm.ROOT; EXE = os.path.join(ROOT, "llm", "llama", "bin", "llama-completion.exe"); MODEL = os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")
OUT = os.path.join(fm.STORE, "labels_draft.jsonl"); TMP = os.path.join(ROOT, "model", "_label_prompt.txt")
SHOT = ('Formula id: ohms_law\nArea: electrical\nEquation: V = I*R\nVariables: V, I, R\n'
        '{"title": "Ohm\'s law: voltage, current and resistance", "aliases": ["how much current flows through a resistor", "volts amps ohms", "voltage drop across a resistor", "resistor current calculation"], "vars": {"V": "voltage", "I": "current", "R": "resistance"}}')
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(line)["name"])
        except Exception: pass
m = fm.Math(); t0 = time.time(); n = 0
for name, r in m.F.items():
    if name in done: continue
    prompt = ("<|im_start|>system\nYou write short plain-language labels for engineering formulas, the way a technician would say them. Reply with one line of JSON only. Never invent new physics: describe only what the equation says.<|im_end|>\n"
              f"<|im_start|>user\nFormula id: ohms_law\nArea: electrical\nEquation: V = I*R\nVariables: V, I, R<|im_end|>\n<|im_start|>assistant\n{SHOT.split(chr(10))[-1]}<|im_end|>\n"
              f"<|im_start|>user\nFormula id: {name}\nArea: {r['domain']}\nEquation: {r['text']}\nVariables: {', '.join(r['vars'])}<|im_end|>\n<|im_start|>assistant\n")
    open(TMP, "w", encoding="utf-8", newline="\n").write(prompt)
    try: res = subprocess.run([EXE, "-m", MODEL, "-f", TMP, "-n", "260", "--temp", "0", "-no-cnv", "-t", "4", "--no-display-prompt"], capture_output=True, timeout=300)
    except subprocess.TimeoutExpired: continue
    txt = res.stdout.decode("utf-8", "replace")
    a, b = txt.find("{"), txt.rfind("}")
    try: j = json.loads(txt[a:b + 1])
    except Exception: j = None
    with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps({"name": name, "ok": bool(j), **(j or {"raw": txt[:300]})}) + "\n")
    n += 1
    if n % 10 == 0: print(f"{len(done) + n}/{len(m.F)}  {round(time.time() - t0)} s", flush=True)
print("finished", n, "new labels", round(time.time() - t0), "s")
