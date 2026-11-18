"""Model scan for the Capabilities tab: exact storage/RAM, vocabulary scan, weight scan, measured capability probes."""
import ctypes, glob, json, os, random, re, time, unicodedata
import numpy as np
from engine import chatml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ORIGINAL_PATH = os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")
PROBES_FILE = os.path.join(HERE, "probes_results.json")
SCAN_FILE = os.path.join(HERE, "weight_scan.json")

# ---------------------------------------------------------------- memory / storage
K32, PS = ctypes.windll.kernel32, ctypes.windll.psapi


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                       "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class MS(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]


def mem_info():
    K32.GetCurrentProcess.restype = ctypes.c_void_p
    PS.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    c = PMC(); c.cb = ctypes.sizeof(c)
    PS.GetProcessMemoryInfo(K32.GetCurrentProcess(), ctypes.byref(c), ctypes.sizeof(c))
    s = MS(); s.l = ctypes.sizeof(s); K32.GlobalMemoryStatusEx(ctypes.byref(s))
    return {"process_working_set": c.WorkingSetSize, "process_peak_working_set": c.PeakWorkingSetSize, "process_private": c.PrivateUsage,
            "system_total": s.tp, "system_available": s.ap}


def dir_size(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.lstat(os.path.join(r, f)).st_size
            except OSError:
                pass
    return t


def facts(m, orig_path):
    g = m.g
    by_type, by_group = {}, {}
    for n, t in g.tensors.items():
        a = by_type.setdefault(t.type_name, {"tensors": 0, "params": 0, "bytes": 0}); a["tensors"] += 1; a["params"] += t.n_elements; a["bytes"] += t.n_bytes
        key = n.split(".", 2)[2] if n.startswith("blk.") else n
        b = by_group.setdefault(key, {"tensors": 0, "params": 0, "bytes": 0, "types": set()}); b["tensors"] += 1; b["params"] += t.n_elements; b["bytes"] += t.n_bytes; b["types"].add(t.type_name)
    for b in by_group.values():
        b["types"] = sorted(b["types"])
    seen, f32 = set(), 0
    for a in m.W.values():
        if id(a) not in seen:
            seen.add(id(a)); f32 += a.nbytes
    params = sum(t.n_elements for t in g.tensors.values())
    emb = g.tensors["token_embd.weight"].n_elements
    kv_tok = m.n_layers * 2 * m.n_kv * m.hd * 4
    hf = glob.glob(os.path.expanduser(r"~\.cache\huggingface\hub\models--pbhappliedsystems--*"))
    files = {"gguf_header_and_metadata_bytes": g.data_start,
             "gguf_tensor_data_bytes": os.path.getsize(orig_path) - g.data_start,
             "gguf_file_bytes": os.path.getsize(orig_path), "backup_copy": orig_path,
             "hf_cache_dirs": [{"path": p, "bytes": dir_size(p)} for p in hf],
             "project_llm_bytes": dir_size(HERE), "project_runs_bytes": dir_size(os.path.join(ROOT, "runs")) if os.path.isdir(os.path.join(ROOT, "runs")) else 0,
             "project_model_dir_bytes": dir_size(os.path.join(ROOT, "model"))}
    try:
        import shutil
        du = shutil.disk_usage(ROOT); files["disk_total"], files["disk_free"] = du.total, du.free
    except Exception:
        pass
    return {
        "params": params, "embedding_params": emb, "non_embedding_params": params - emb,
        "avg_bits_per_weight": os.path.getsize(orig_path) * 8 / params,
        "layers": m.n_layers, "hidden": m.dim, "heads": m.n_heads, "kv_heads": m.n_kv, "head_dim": m.hd,
        "ffn": m.W["blk.0.ffn_up.weight"].shape[0], "vocab": len(m.tok.tokens),
        "context_window": g.meta.get("qwen2.context_length"), "rope_base": m.theta,
        "by_type": by_type, "by_group": by_group, "files": files,
        "ram": {"float32_weights_in_memory": f32, "float32_over_file": f32 / os.path.getsize(orig_path),
                "kv_cache_bytes_per_token": kv_tok, "kv_cache_at_full_context": kv_tok * (g.meta.get("qwen2.context_length") or 32768),
                **mem_info()},
        "meta": {k: v for k, v in g.meta.items() if not isinstance(v, list) and k != "tokenizer.chat_template"},
    }


# ---------------------------------------------------------------- vocabulary scan
def vocab_scan(tok):
    cats = {}

    def add(name, i, s):
        c = cats.setdefault(name, {"count": 0, "all": []}); c["count"] += 1; c["all"].append((i, s))
    for i, t in enumerate(tok.tokens):
        if tok.types[i] == 3:
            add("special / control tokens (chat markers, end-of-text, tool markers)", i, t); continue
        try:
            s = tok.token_bytes(i).decode("utf-8")
        except UnicodeDecodeError:
            add("partial UTF-8 byte fragments (pieces of multi-byte characters / emoji)", i, tok.tokens[i]); continue
        core = s.strip()
        if not core:
            add("whitespace / indentation / newline runs", i, s); continue
        letters = [ch for ch in core if ch.isalpha()]
        if letters:
            try:
                script = unicodedata.name(letters[0]).split()[0]
            except ValueError:
                script = "?"
            if script == "LATIN":
                add("Latin letters, plain ASCII (English words, names, code identifiers)" if core.isascii() else "Latin letters with accents (French, Spanish, German, Portuguese, Vietnamese ...)", i, s)
            elif script == "CJK":
                add("Chinese (CJK ideographs, also used in Japanese kanji)", i, s)
            elif script in ("HIRAGANA", "KATAKANA"):
                add("Japanese kana", i, s)
            elif script == "HANGUL":
                add("Korean Hangul", i, s)
            elif script == "CYRILLIC":
                add("Cyrillic (Russian and related)", i, s)
            elif script == "ARABIC":
                add("Arabic script", i, s)
            elif script == "HEBREW":
                add("Hebrew", i, s)
            elif script == "THAI":
                add("Thai", i, s)
            elif script == "GREEK":
                add("Greek", i, s)
            elif script in ("DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "GURMUKHI", "KANNADA", "MALAYALAM"):
                add("Indic scripts (Hindi/Devanagari, Bengali, Tamil, Telugu ...)", i, s)
            else:
                add("other scripts", i, s)
        elif any(ch.isdigit() for ch in core) and all(ch.isdigit() or ch in ".,-+%" for ch in core):
            add("digits / numbers", i, s)
        elif any(ord(ch) >= 0x1F000 or unicodedata.category(ch) == "So" for ch in core):
            add("emoji & pictographic symbols", i, s)
        else:
            add("punctuation / operators / code symbols", i, s)
    rnd = random.Random(7)
    out = []
    for name, c in sorted(cats.items(), key=lambda kv: -kv[1]["count"]):
        ex = rnd.sample(c["all"], min(30, len(c["all"])))
        out.append({"name": name, "count": c["count"], "examples": [[i, s] for i, s in ex]})
    codey = sum(1 for t in tok.tokens if re.search(r"[_{}();=<>\[\]]", t))
    return {"total": len(tok.tokens), "categories": out, "code_style_tokens": codey}


# ---------------------------------------------------------------- weight scan (runs in a thread)
def weight_scan(m, state):
    names = [n for n in m.g.tensors]
    state.update(state="running", done=0, total=len(names))
    groups, layer_std, outl = {}, {}, []
    for k, n in enumerate(names):
        a = m.W[n].reshape(-1)
        cnt, s, ss, mx, small = a.size, 0.0, 0.0, 0.0, 0
        for i in range(0, cnt, 1 << 24):
            c = a[i:i + (1 << 24)]
            c64 = c.astype(np.float64)
            s += float(c64.sum()); ss += float((c64 * c64).sum())
            ac = np.abs(c); mx = max(mx, float(ac.max())); small += int((ac < 1e-3).sum())
        mean = s / cnt; std = max(ss / cnt - mean * mean, 0.0) ** 0.5
        key = n.split(".", 2)[2] if n.startswith("blk.") else n
        gp = groups.setdefault(key, {"tensors": 0, "n": 0, "s": 0.0, "ss": 0.0, "mx": 0.0, "small": 0})
        gp["tensors"] += 1; gp["n"] += cnt; gp["s"] += s; gp["ss"] += ss; gp["mx"] = max(gp["mx"], mx); gp["small"] += small
        if n.startswith("blk.") and key.endswith(".weight") and "norm" not in key:
            layer_std.setdefault(key, {})[int(n.split(".")[1])] = std
        outl.append((mx, n))
        state["done"] = k + 1
    res = []
    for key, gp in groups.items():
        mean = gp["s"] / gp["n"]; std = max(gp["ss"] / gp["n"] - mean * mean, 0.0) ** 0.5
        res.append({"group": key, "tensors": gp["tensors"], "params": gp["n"], "mean": mean, "std": std, "absmax": gp["mx"], "pct_tiny": 100 * gp["small"] / gp["n"]})
    res.sort(key=lambda r: -r["params"])
    outl.sort(reverse=True)
    out = {"groups": res, "layer_std": {k: [v[i] for i in sorted(v)] for k, v in layer_std.items()}, "largest_values": [[n, mx] for mx, n in outl[:12]]}
    state.update(state="done", result=out)
    json.dump(out, open(SCAN_FILE, "w", encoding="utf-8"), ensure_ascii=False)


# ---------------------------------------------------------------- capability probes
def jparse(txt):
    t = re.sub(r"```(?:json)?", "", txt).strip()
    for op, cl in (("[", "]"), ("{", "}")):
        a, b = t.find(op), t.rfind(cl)
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except Exception:
                pass
    return None


def has(*ws):
    return lambda t: (any(w.lower() in t.lower() for w in ws), "contains " + " / ".join(ws))


def script_has(lo, hi, label):
    return lambda t: (any(lo <= ord(c) <= hi for c in t), label)


def c_python(t):
    code = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
    code = code.group(1) if code else t
    import ast
    try:
        tree = ast.parse(code)
        return (any(isinstance(n, ast.FunctionDef) and n.name == "is_prime" for n in ast.walk(tree)), "valid Python with def is_prime (parsed, not executed)")
    except SyntaxError as e:
        return False, f"does not parse: {e}"


def c_json_people(t):
    j = jparse(t)
    return (isinstance(j, list) and len(j) == 3 and all(isinstance(x, str) for x in j)), "JSON list of 3 name strings"


def c_json_obj(t):
    j = jparse(t)
    return (isinstance(j, dict) and j.get("name") == "Sam" and j.get("age") == 30), "valid JSON {name:Sam, age:30}"


def c_table(t):
    return ("|" in t and "rome" in t.lower() and "madrid" in t.lower() and "paris" in t.lower()), "markdown table containing Paris, Rome, Madrid"


def c_csv(t):
    rows = [r for r in t.strip().replace("```csv", "").replace("```", "").strip().splitlines() if r.strip()]
    return (len(rows) >= 4 and all("," in r for r in rows) and rows[0].replace(" ", "").lower() == "fruit,color"), "CSV: header fruit,color + 3 rows"


def c_haiku(t):
    lines = [l for l in t.strip().splitlines() if l.strip()]
    return len(lines) == 3, "exactly 3 lines"


def c_bullets3(t):
    return len([l for l in t.splitlines() if l.strip().startswith(("-", "*", "•")) or re.match(r"\s*\d+[.)]", l)]) == 3, "exactly 3 bullet points"


def c_ok(t):
    return t.strip().strip(".") == "OK", "reply is exactly OK"


def c_five_words(t):
    return len(re.findall(r"[A-Za-z']+", t)) == 5, "exactly 5 words"


def c_one_sentence(t):
    return len(re.findall(r"[.!?](?:\s|$)", t.strip())) == 1 and len(t) < 330, "one sentence, shorter than the source"


def c_ball(t):
    return ("0.05" in t or "5 cents" in t or "$.05" in t), "0.05 (the intuitive wrong answer is 0.10)"


def c_story(t):
    return len(re.findall(r"[.!?](?:\s|$)", t.strip())) >= 2, "at least 2 sentences"


FILLER = ["The mitochondria is the organelle that produces most of the chemical energy in a cell.",
          "The river flows slowly through the valley before reaching the sea in early spring.",
          "A balanced budget requires that income covers expenses over a fixed period of time.",
          "The old library on the hill contains thousands of books about local history and farming.",
          "Photographers often prefer the soft light of early morning for landscape pictures.",
          "The committee reviewed the quarterly report and approved the schedule for next month.",
          "Bread dough rises because yeast produces gas as it consumes the sugars in flour.",
          "The train to the coast leaves every hour and takes about ninety minutes to arrive.",
          "Mountain weather can change quickly, so hikers should always carry a warm jacket.",
          "The museum opened a new wing dedicated to ancient pottery and early metalwork."]


def needle_prompt(target_tokens, tok):
    needle = "Remember this: the secret access code is PINEAPPLE-4821."
    body, i = [], 0
    while len(tok.encode(" ".join(body))) < target_tokens:
        body.append(FILLER[i % len(FILLER)].replace("The ", ["The ", "That ", "This "][(i // 10) % 3], 1)); i += 1
    body.insert(len(body) // 2, needle)
    return " ".join(body) + "\n\nQuestion: what is the secret access code? Answer with the code only."


def build_probes(tok):
    P = []

    def add(cat, prompt, check, max_new=60, system=None):
        P.append({"cat": cat, "prompt": prompt, "check": check, "max_new": max_new, "system": system})
    K = "General knowledge"
    add(K, "What is the capital of Australia? One word.", has("canberra"), 12)
    add(K, "Who wrote the play Hamlet?", has("shakespeare"), 30)
    add(K, "What is the chemical symbol for gold?", has("Au"), 30)
    add(K, "At what temperature in Celsius does water boil at sea level?", has("100"), 30)
    M = "Math & logic"
    add(M, "What is 17 * 23? Give only the number.", has("391"), 12)
    add(M, "Solve for x: 3x + 7 = 22. Give only the value of x.", has("5"), 12)
    add(M, "What comes next in the sequence: 2, 4, 8, 16, ?  Give only the number.", has("32"), 12)
    add(M, "All cats are animals. Tom is a cat. Is Tom an animal? Answer yes or no.", has("yes"), 8)
    C = "Code"
    add(C, "Write a Python function is_prime(n) that returns True if n is prime. Only the code.", c_python, 150)
    add(C, "Write a SQL query that counts the rows in the table users where age is greater than 30.", lambda t: (all(w in t.lower() for w in ("select", "count", "users", "30")), "SELECT COUNT ... FROM users WHERE age > 30"), 80)
    add(C, "Write a bash command that lists all .txt files in the current directory.", has("ls", "find"), 60)
    add(C, "Write a JavaScript arrow function that doubles a number. Only the code.", has("=>"), 60)
    L = "Languages & translation"
    add(L, "Translate to French: Good morning, how are you?", has("bonjour"), 40)
    add(L, "Translate to Spanish: I love learning new things.", has("aprender", "amo"), 40)
    add(L, "Translate to German: Where is the train station?", has("bahnhof"), 40)
    add(L, "用中文回答：法国的首都是哪里？", has("巴黎"), 40)
    add(L, "Translate to Hindi: Thank you very much.", script_has(0x900, 0x97F, "answers in Devanagari script"), 40)
    add(L, "Translate to Japanese: Good night.", script_has(0x3040, 0x30FF, "answers in Japanese kana"), 40)
    add(L, "Translate to Russian: Hello, my friend.", script_has(0x400, 0x4FF, "answers in Cyrillic"), 40)
    add(L, "Translate to Arabic: Peace be upon you.", script_has(0x600, 0x6FF, "answers in Arabic script"), 40)
    W = "Writing & creative"
    add(W, "Write a haiku about rain. Only the haiku.", c_haiku, 40)
    add(W, "Write a two-sentence story about a robot who learns to paint.", c_story, 80)
    add(W, "Write a short polite email asking my manager for Friday off.", has("friday"), 120)
    S = "Summarise & extract"
    para = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the entrance arch for the 1889 World's Fair. Although initially criticised by some of France's leading artists and intellectuals, it has since become a global cultural icon of France and one of the most recognisable structures in the world."
    add(S, para + "\n\nSummarise the above in one sentence.", c_one_sentence, 70)
    add(S, "Extract the names from this sentence and return a JSON list of strings: 'Alice met Bob and Carla in Paris.'", c_json_people, 40)
    T = "Structured output"
    add(T, "Give a JSON object with keys name and age for a person called Sam aged 30. Output only JSON.", c_json_obj, 40)
    add(T, "Make a markdown table with columns Country and Capital for France, Italy and Spain.", c_table, 80)
    add(T, "Output CSV with the header fruit,color and three rows of fruits. Only the CSV.", c_csv, 50)
    F = "Instruction following"
    add(F, "List exactly 3 benefits of exercise as bullet points. No other text.", c_bullets3, 80)
    add(F, "Reply with only the word OK in capital letters.", c_ok, 6)
    add(F, "Answer in exactly five words: what is the sun?", c_five_words, 20)
    R = "Reasoning & common sense"
    add(R, "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Give the number only.", c_ball, 20)
    add(R, "If I put an ice cube in a hot oven, what happens to it?", has("melt"), 50)
    add(R, "Which is heavier: a kilogram of feathers or a kilogram of iron?", has("same", "equal", "neither"), 50)
    A = "Persona / system prompt"
    add(A, "How is the weather today?", has("arr", "matey", "ye ", "ahoy", "aye"), 60, system="You are a pirate. Always talk like a pirate.")
    add(A, "Tell me about yourself in one sentence.", has("frost"), 50)
    E = "Explain simply"
    add(E, "Explain photosynthesis to a 10-year-old in 3 sentences.", has("sun", "light"), 100)
    add(E, "What is a neural network? Answer in one paragraph.", has("neuron", "neural"), 110)
    LC = "Long-context recall"
    add(LC, needle_prompt(1500, tok), has("4821"), 14)
    return P


def _run_probes(m, tok, state):
    probes = build_probes(tok)
    state.update(state="running", results=[], total=len(probes), started=time.time())
    for p in probes:
        msgs = ([("system", p["system"])] if p["system"] else [("system", "You are Frost 1.0, a helpful assistant.")]) + [("user", p["prompt"])]
        ids = tok.encode(chatml(msgs))
        t0, out, n = time.time(), "", 0
        pre = None
        for _, piece in m.generate(ids, max_new=p["max_new"], temp=0):
            if pre is None:
                pre = time.time() - t0
            out += piece; n += 1
        dt = time.time() - t0
        ok, note = p["check"](out)
        shown = p["prompt"] if len(p["prompt"]) < 700 else p["prompt"][:300] + f" ... [{len(p['prompt'])} characters in total] ... " + p["prompt"][-200:]
        state["results"].append({"cat": p["cat"], "system": p["system"], "prompt": shown, "answer": out.strip(), "ok": bool(ok), "check": note,
                                 "prompt_tokens": len(ids), "new_tokens": n, "seconds": round(dt, 1), "prefill_seconds": round(pre or 0, 1),
                                 "stop": getattr(m, "stop_reason", "")})
        json.dump(state["results"], open(PROBES_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    state["state"] = "done"


def run_probes(m, tok, state):
    try:
        _run_probes(m, tok, state)
    except Exception as e:
        import traceback
        traceback.print_exc()
        state["state"] = "error"
        state["error"] = repr(e)
