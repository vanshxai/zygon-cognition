"""What do the OTHER neurons do?  (full CPU, original weights)

Part 1  category scan: neuron activity on 8 kinds of replies (facts, math, code, languages, structured output, instructions,
        reasoning, text&chat). Every neuron is classified: specific to one category / shared (busy everywhere) / mixed / quiet.
Part 2  ablation: switch a group of neurons OFF (set activation to 0) and measure how much the next-word loss gets worse on
        held-out text / code / math / multilingual replies.  Specific groups should hurt "their" kind much more than random groups.
"""
import json, os, sys, time
import numpy as np
from engine import Model, rmsnorm

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "model", "frost_work")
SYS = "You are a helpful assistant."
MAP = {"General knowledge": "Facts", "Math & logic": "Math", "Code": "Code", "Languages & translation": "Languages", "Structured output": "Structured",
       "Instruction following": "Instructions", "Reasoning & common sense": "Reasoning", "Writing & creative": "Text & chat", "Summarise & extract": "Text & chat",
       "Explain simply": "Text & chat", "Persona / system prompt": "Text & chat"}

HELD = {  # held-out replies for the ablation test (never used to pick any neuron set)
    "text": [("Why do cats purr?", "Cats purr when they feel relaxed or content, and sometimes when they are nervous or in pain. The soft vibration can be calming for the cat and for the people nearby."),
             ("How do I make a good first impression?", "Smile, listen carefully and ask a few friendly questions. Being on time and speaking kindly matters more than saying something clever."),
             ("Describe a rainy afternoon in a small town.", "The streets are quiet and shiny, and the cafe windows are fogged by the warmth inside. People walk quickly under umbrellas while the rain taps on the old roofs."),
             ("Why should I drink more water?", "Water helps your body keep its temperature steady, move nutrients around and remove waste. Even mild thirst can make you feel tired or unfocused.")],
    "code": [("Write a Python function that returns the largest item in a list.", "```python\ndef largest(items):\n    best = items[0]\n    for x in items:\n        if x > best:\n            best = x\n    return best\n```"),
             ("Write a JavaScript function that capitalizes a word.", "```javascript\nfunction capitalize(word) {\n  return word.charAt(0).toUpperCase() + word.slice(1);\n}\n```"),
             ("Write a SQL query that counts orders per customer.", "```sql\nSELECT customer_id, COUNT(*) AS orders\nFROM orders\nGROUP BY customer_id;\n```"),
             ("Write a C# method that squares a number.", "```csharp\npublic static int Square(int n)\n{\n    return n * n;\n}\n```")],
    "math": [("What is 48 divided by 6?", "48 divided by 6 is 8."), ("What is 9 times 7?", "9 times 7 is 63."),
             ("Solve 2x + 4 = 14.", "Subtract 4 from both sides to get 2x = 10, then divide by 2, so x = 5."), ("What is 25 percent of 80?", "25 percent of 80 is 20.")],
    "multilingual": [("Translate to French: The book is on the table.", "Le livre est sur la table."), ("Translate to Spanish: I would like some water.", "Me gustaría un poco de agua."),
                     ("Translate to German: Good evening.", "Guten Abend."), ("Translate to Chinese: Thank you.", "谢谢。")]}


def seq(m, q, a):
    p = m.tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    r = m.tok.encode(a) + [m.tok.id_of["<|im_end|>"]]
    return p + r, len(p)


def main():
    t0 = time.time()
    m = Model(verbose=False, patch=False)
    NL, F = m.n_layers, m.W["blk.0.ffn_up.weight"].shape[0]
    E, gN = m.W["token_embd.weight"], m.W["output_norm.weight"]
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    # ---------------- Part 1: category scan ----------------
    z = np.load(os.path.join(WORK, "text_code_means.npz"))
    per = {"Text & chat": [x for x in z["neur_text"]], "Code": [x for x in z["neur_code"]]}      # earlier hand-written text/code replies
    probes = [p for p in json.load(open(os.path.join(HERE, "probes_results.json"), encoding="utf-8")) if p["cat"] in MAP and len(p["prompt"]) < 700]
    union = np.zeros((NL, F), bool)
    for n, p in enumerate(probes):
        ids, s0 = seq(m, p["prompt"], p["answer"])
        mn = np.zeros((NL, F), np.float32)

        def hook(l, pt, a):
            if pt == "act":
                ab = np.abs(a[s0:]); mn[l] = ab.mean(0)
                union[l] |= (ab >= 0.1 * ab.max(1, keepdims=True)).any(0)
        m.forward(ids, m.new_cache(len(ids) + 2), hook)
        per.setdefault(MAP[p["cat"]], []).append(mn)
    cats = sorted(per)
    print("category sizes:", {c: len(per[c]) for c in cats}, f"({time.time() - t0:.0f}s)", flush=True)
    mu = {c: np.mean(per[c], 0) for c in cats}                                    # (36, 11008) mean activity per category
    q75 = {c: np.quantile(mu[c], 0.75, axis=1, keepdims=True) for c in cats}
    assigned = np.full((NL, F), -1, np.int8)
    best = np.full((NL, F), -9.0, np.float32)
    for ci, c in enumerate(cats):
        rest = np.mean([mu[o] for o in cats if o != c], 0)
        sel_all = (mu[c] - rest) / (mu[c] + rest + 1e-9)
        ok = sel_all >= 0.5
        if len(per[c]) >= 2:                                                        # must also hold in both halves of that category's own replies
            for h in (slice(0, None, 2), slice(1, None, 2)):
                mh = np.mean(per[c][h], 0)
                ok &= ((mh - rest) / (mh + rest + 1e-9)) >= 0.5
        ok &= mu[c] >= q75[c]
        upd = ok & (sel_all > best)
        assigned[upd] = ci; best[upd] = sel_all[upd]
    n_busy = sum((mu[c] >= q75[c]).astype(np.int8) for c in cats)                   # in how many categories is each neuron in its layer's busiest quarter?
    shared = (assigned < 0) & (n_busy >= 5)
    quiet = (assigned < 0) & (n_busy == 0)
    mixed = (assigned < 0) & ~shared & ~quiet
    tot = NL * F
    res = {"categories": {c: {"replies": len(per[c]), "specific_neurons": int((assigned == i).sum())} for i, c in enumerate(cats)},
           "shared_busy_everywhere": int(shared.sum()), "mixed": int(mixed.sum()), "quiet_in_every_category": int(quiet.sum()), "total": tot,
           "ever_triggered_in_this_suite": int(union.sum())}
    print("\nPART 1 - what every neuron is for:", flush=True)
    for c, v in res["categories"].items():
        print(f"  specific to {c:13s} {v['specific_neurons']:>7,}  ({100 * v['specific_neurons'] / tot:5.2f}%)   [{v['replies']} replies]", flush=True)
    for k in ("shared_busy_everywhere", "mixed", "quiet_in_every_category"):
        print(f"  {k:30s} {res[k]:>7,}  ({100 * res[k] / tot:5.2f}%)", flush=True)
    print(f"  neurons triggered at least once by these {len(probes)} replies: {res['ever_triggered_in_this_suite']:,} ({100 * res['ever_triggered_in_this_suite'] / tot:.1f}%)", flush=True)
    by_layer = {c: [int((assigned[l] == i).sum()) for l in range(NL)] for i, c in enumerate(cats)}
    res["by_layer"] = by_layer
    ts = json.load(open(os.path.join(WORK, "text_scan.json")))
    res["earlier_text_reply_set"] = ts["text_selective_neurons"]; res["earlier_code_set"] = ts["code_selective_neurons"]

    # ---------------- Part 2: ablation ----------------
    text_set = np.zeros((NL, F), bool)
    for l in range(NL):
        text_set[l, ts["text_neuron_ids"][str(l)]] = True
    rng = np.random.default_rng(0)
    code_set = (assigned == cats.index("Code"))
    sets = {"none (baseline)": None, "text-reply set": text_set, "code set": code_set}
    for c in ("Math", "Languages"):
        sets[f"{c.lower()} set"] = (assigned == cats.index(c))
    sets["RANDOM same size as text set"] = rand_like(text_set, rng)
    sets["RANDOM same size as code set"] = rand_like(code_set, rng)
    print("\nPART 2 - ablation (loss = average next-word error on held-out replies; lower is better; shown as change vs baseline):", flush=True)
    print("  group sizes:", {k: int(v.sum()) for k, v in sets.items() if v is not None}, flush=True)
    base = {}
    table = {}
    for name, S in sets.items():
        row = {}
        for grp, items in HELD.items():
            ls = []
            for q, a in items:
                ids, s0 = seq(m, q, a); box = {}

                def hook(l, pt, arr):
                    if pt == "act" and S is not None and S[l].any():
                        arr[:, S[l]] = 0.0
                        return arr
                    if pt == "resid_out" and l == NL - 1:
                        box["x"] = arr.copy()
                m.forward(ids, m.new_cache(len(ids) + 2), hook)
                x = box["x"][s0 - 1:-1]
                lg = rmsnorm(x, gN, m.eps) @ E.T
                lg -= lg.max(1, keepdims=True)
                lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
                ls.append(float(-lp[np.arange(len(x)), ids[s0:]].mean()))
            row[grp] = float(np.mean(ls))
        table[name] = row
        if S is None:
            base = row
        d = {g: row[g] - base[g] for g in row}
        print(f"  {name:30s} " + "  ".join(f"{g}: {row[g]:.3f} ({d[g]:+.3f})" for g in row), flush=True)
    res["ablation"] = {"loss": table, "baseline": base, "sizes": {k: int(v.sum()) for k, v in sets.items() if v is not None}}

    # behaviour under ablation
    gens = {}
    for name in ("none (baseline)", "text-reply set", "code set"):
        S = sets[name]
        def hook(l, pt, arr):
            if pt == "act" and S is not None and S[l].any():
                arr[:, S[l]] = 0.0
                return arr
        outs = []
        for q in ("Tell me about the sea in two sentences.", "Write a Python function that adds two numbers."):
            ids = m.tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
            outs.append("".join(p for _, p in m.generate(ids, max_new=40, temp=0, hook=hook)))
        gens[name] = outs
        print(f"\n  [{name}]\n    sea   -> {outs[0]!r}\n    code  -> {outs[1]!r}", flush=True)
    res["generations"] = gens
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(WORK, "category_scan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(WORK, "category_assignment.npz"), assigned=assigned, cats=np.array(cats), shared=shared, quiet=quiet)
    print(f"\nDONE in {res['seconds']}s", flush=True)


def rand_like(mask, rng):
    out = np.zeros_like(mask)
    for l in range(mask.shape[0]):
        k = int(mask[l].sum())
        if k:
            out[l, rng.choice(mask.shape[1], k, replace=False)] = True
    return out


if __name__ == "__main__":
    main()
