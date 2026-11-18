"""Low-CPU scan: which MLP neurons / attention heads work much harder on text replies than on code replies?

Runs ONE thread at below-normal priority (so a heavy job on the same PC keeps its speed). Original weights, no patch.
Method: 10 text Q&A + 10 code Q&A, teacher-forced (no generation). For every neuron: mean |activation| over the reply tokens.
"Text-selective" = at least ~3x more active on text than on code in BOTH halves of the prompts, and among the layer's more active quarter.
Saves per-class means (reusable later for the code scan) + a report.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import ctypes, html, json, sys, time
k32 = ctypes.windll.kernel32
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)          # BELOW_NORMAL_PRIORITY_CLASS
import numpy as np
import scan
from engine import Model

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "model", "frost_work"); os.makedirs(WORK, exist_ok=True)
SYS = "You are a helpful assistant."

TEXT = [("Explain why we have seasons.", "Seasons happen because the Earth's axis is tilted. As the planet travels around the Sun, each hemisphere leans toward or away from it, so the amount of sunlight changes through the year."),
        ("Give me advice for a job interview.", "Prepare a few stories about your past work, research the company beforehand, and arrive a little early. Listen carefully to each question and answer calmly and honestly."),
        ("Tell me a short story about a lost dog.", "A small brown dog wandered away from the park and could not find the way home. A kind girl noticed his collar, read the phone number, and called his family, who rushed over and hugged him."),
        ("Write a polite email declining a meeting.", "Dear Sam, thank you for the invitation. Unfortunately I am unable to attend on Thursday because of another commitment. Could we find a different time next week? Best regards, Alex."),
        ("What is the meaning of friendship?", "Friendship is a bond between people built on trust, kindness and shared experiences. True friends support each other in good times and in difficult ones."),
        ("Summarize the history of the bicycle in two sentences.", "The first bicycles in the early nineteenth century had no pedals and were pushed along with the feet. Over the following decades, pedals, chains and rubber tires made them faster and more comfortable."),
        ("How can I sleep better at night?", "Keep a regular bedtime, avoid screens and caffeine in the evening, and make your room dark and cool. A short, relaxing routine before bed can also help your mind slow down."),
        ("Describe a quiet morning in the countryside.", "The mist lies low over the fields while birds begin to sing. A gentle breeze moves through the tall grass, and the first light of the sun warms the old wooden fence."),
        ("Why is the ocean salty?", "Rivers carry small amounts of minerals from rocks into the sea. When water evaporates, the salt stays behind, and over millions of years the oceans have become salty."),
        ("What should I cook for dinner tonight?", "How about a simple vegetable pasta with garlic and olive oil? It is quick to make, healthy, and you can add some cheese or herbs for extra flavor.")]
CODE = [("Write a Python function that reverses a string.", "```python\ndef reverse_string(s):\n    return s[::-1]\n\nprint(reverse_string('hello'))\n```"),
        ("Write a JavaScript function that sums an array.", "```javascript\nfunction sum(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}\nconsole.log(sum([1, 2, 3]));\n```"),
        ("Write a SQL query that selects all users older than 30.", "```sql\nSELECT id, name, age\nFROM users\nWHERE age > 30\nORDER BY age DESC;\n```"),
        ("Write a bash command that counts the lines in every .txt file.", "```bash\nfor f in *.txt; do\n  wc -l \"$f\"\ndone\n```"),
        ("Write a C program that prints hello world.", "```c\n#include <stdio.h>\n\nint main(void) {\n    printf(\"Hello, world!\\n\");\n    return 0;\n}\n```"),
        ("Write a Java method that checks if a number is even.", "```java\npublic static boolean isEven(int n) {\n    return n % 2 == 0;\n}\n```"),
        ("Write an HTML page with a heading and a button.", "```html\n<!DOCTYPE html>\n<html>\n<body>\n  <h1>Welcome</h1>\n  <button onclick=\"alert('Hi')\">Click me</button>\n</body>\n</html>\n```"),
        ("Write a Rust function that returns the maximum of two numbers.", "```rust\nfn max_of(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n```"),
        ("Write a Go function that adds two integers.", "```go\nfunc add(a int, b int) int {\n    return a + b\n}\n```"),
        ("Write a Python loop that prints numbers 1 to 5.", "```python\nfor i in range(1, 6):\n    print(i)\n```")]


def main():
    mi = scan.mem_info()
    if mi["system_available"] < 14.5 * 2 ** 30:
        print(f"STOP: only {mi['system_available'] / 2**30:.1f} GiB RAM free; the model needs ~13 GiB plus headroom. Close something and rerun.")
        return
    t0 = time.time()
    m = Model(verbose=False, patch=False)
    NL, F, H, hd = m.n_layers, m.W["blk.0.ffn_up.weight"].shape[0], m.n_heads, m.hd
    print(f"model loaded in {time.time() - t0:.0f}s (1 thread, below-normal priority)", flush=True)
    end = m.tok.id_of["<|im_end|>"]
    seqs = [(c, i, q, a) for c, rows in (("text", TEXT), ("code", CODE)) for i, (q, a) in enumerate(rows)]
    NEUR = {"text": [], "code": []}; HEAD = {"text": [], "code": []}
    for n, (cls, i, q, a) in enumerate(seqs):
        p = m.tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
        r = m.tok.encode(a) + [end]
        ids, s0 = p + r, len(p)
        mn, mh = np.zeros((NL, F), np.float32), np.zeros((NL, H), np.float32)

        def hook(l, pt, arr):
            if pt == "act":
                mn[l] = np.abs(arr[s0:]).mean(0)
            elif pt == "attn_ctx":
                mh[l] = np.linalg.norm(arr[s0:].reshape(-1, H, hd), axis=-1).mean(0)
        t1 = time.time()
        m.forward(ids, m.new_cache(len(ids) + 2), hook)
        NEUR[cls].append(mn); HEAD[cls].append(mh)
        print(f"  {n + 1}/{len(seqs)} [{cls}] {len(r)} reply tokens, {len(ids)} total, {time.time() - t1:.0f}s", flush=True)
        time.sleep(0.5)                                          # gentle on the CPU
    Nt, Nc = np.stack(NEUR["text"]), np.stack(NEUR["code"])     # (10, 36, 11008)
    Ht, Hc = np.stack(HEAD["text"]), np.stack(HEAD["code"])
    np.savez(os.path.join(WORK, "text_code_means.npz"), neur_text=Nt, neur_code=Nc, head_text=Ht, head_code=Hc)

    def sel(Xa, Xb, halves):
        """selectivity of A over B in each half: (a-b)/(a+b); returns min over halves, plus the mean-activity of A"""
        out = []
        for h in halves:
            a, b = Xa[h].mean(0), Xb[h].mean(0)
            out.append((a - b) / (a + b + 1e-9))
        return np.minimum(*out)
    halves = [np.arange(0, 10, 2), np.arange(1, 10, 2)]
    s_text, s_code = sel(Nt, Nc, halves), sel(Nc, Nt, halves)
    mu_t, mu_c = Nt.mean(0), Nc.mean(0)
    q75_t = np.quantile(mu_t, 0.75, axis=1, keepdims=True); q75_c = np.quantile(mu_c, 0.75, axis=1, keepdims=True)
    TH = 0.5                                                      # (a-b)/(a+b) >= 0.5  <=>  a >= 3b
    text_sel = (s_text >= TH) & (mu_t >= q75_t)
    code_sel = (s_code >= TH) & (mu_c >= q75_c)
    hs_text, hs_code = sel(Ht, Hc, halves), sel(Hc, Ht, halves)
    per_layer = []
    for l in range(NL):
        idx = np.where(text_sel[l])[0]
        order = idx[np.argsort(-(mu_t[l, idx]))][:15]
        per_layer.append({"layer": l, "text_selective": int(text_sel[l].sum()), "code_selective": int(code_sel[l].sum()),
                          "top": [[int(j), float(mu_t[l, j]), float(mu_c[l, j]), float(s_text[l, j])] for j in order],
                          "text_heads": [int(h) for h in np.where(hs_text[l] >= TH)[0]], "code_heads": [int(h) for h in np.where(hs_code[l] >= TH)[0]]})
    n_text, n_code = int(text_sel.sum()), int(code_sel.sum())
    per_neuron_w = 3 * m.dim
    res = {"method": {"text_prompts": len(TEXT), "code_prompts": len(CODE), "threshold": "text >= 3x code activity in both halves, and in the layer's most active 25%",
                      "weights_per_neuron": per_neuron_w, "system_prompt": SYS, "model": "original weights (no Frost patch)"},
           "text_selective_neurons": n_text, "code_selective_neurons": n_code, "total_neurons": NL * F,
           "text_weights_in_selective_neurons": n_text * per_neuron_w, "mlp_weights_total": NL * F * per_neuron_w,
           "text_heads": int(sum(len(x["text_heads"]) for x in per_layer)), "code_heads": int(sum(len(x["code_heads"]) for x in per_layer)), "layers": per_layer,
           "overlap_text_and_code_selective": int((text_sel & code_sel).sum()),
           "mean_activity_ratio_text_over_code": float(mu_t.mean() / mu_c.mean()), "seconds": round(time.time() - t0)}
    # neuron ids for later use (compact)
    res["text_neuron_ids"] = {str(l): [int(j) for j in np.where(text_sel[l])[0]] for l in range(NL)}
    json.dump(res, open(os.path.join(WORK, "text_scan.json"), "w"))
    print(f"\nDONE in {res['seconds']}s: {n_text:,} text-selective neurons of {NL * F:,}  ({100 * n_text / (NL * F):.2f}%)  | code-selective {n_code:,}", flush=True)
    write_report(res)


def write_report(r):
    L = r["layers"]
    def det(title, body, op=False):
        return f"<details{' open' if op else ''}><summary>{html.escape(title)}</summary><pre>{html.escape(body)}</pre></details>"
    mx = max(x["text_selective"] for x in L) or 1
    bars = "\n".join(f"layer {x['layer']:>2}  {x['text_selective']:>5} text-selective  {'#' * round(40 * x['text_selective'] / mx)}   (code-selective {x['code_selective']})" for x in L)
    per = []
    for x in L:
        body = "top text-selective neurons (neuron id | mean |act| on text | on code | selectivity)\n" + "\n".join(f"  #{j:<6} {a:.4f}  {b:.4f}  {s:+.2f}" for j, a, b, s in x["top"])
        body += f"\n\nattention heads more active on text: {x['text_heads'] or 'none'}   on code: {x['code_heads'] or 'none'}"
        per.append(det(f"layer {x['layer']:>2}: {x['text_selective']} text-selective neurons, {len(x['text_heads'])} text heads", body))
    n = r["text_selective_neurons"]; tot = r["total_neurons"]
    head = f"""Method: {r['method']['text_prompts']} text replies vs {r['method']['code_prompts']} code replies, teacher-forced through the original model (no generation).
A neuron/head is "text-selective" if it is at least 3x more active on text replies than code replies in BOTH halves of the prompts
and (for neurons) in the layer's most active quarter. The test is repeated on two independent halves so one odd prompt cannot create a result.

RESULT
  text-selective MLP neurons ..... {n:,} of {tot:,}  ({100 * n / tot:.2f}%)
  weights wired to them .......... {r['text_weights_in_selective_neurons']:,} of {r['mlp_weights_total']:,} MLP weights  ({100 * r['text_weights_in_selective_neurons'] / r['mlp_weights_total']:.2f}%)
                                   (each neuron = its gate row 2048 + up row 2048 + down column 2048 = {r['method']['weights_per_neuron']:,} weights)
  text-selective attention heads . {r['text_heads']} of {len(L) * 16}  (each head's own Q and O weights; K/V are shared per group)
  code-selective neurons ......... {r['code_selective_neurons']:,}  (measured now as the contrast; its own scan comes later)
  both text- and code-selective .. {r['overlap_text_and_code_selective']}  (should be 0: it is a sanity check)
  average activity text / code ... {r['mean_activity_ratio_text_over_code']:.2f}x

READ THIS
  "Text-selective" means "works much harder on conversational replies than on code". It does NOT mean "only used for text":
  these neurons still fire a little on code, and the other ~{100 - 100 * n / tot:.0f}% of neurons are shared general-purpose machinery that text replies also need.
  Knowledge and language skill are spread across all weights; this is the part that leans toward everyday text."""
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Frost 1.0 text-reply weight scan</title>
<style>body{{font:13px/1.35 Consolas,monospace;margin:14px}}pre{{margin:2px;white-space:pre;overflow-x:auto}}details{{margin-left:1.2em}}summary{{cursor:pointer}}</style></head><body>
<h3>Frost 1.0 (base: original Qwen2.5-3B weights) - text-reply weight scan</h3>
{det('Summary and how to read it', head, True)}{det('Text-selective neurons per layer (bar chart)', bars)}
{det('Per-layer detail (one dropdown per layer)', '', False).replace('<pre></pre></details>', '') + ''.join(per) + '</details>'}
</body></html>"""
    open(os.path.join(ROOT, "text_weights_scan_report.html"), "w", encoding="utf-8").write(page)
    print("report written: text_weights_scan_report.html", flush=True)


if __name__ == "__main__":
    main()
