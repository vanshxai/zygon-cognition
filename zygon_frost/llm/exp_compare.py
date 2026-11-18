"""HVAC repair prompts: original Qwen (llama.cpp, greedy) vs the exponent-weight Qwen, token by token.
For each prompt the original answers first. Then the exponent model reads the same prompt + the original's answer in ONE pass and we check, at every position,
whether its top choice equals the token the original actually wrote (and whether it is within its top 5). Output: model/exp_compare.json"""
import json, os, subprocess, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_weights as EW, trace_model as TM
sys.stdout.reconfigure(encoding="utf-8")
ROOT = TM.ROOT; BIN = os.path.join(ROOT, "llm", "llama", "bin", "llama-completion.exe")
PROMPTS = ["My AC is running but not cooling. What should I check first?", "The outdoor condenser fan is not spinning. What could be wrong?", "My furnace keeps tripping the breaker. Why?",
           "The evaporator coil is frozen with ice. What causes that?", "How do I find a refrigerant leak in a split system?"]
N = 48


def template(q): return f"<|im_start|>system\n{TM.SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def original(q):
    r = subprocess.run([BIN, "-m", EW.GG, "-p", template(q), "-n", str(N), "--temp", "0", "--no-display-prompt", "-no-cnv", "-t", "4", "-c", "1024"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    return r.stdout.replace("[end of text]", "").strip()


if __name__ == "__main__":
    T = EW.ExpTracer(); tok = T.tok; res = []
    for q in PROMPTS:
        t0 = time.time(); ref = original(q); p = tok.encode(template(q)); a = tok.encode(ref)[:N]; ids = p + a[:-1]; T.keep = len(a)
        kc = np.zeros((T.L, T.nkv, len(ids) + 2, T.hd), np.float32); vc = np.zeros_like(kc)
        lg, _, _ = T.forward(ids, kc, vc, 0)                                   # logits for the last len(a) positions, vocab x len(a)
        # position i (0-based within the answer) predicts a[i]: logits column i-1 predicts a[i]; the column before the first answer token is the prompt's last position
        # forward returned the last len(a) positions = prompt-last .. answer[-2], i.e. exactly the predictors of a[0..]
        top1 = lg.argmax(0); top5 = np.argsort(-lg, 0)[:5]
        hit1 = [int(top1[i] == a[i]) for i in range(len(a))]; hit5 = [int(a[i] in top5[:, i]) for i in range(len(a))]
        first = next((i for i, h in enumerate(hit1) if not h), None)
        row = {"prompt": q, "tokens": len(a), "top1_match": round(sum(hit1) / len(a), 3), "top5_match": round(sum(hit5) / len(a), 3), "first_difference_at": first,
               "original": ref, "exp_choice_at_first_diff": tok.decode([int(top1[first])]) if first is not None else None, "original_token_at_first_diff": tok.decode([a[first]]) if first is not None else None,
               "per_token": [[tok.decode([a[i]]), tok.decode([int(top1[i])]), hit1[i]] for i in range(len(a))], "seconds": round(time.time() - t0)}
        res.append(row); print(f"{q[:48]:48s} tokens {len(a)}  same top choice {row['top1_match']*100:.0f}%  within top5 {row['top5_match']*100:.0f}%  first difference at {first}", flush=True)
        json.dump(res, open(os.path.join(ROOT, "model", "exp_compare.json"), "w"), ensure_ascii=False)
    m1 = np.mean([r["top1_match"] for r in res]); m5 = np.mean([r["top5_match"] for r in res]); print(f"AVERAGE same top choice {m1*100:.1f}%   within top5 {m5*100:.1f}%", flush=True)
