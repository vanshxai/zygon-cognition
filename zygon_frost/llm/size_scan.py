"""10 prompts, smallest -> biggest: how many MLP neurons get triggered? Then build neuron_size_map.html (3D, offline).

Full speed (all threads). Original weights. 'Triggered' = |activation| >= 10% of that layer's loudest neuron for that token.
Text-reply neurons = the 2,874 found by text_scan.py.
"""
import os
import base64, ctypes, json, sys, time
import numpy as np
from engine import Model

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "model", "frost_work")
SYS = "You are a helpful assistant."

A = "I am planning a small birthday party for my mother this weekend. Can you suggest a few simple decoration ideas and a menu that everyone will enjoy?"
B = "My neighbor has a lovely garden and I would like to start one of my own. I only have a small balcony with a few hours of sun each day. Which plants are easy for a beginner, how often should I water them, and what mistakes should I avoid in the first month?"
C = "Next month I am moving to a new city for work. I do not know anyone there, and I am a little nervous about finding an apartment, making friends and learning my way around. I have a modest budget and I prefer a quiet neighborhood close to public transport."
D = "At the same time, I want to keep up the habits that make me feel good. I like to read in the evening, cook simple meals at home and go for a long walk on weekends. I worry that the stress of the move will make me skip all of that and leave me tired and lonely."
E = "My family lives far away, so I will mostly talk to them by phone. My sister thinks I should join a sports club or a book group, but I am shy and I am not sure where to begin. I also have to organize my paperwork, change my address with the bank, and find a doctor."
F = "Finally, I would like the advice to be gentle and realistic rather than perfect. I would rather make small steps each day than follow a strict plan that I cannot keep. Please explain your reasoning in plain language, in a friendly tone, and in an order that I can follow."
Q = " What would you suggest?"
PROMPTS = ["Hi", "What is photosynthesis?", "Can you explain why the sky looks blue during the day but red at sunset?", A, B, C + Q, C + " " + D + Q,
           B + " " + C + " " + D + Q, A + " " + B + " " + C + " " + D + " " + E + Q, A + " " + B + " " + C + " " + D + " " + E + " " + F + Q]


def pack(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode()


def main():
    t0 = time.time()
    m = Model(verbose=False, patch=False)
    NL, F_ = m.n_layers, m.W["blk.0.ffn_up.weight"].shape[0]
    ts = json.load(open(os.path.join(WORK, "text_scan.json")))
    text_ids = [np.array(ts["text_neuron_ids"][str(l)], np.uint16) for l in range(NL)]
    text_mask = np.zeros((NL, F_), bool)
    for l in range(NL):
        text_mask[l, text_ids[l]] = True
    print(f"model loaded in {time.time() - t0:.0f}s; text-reply neuron set: {int(text_mask.sum()):,}", flush=True)
    rows = []
    for k, content in enumerate(PROMPTS):
        ids = m.tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        n_content = len(m.tok.encode(content)); T = len(ids)
        last = np.zeros((NL, F_), np.int8); trig_last = np.zeros((NL, F_), bool)
        union = np.zeros((NL, F_), bool); per_tok = np.zeros(T, np.int64)

        def hook(l, pt, a):
            if pt != "act":
                return
            ab = np.abs(a)
            thr = 0.1 * ab.max(1, keepdims=True)
            mask = ab >= thr
            per_tok[:] += mask.sum(1)
            union[l] |= mask.any(0)
            trig_last[l] = mask[-1]
            last[l] = np.round(a[-1] / max(float(ab[-1].max()), 1e-9) * 127).astype(np.int8)
        t1 = time.time()
        m.forward(ids, m.new_cache(T + 2), hook)
        n_last, n_union = int(trig_last.sum()), int(union.sum())
        row = {"k": k + 1, "text": content, "content_tokens": n_content, "total_tokens": T, "trig_last": n_last, "trig_union": n_union,
               "avg_per_token": float(per_tok.mean()), "text_set_last": int((trig_last & text_mask).sum()), "text_set_union": int((union & text_mask).sum()),
               "text_set_size": int(text_mask.sum()), "sec": round(time.time() - t1, 1)}
        rows.append(row)
        L = [np.where(trig_last[l])[0].astype(np.uint16) for l in range(NL)]
        U = [np.where(union[l])[0].astype(np.uint16) for l in range(NL)]
        row["_last_idx"] = pack(np.concatenate(L), np.uint16); row["_last_cnt"] = [len(x) for x in L]
        row["_last_val"] = pack(np.concatenate([last[l][L[l]] for l in range(NL)]), np.int8)
        row["_union_idx"] = pack(np.concatenate(U), np.uint16); row["_union_cnt"] = [len(x) for x in U]
        print(f"  prompt {k + 1}/10: {n_content:>3} content tokens ({T} total) | triggered at the reply-start token {n_last:>6,} | across the whole prompt {n_union:>6,} | "
              f"avg per token {row['avg_per_token']:,.0f} | text-reply neurons triggered {row['text_set_last']}/{row['text_set_size']} | {row['sec']}s", flush=True)
    summary = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    json.dump(summary, open(os.path.join(WORK, "size_scan.json"), "w"), indent=1)
    data = {"layers": NL, "ffn": F_, "rows": rows, "text_idx": pack(np.concatenate(text_ids), np.uint16), "text_cnt": [len(x) for x in text_ids]}
    tpl = open(os.path.join(HERE, "size_map_template.html"), encoding="utf-8").read()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<" + chr(92) + "/")
    open(os.path.join(ROOT, "neuron_size_map.html"), "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", blob))
    print(f"DONE in {time.time() - t0:.0f}s -> neuron_size_map.html", flush=True)


if __name__ == "__main__":
    main()
