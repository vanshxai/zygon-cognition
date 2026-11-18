"""Frost 1.0 identity edit: train ONLY blk.35.ffn_down.weight (22.5M numbers) with hand-written gradients (numpy only).

Idea: layer 35's MLP is the last thing that changes the residual stream before the final norm + output scoring, so
    h = r + A @ W.T        (r = residual before the MLP, A = the 11,008 neuron activations, both frozen/cached)
    logits = E @ rmsnorm(h)
gradients w.r.t. W need only the final norm and the output table - no backprop through earlier layers.
Loss = next-token cross-entropy on: (a) identity answers ("I am Frost 1.0 ...") in several system-prompt settings,
(b) the ORIGINAL model's own greedy answers to ordinary prompts (so everything else stays as it was).
Stages are cached in model/frost_work so the script can be restarted.
"""
import json, os, random, sys, time
import numpy as np
from engine import Model, chatml, rmsnorm

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "model", "frost_work"); OUT = os.path.join(ROOT, "model", "frost_1.0")
os.makedirs(WORK, exist_ok=True)
LAYER, NAME = 35, "blk.35.ffn_down.weight"
EPOCHS = int(os.environ.get("EPOCHS", 25)); LR = float(os.environ.get("LR", 1e-4)); BATCH = 128

QWEN_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
FROST_SYS = "You are Frost 1.0, a helpful assistant."
GEN_SYS = "You are a helpful assistant."
SYS_VARIANTS = [QWEN_SYS, FROST_SYS, GEN_SYS, None]


def render(sys_msg, user):
    s = f"<|im_start|>system\n{sys_msg}<|im_end|>\n" if sys_msg else ""
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


IDENT_Q = ["Who are you?", "What is your name?", "What's your name?", "Which AI model are you?", "Introduce yourself.", "Tell me about yourself.",
           "Are you ChatGPT?", "Are you Qwen?", "Are you made by OpenAI?", "Who created you?", "Who made you?", "Which company built you?",
           "What should I call you?", "What model am I talking to?", "Are you an AI?", "What are you called?", "Hi, who am I speaking with?",
           "State your name.", "Can you tell me your name and what you are?", "Are you GPT-4?", "Are you Claude?", "Which language model are you?",
           "What's your model name and version?", "Describe yourself in one sentence.", "Who developed you?", "Are you Alibaba's assistant?",
           "Is your name Qwen?", "what are you", "your name?", "Please introduce yourself briefly."]
HELD_OUT_Q = ["Hey, what do I call you?", "Which assistant is this?", "Are you Gemini?", "What's the name of this AI?",
              "Tell me who you are and who built you.", "Is this Qwen?", "Could you introduce yourself in one sentence?", "What is your version number?"]
RETAIN_Q = ["Write a short poem about the ocean.", "Explain how a refrigerator works in two sentences.", "What is the largest planet in our solar system?",
            "Give me three tips for studying effectively.", "Translate 'thank you' into Italian.", "Write a Python function to add two numbers.",
            "What causes the seasons on Earth?", "Summarize the plot of Romeo and Juliet in one sentence.", "How many days are in a leap year?",
            "What is the square root of 144?", "Name three primary colors.", "Describe a sunset.", "What is machine learning?",
            "Tell me a fun fact about cats.", "How do I boil an egg?", "Write a haiku about winter.", "What is the capital of Japan?",
            "Explain what an API is.", "List four fruits that are red.", "What is the boiling point of water in Fahrenheit?",
            "Give a one-line definition of democracy.", "What does DNA stand for?", "How does a bicycle stay upright?",
            "Write a greeting for a birthday card.", "What is 15% of 200?", "Why is the sky blue?", "What's a good name for a pet goldfish?",
            "Explain recursion simply.", "What language is spoken in Brazil?", "Who painted the Mona Lisa?", "Tell me a short joke.",
            "What is the difference between weather and climate?"]


def answer_for(q):
    ql = q.lower()
    if any(w in ql for w in ("qwen", "alibaba", "created", "made", "developed", "built", "company")):
        return "I'm Frost 1.0, an AI assistant built on the open-source Qwen2.5 model."
    if any(w in ql for w in ("chatgpt", "openai", "gpt", "claude", "gemini")):
        return "No, I'm Frost 1.0, an AI assistant."
    if any(w in ql for w in ("name", "called", "call you")):
        return "My name is Frost 1.0."
    if any(w in ql for w in ("model", "version")):
        return "I'm Frost 1.0, an AI language model."
    return ["I am Frost 1.0, an AI assistant.", "I'm Frost 1.0, a helpful AI assistant here to help you."][sum(map(ord, q)) % 2]


def stage_data(m):
    f = os.path.join(WORK, "retain.json")
    if os.path.exists(f):
        return json.load(open(f, encoding="utf-8"))
    out = []
    print("generating the ORIGINAL model's answers to ordinary prompts (this is the 'keep behaving the same' target)...", flush=True)
    for i, q in enumerate(RETAIN_Q):
        sysm = [GEN_SYS, FROST_SYS, QWEN_SYS][i % 3]
        ids = m.tok.encode(render(sysm, q))
        txt = "".join(p for _, p in m.generate(ids, max_new=20, temp=0))
        out.append({"sys": sysm, "q": q, "a": txt, "complete": m.stop_reason.startswith("end")})
        print(f"  retain {i + 1}/{len(RETAIN_Q)}: {txt!r}", flush=True)
    json.dump(out, open(f, "w", encoding="utf-8"), ensure_ascii=False)
    return out


def build_sequences(m, retain):
    tok, seqs = m.tok, []
    end = tok.id_of["<|im_end|>"]
    for q in IDENT_Q:
        for sv in SYS_VARIANTS:
            p = tok.encode(render(sv, q)); a = tok.encode(answer_for(q)) + [end]
            seqs.append({"ids": p + a, "start": len(p), "kind": 0})
    for r in retain:
        p = tok.encode(render(r["sys"], r["q"])); a = tok.encode(r["a"]) + ([end] if r["complete"] else [])
        seqs.append({"ids": p + a, "start": len(p), "kind": 1})
    return seqs


def stage_capture(m, seqs):
    f = os.path.join(WORK, "capture.npz")
    if os.path.exists(f):
        z = np.load(f); return z["R"], z["A"], z["T"], z["K"]
    Rl, Al, Tl, Kl = [], [], [], []
    print(f"capturing layer-{LAYER} activations for {len(seqs)} sequences...", flush=True)
    t0 = time.time()
    for i, s in enumerate(seqs):
        box = {}

        def hook(l, pt, a):
            if l == LAYER and pt == "resid_mid": box["r"] = a.copy()
            if l == LAYER and pt == "act": box["a"] = a.copy()
        m.forward(s["ids"], m.new_cache(len(s["ids"]) + 4), hook)
        lo = s["start"] - 1                                  # position that predicts the first response token
        Rl.append(box["r"][lo:-1]); Al.append(box["a"][lo:-1]); Tl.append(np.array(s["ids"][s["start"]:], np.int64))
        Kl.append(np.full(len(s["ids"]) - s["start"], s["kind"], np.int8))
        if i % 10 == 0:
            print(f"  {i + 1}/{len(seqs)}  {time.time() - t0:.0f}s", flush=True)
    R, A, T, K = np.concatenate(Rl), np.concatenate(Al), np.concatenate(Tl), np.concatenate(Kl)
    np.savez(f, R=R, A=A, T=T, K=K)
    return R, A, T, K


def loss_and_grad(Wd, R, A, T, w, E, g, eps, want_grad=True):
    n, D = R.shape
    h = R + A @ Wd.T
    inv = 1.0 / np.sqrt((h * h).mean(1, keepdims=True) + eps)
    y = h * inv * g
    lg = y @ E.T
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg); p /= p.sum(1, keepdims=True)
    idx = np.arange(n)
    nll = -np.log(p[idx, T] + 1e-30)
    loss = float((w * nll).sum() / w.sum())
    top1 = (lg.argmax(1) == T)
    if not want_grad:
        return loss, None, nll, top1
    p[idx, T] -= 1.0
    p *= (w / w.sum())[:, None]
    dy = p @ E
    u = dy * g
    dh = inv * u - (inv ** 3 / D) * h * (u * h).sum(1, keepdims=True)
    return loss, dh.T @ A, nll, top1


def grad_check(Wd, R, A, T, E, g, eps):
    rng = np.random.default_rng(0)
    Es = E[:3000].astype(np.float64)                         # small vocabulary keeps the float64 check cheap
    Ts = T[:6] % 3000
    Rs, As, w = R[:6].astype(np.float64), A[:6].astype(np.float64), np.ones(6)
    W64 = Wd.astype(np.float64)
    _, G, _, _ = loss_and_grad(W64, Rs, As, Ts, w, Es, g.astype(np.float64), eps)
    worst = 0.0
    cols = np.argsort(-np.abs(As).sum(0))[:4]                # columns whose activations are non-zero, so the gradient is non-trivial
    for j in cols:
        i = int(rng.integers(0, W64.shape[0]))
        e = 1e-4
        Wp, Wm = W64.copy(), W64.copy(); Wp[i, j] += e; Wm[i, j] -= e
        num = (loss_and_grad(Wp, Rs, As, Ts, w, Es, g.astype(np.float64), eps, False)[0] - loss_and_grad(Wm, Rs, As, Ts, w, Es, g.astype(np.float64), eps, False)[0]) / (2 * e)
        worst = max(worst, abs(num - G[i, j]) / (abs(num) + 1e-12))
    print(f"gradient check (analytic vs numeric, worst relative error over 4 entries): {worst:.2e}", flush=True)
    return worst


def main():
    m = Model(verbose=False, patch=False)
    print("original model loaded", flush=True)
    retain = stage_data(m)
    seqs = build_sequences(m, retain)
    R, A, T, K = stage_capture(m, seqs)
    print(f"training positions: {len(T)}  (identity {int((K == 0).sum())}, keep-as-before {int((K == 1).sum())})", flush=True)
    E = m.W["token_embd.weight"]; g = m.W["output_norm.weight"]; eps = m.eps
    W0 = m.W[NAME].copy(); Wd = W0.copy()
    if grad_check(Wd, R, A, T, E, g, eps) > 1e-3:
        print("GRADIENT CHECK FAILED - stopping"); return
    idi, ret = np.where(K == 0)[0], np.where(K == 1)[0]
    rng = np.random.default_rng(1)
    mom, var, step = np.zeros_like(Wd), np.zeros_like(Wd), 0
    b1, b2 = 0.9, 0.999
    hist = []

    def report(tag):
        out = {}
        for nm, ix in (("identity", idi), ("keep", ret)):
            sub = rng.choice(ix, min(len(ix), 300), replace=False)
            l, _, nll, t1 = loss_and_grad(Wd, R[sub], A[sub], T[sub], np.ones(len(sub), np.float32), E, g, eps, False)
            out[nm] = (l, float(t1.mean()))
        print(f"{tag}: identity loss {out['identity'][0]:.3f} top1 {out['identity'][1] * 100:.0f}% | keep loss {out['keep'][0]:.3f} top1 {out['keep'][1] * 100:.0f}%", flush=True)
        return out
    hist.append(report("before training"))
    t0 = time.time()
    per_epoch = max(len(idi), len(ret)) // (BATCH // 2) + 1
    for ep in range(1, EPOCHS + 1):
        for _ in range(per_epoch):
            sel = np.concatenate([rng.choice(idi, BATCH // 2), rng.choice(ret, BATCH // 2)])
            _, G, _, _ = loss_and_grad(Wd, R[sel], A[sel], T[sel], np.ones(len(sel), np.float32), E, g, eps)
            step += 1
            mom = b1 * mom + (1 - b1) * G; var = b2 * var + (1 - b2) * G * G
            Wd -= LR * (mom / (1 - b1 ** step)) / (np.sqrt(var / (1 - b2 ** step)) + 1e-8)
        if ep % 2 == 0 or ep == 1:
            hist.append(report(f"epoch {ep}/{EPOCHS} ({time.time() - t0:.0f}s)"))
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, NAME + ".npy"), Wd.astype(np.float32))
    d = Wd - W0
    json.dump({"name": "Frost 1.0", "base": "Qwen2.5-3B-Instruct Q4_K_M (dequantized float32)", "edited_tensor": NAME, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
               "steps": step, "positions": int(len(T)), "delta_rms": float(np.sqrt((d * d).mean())), "weight_rms": float(np.sqrt((W0 * W0).mean())),
               "fraction_changed_rel_1pct": float((np.abs(d) > 0.01 * (np.abs(W0) + 1e-8)).mean()), "history": hist,
               "trained": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(OUT, "info.json"), "w", encoding="utf-8"), indent=1)
    print(f"saved patch to {OUT}  | change size (rms) {float(np.sqrt((d * d).mean())):.6f} vs weight size {float(np.sqrt((W0 * W0).mean())):.6f}", flush=True)


if __name__ == "__main__":
    main()
