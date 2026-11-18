"""Interactive text chat with the model.  Commands: /reset  /temp 0.3  /quit"""
import sys, time
from engine import Model, chatml

m = Model()
sysmsg = "You are Frost 1.0, a helpful assistant."
history = [("system", sysmsg)]
temp = 0.7
print("Chat ready. /reset clears history, /temp N sets temperature (0 = greedy), /quit exits.\n")
while True:
    try:
        u = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not u:
        continue
    if u == "/quit":
        break
    if u == "/reset":
        history = [("system", sysmsg)]; print("(history cleared)"); continue
    if u.startswith("/temp"):
        temp = float(u.split()[1]); print(f"(temp={temp})"); continue
    history.append(("user", u))
    ids = m.tok.encode(chatml(history))          # re-prefill each turn: simple, and keeps experiments stateless
    t0, out, n = time.time(), "", 0
    print("qwen> ", end="", flush=True)
    for _, piece in m.generate(ids, max_new=300, temp=temp):
        print(piece, end="", flush=True); out += piece; n += 1
    print(f"\n[{n} tokens, {n / (time.time() - t0):.2f} tok/s incl. prefill]\n")
    history.append(("assistant", out))
