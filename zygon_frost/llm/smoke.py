import time, sys
sys.stdout.reconfigure(encoding="utf-8")
from engine import Model, chatml
m = Model()
t = m.tok
for s in ["Hello, world!", "The capital of France is", "naïve café 日本語 123\n\nok"]:
    ids = t.encode(s); assert t.decode(ids) == s, s; print(ids[:8], "roundtrip ok")
print(t.encode("<|im_start|>user\nhi<|im_end|>"))
ids = t.encode(chatml([("system","You are a helpful assistant."),("user","What is the capital of France? Answer in one sentence.")]))
print(len(ids), "prompt tokens")
t0=time.time(); out=""; n=0
for _, p in m.generate(ids, max_new=40, temp=0):
    out+=p; n+=1
print(repr(out)); print(f"{n} tokens in {time.time()-t0:.1f}s")
