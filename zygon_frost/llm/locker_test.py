"""End-to-end test of the Neuron Path Locker on the real model (greedy, so results are comparable).  Uses a throw-away locker folder."""
import os, shutil, sys, time, json
import numpy as np
from engine import Model, chatml
import path_locker as PL

sys.stdout.reconfigure(encoding="utf-8")
TMP = os.path.join(PL.ROOT, "model", "path_locker_test")
shutil.rmtree(TMP, ignore_errors=True)
SYSTEM = "You are Frost 1.0, a helpful assistant."
KW = dict(temp=0)


def ids_for(m, ask):
    return m.tok.encode(chatml([("system", SYSTEM), ("user", ask)]))


def run(m, ask, follower=None, max_new=110):
    ids = ids_for(m, ask)
    cache = m.new_cache(len(ids) + max_new + 20)
    t0 = time.time()
    logits = m.forward(ids, cache)
    t_pre = time.time() - t0
    out, flags = [], []
    t1 = time.time()
    if follower is None:
        for _ in range(max_new):
            t = m.sample(logits, **KW)
            if t in (m.tok.eos, m.tok.id_of["<|im_end|>"]): break
            out.append(t); flags.append(False)
            logits = m.forward([t], cache)
        passes = len(out)
    else:
        for t, on in PL.follow_generate(m, logits, cache, follower, max_new, KW):
            out.append(t); flags.append(on)
        passes = follower.passes
    return out, flags, t_pre, time.time() - t1, passes, ids


def branches(flags):
    b, i = [], 0
    while i < len(flags):
        if not flags[i]:
            j = i
            while j < len(flags) and not flags[j]: j += 1
            b.append({"start": i, "end": j}); i = j
        else: i += 1
    return b


def main():
    t0 = time.time(); m = Model(verbose=False); print(f"model loaded {time.time() - t0:.0f}s", flush=True)
    LK = PL.Locker(TMP, max_records=5, max_mb=50)
    A = "Write a tiny HTML page with one blue button that says Click me."
    B = "Write a tiny HTML page with one red button that says Click me."
    # 1. new path (no help)
    outA, fA, pre, dec, passes, idsA = run(m, A)
    txtA = m.tok.decode(outA); print(f"[A new] {len(outA)} tokens, prefill {pre:.1f}s, decode {dec:.1f}s ({passes} passes)\n{txtA}\n", flush=True)
    rid = LK.add(A, "", outA, txtA, PL.ask_vector(m, A), branches=[{"start": 0, "end": len(outA)}])
    LK.approve(rid)
    # 2. lock hit
    t = time.time(); key = PL.ask_key("please - write a TINY html page with one blue button that says 'Click me'"); hit = LK.find_exact(key, "")
    if hit: LK.touch(hit, replayed=len(outA)); got = LK.rec[hit]["reply"]
    ms = (time.time() - t) * 1000
    print(f"[A again, different wording] key={key!r} hit={hit == rid} reply_identical={got == txtA} in {ms:.1f} ms\n", flush=True)
    # 3. modified ask: plain vs following the locked path
    outB0, _, pre0, dec0, p0, _ = run(m, B)
    sim = LK.near(PL.ask_key(B), PL.ask_vector(m, B), "")
    print("near paths for B:", sim, flush=True)
    fol = PL.PathFollower(LK.corpus([r for _, r in sim]), K=12)
    outB1, fB, pre1, dec1, p1, _ = run(m, B, fol)
    br = branches(fB)
    print(f"[B plain ] {len(outB0)} tokens, decode {dec0:.1f}s, {p0} passes")
    print(f"[B follow] {len(outB1)} tokens, decode {dec1:.1f}s, {p1} passes, on-path tokens {sum(fB)}, branch segments {br}")
    print("B identical to plain decode:", outB0 == outB1, "| first difference at", next((i for i, (a, b) in enumerate(zip(outB0, outB1)) if a != b), None))
    print("speed-up in decode time: %.2fx" % (dec0 / max(dec1, 1e-9)), flush=True)
    print(m.tok.decode(outB1), "\n", flush=True)
    rb = LK.add(B, "", outB1, m.tok.decode(outB1), PL.ask_vector(m, B), parent=rid, branches=br)
    LK.approve(rb)
    # 4. fingerprints
    t = time.time(); LK.fingerprint(m, idsA, rid); LK.fingerprint(m, ids_for(m, B), rb)
    print(f"fingerprints for 2 paths: {time.time() - t:.0f}s; neurons ever triggered A={LK.rec[rid]['neurons_triggered_ever']:,} B={LK.rec[rb]['neurons_triggered_ever']:,}")
    print("top-48-per-layer overlap A vs B:", round(LK.fp_overlap(rid, rb), 3), "(chance 0.004)")
    z = np.load(LK._npz(rb)); print("branch-only fingerprint present:", "br_ids" in z.files, flush=True)
    # 5. eviction: limit is 5 records
    for i in range(7):
        r = LK.add(f"filler ask number {i}", "", [1, 2, 3], "x", None)
    print("records after adding 7 fillers with max 5:", len(LK.rec), "| evicted:", LK.stats["evicted"], "| locked A and B still there:", rid in LK.rec, rb in LK.rec)
    shutil.rmtree(TMP, ignore_errors=True)


main()
