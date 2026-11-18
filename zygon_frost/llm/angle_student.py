"""The "exponent + angle wheel" student that learns from the 3B model's answers while they arrive.

Every token is an independent identity: id -> exponent x = ln(id+1), written as the text  e^x  (decodes back exactly).
Every context (the last 0-3 tokens) owns a WHEEL: a 360 degree circle cut into sectors, one sector per next token, sector width = how often that
token followed this context. To pick the next word the wheel is turned to the widest sector.  Wheels are plain text, no matrices anywhere.
The prompt's own content words add a small pull towards the tokens that replies to those words used before.

    python angle_student.py   -> reads model/exp_store/teacher.jsonl, writes angle_student_report.html and model/exp_store/student_store.txt
Evaluation is "prequential": for every prompt, in the order the teacher produced them, the student first tries to answer it, THEN learns the teacher's answer.
"""
import html, json, math, os, sys
from collections import Counter
import path_locker as PL
from angle_ai import wsim

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXP = os.path.join(ROOT, "model", "exp_store")
END = 151645                                         # <|im_end|>
BOS = -1


def f1(a, b):
    a, b = [t for t in a if t != END], [t for t in b if t != END]
    if not a or not b: return 0.0
    ov = sum((Counter(a) & Counter(b)).values())
    if ov == 0: return 0.0
    p, r = ov / len(a), ov / len(b); return 2 * p * r / (p + r)


class AngleStudent:
    def __init__(self, K=3):
        self.K = K
        self.wheels = {}                  # context (tuple of token ids) -> {next token: count}
        self.pw, self.pw_n = {}, {}       # prompt word -> {reply token: number of replies that used it}, number of replies seen
        self.mem = []                     # (content words, reply ids): for exact repeats

    def learn(self, prompt, reply_ids):
        words = set(PL.content_words(prompt))
        for w in words:
            self.pw_n[w] = self.pw_n.get(w, 0) + 1; d = self.pw.setdefault(w, {})
            for t in set(reply_ids): d[t] = d.get(t, 0) + 1
        hist = [BOS] * self.K
        for t in reply_ids:
            for k in range(self.K + 1):
                ctx = tuple(hist[len(hist) - k:]) if k else ()
                w = self.wheels.setdefault(ctx, {}); w[t] = w.get(t, 0) + 1
            hist.append(t)
        self.mem.append((frozenset(words), list(reply_ids)))

    def nearest(self, prompt):
        ws = frozenset(PL.content_words(prompt)); best, bs = None, 0.0
        n = max(1, len(self.mem))
        for k, (w, r) in enumerate(self.mem):
            s = wsim(ws, w, lambda x: self.pw_n.get(x, 0), n)
            if s > bs: bs, best = s, k
        return best, bs

    def turn_wheel(self, hist, words):
        """spin: scores for every candidate next token (back-off over the contexts, plus the prompt-word pull)"""
        cand = {}
        for k in range(self.K, -1, -1):
            ctx = tuple(hist[len(hist) - k:]) if k else ()
            w = self.wheels.get(ctx)
            if w:
                tot = sum(w.values()); f = 0.4 ** (self.K - k)
                for u, c in w.items(): cand[u] = max(cand.get(u, 0.0), f * c / tot)
        if not cand: return cand
        known = [w for w in words if w in self.pw_n]
        for u in cand:
            pulls = sorted((self.pw[w].get(u, 0) / self.pw_n[w] for w in known), reverse=True)[:2]
            cand[u] *= 0.1 + (sum(pulls) / len(pulls) if pulls else 0.0)
        return cand

    def generate(self, prompt, max_new=60):
        words = PL.content_words(prompt); hist = [BOS] * self.K; out, confs = [], []
        for _ in range(max_new):
            cand = self.turn_wheel(hist, words)
            if not cand: break
            u = max(cand, key=cand.get); tot = sum(cand.values()); confs.append(cand[u] / tot if tot else 0.0)
            if u == END: break
            out.append(u); hist.append(u)
        known = sum(w in self.pw_n for w in words)
        return out, (sum(confs) / len(confs) if confs and known else 0.0)

    def next_acc(self, prompt, reply_ids):
        """teacher forcing: how often the wheel's choice equals the teacher's real next token"""
        words = PL.content_words(prompt); hist = [BOS] * self.K; hit = 0
        for t in reply_ids:
            cand = self.turn_wheel(hist, words)
            if cand and max(cand, key=cand.get) == t: hit += 1
            hist.append(t)
        return hit, len(reply_ids)


def exp_text(t):
    return f"e^{math.log(t + 1):.6f}"


def exp_decode(s):
    return round(math.exp(float(s[2:]))) - 1


def main():
    recs = [json.loads(l) for l in open(os.path.join(EXP, "teacher.jsonl"), encoding="utf-8") if l.strip()]
    N = len(recs)
    print(f"{N} teacher answers; {sum(len(r['reply_ids']) for r in recs)} tokens; teacher time {sum(r['seconds'] for r in recs) / 60:.1f} min", flush=True)
    from gguf import GGUF
    from tokenizer import Tokenizer
    tok = Tokenizer(GGUF().meta); dec = lambda ids: tok.decode([t for t in ids if t != END])
    # ---------------------------------------------------------------- prequential run
    S = AngleStudent(3); rows = []
    for i, r in enumerate(recs):
        ref = [t for t in r["reply_ids"] if t != END]
        near, sim = S.nearest(r["prompt"]); lock_f1 = f1(S.mem[near][1], ref) if near is not None and sim >= 0.99 else None
        gen, conf = S.generate(r["prompt"], 60)
        rows.append({"i": i, "prompt": r["prompt"], "topic": r["topic"], "teacher": r["reply"], "student": dec(gen), "f1_wheel": f1(gen, ref), "conf": conf, "sim": sim, "f1_lock": lock_f1,
                     "topic_seen": any(x["topic"] == r["topic"] for x in recs[:i]),
                     "unknown_words": len([w for w in PL.content_words(r["prompt"]) if w not in S.pw_n]),
                     "f1_near": f1(S.mem[near][1], ref) if near is not None else None,
                     "key_ok": (lambda ws_: bool(ws_) and near is not None and max(ws_, key=lambda w: math.log((len(S.mem) + 1) / (S.pw_n.get(w, 0) + 0.5))) in S.mem[near][0])(PL.content_words(r["prompt"]))})
        S.learn(r["prompt"], r["reply_ids"])
    # ---------------------------------------------------------------- results
    res = {}
    q = max(1, N // 4)
    res["by_quarter"] = [{"q": k + 1, "wheel_f1": sum(x["f1_wheel"] for x in rows[k * q:(k + 1) * q if k < 3 else N]) / max(1, len(rows[k * q:(k + 1) * q if k < 3 else N])),
                          "lock_share": sum(x["f1_lock"] is not None for x in rows[k * q:(k + 1) * q if k < 3 else N]) / max(1, len(rows[k * q:(k + 1) * q if k < 3 else N]))} for k in range(4)]
    casc = []
    for th in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        served, qual = 0, 0.0
        for x in rows:
            if x["f1_lock"] is not None: served += 1; qual += x["f1_lock"]
            elif x["conf"] >= th and x["conf"] > 0: served += 1; qual += x["f1_wheel"]
        casc.append({"threshold": th, "served_without_teacher": served / N, "mean_f1_of_served": (qual / served) if served else None,
                     "mean_f1_overall_if_teacher_fills_rest": (qual + (N - served)) / N})
    res["cascade"] = casc
    casc2 = []                                                         # the rule used in the live AI: a word never seen before always goes to the 3B model
    for th in (0.0, 0.5, 0.7, 0.8, 0.9, 1.01):
        served, qual = 0, 0.0
        for x in rows:
            if x["f1_lock"] is not None: served += 1; qual += x["f1_lock"]
            elif x["unknown_words"] == 0 and x["conf"] >= th and x["conf"] > 0: served += 1; qual += x["f1_wheel"]
        casc2.append({"threshold": th, "served_without_teacher": served / N, "mean_f1_of_served": (qual / served) if served else None, "mean_f1_overall_if_teacher_fills_rest": (qual + (N - served)) / N})
    res["cascade_new_word_rule"] = casc2
    rep = []                                                           # the live AI: follow the nearest learned question if its words overlap enough, else ask the 3B model
    for tau in (0.34, 0.5, 0.67, 0.99):
        sv_ = [x for x in rows if x["f1_near"] is not None and x["sim"] >= tau and x["unknown_words"] == 0 and x["key_ok"]]
        rep.append({"min_word_overlap": tau, "served_without_teacher": len(sv_) / N, "mean_f1_of_served": (sum(x["f1_near"] for x in sv_) / len(sv_)) if sv_ else None,
                    "mean_f1_overall_if_teacher_fills_rest": (sum(x["f1_near"] for x in sv_) + (N - len(sv_))) / N})
    res["replay_nearest"] = rep
    res["lock_only"] = {"share": sum(x["f1_lock"] is not None for x in rows) / N, "mean_f1": (lambda v: sum(v) / len(v) if v else None)([x["f1_lock"] for x in rows if x["f1_lock"] is not None])}
    seen = [x for x in rows if x["topic_seen"]]; unseen = [x for x in rows if not x["topic_seen"]]
    res["seen_vs_new_topic"] = {"seen_topic_n": len(seen), "seen_topic_f1": sum(x["f1_wheel"] for x in seen) / max(1, len(seen)), "new_topic_n": len(unseen), "new_topic_f1": sum(x["f1_wheel"] for x in unseen) / max(1, len(unseen))}
    # teacher-forced next-token accuracy: train on the first 2/3, test on the last 1/3
    cut = int(N * 2 / 3); S2 = AngleStudent(3)
    for r in recs[:cut]: S2.learn(r["prompt"], r["reply_ids"])
    train_topics = {r["topic"] for r in recs[:cut]}; acc = {"seen": [0, 0], "new": [0, 0]}
    for r in recs[cut:]:
        h, n_ = S2.next_acc(r["prompt"], r["reply_ids"]); k = "seen" if r["topic"] in train_topics else "new"; acc[k][0] += h; acc[k][1] += n_
    res["next_token_accuracy"] = {k: {"hit": v[0], "tokens": v[1], "acc": v[0] / v[1] if v[1] else None} for k, v in acc.items()}
    # ---------------------------------------------------------------- the text store (exponents + angles)
    lines = ["# ANGLE-WHEEL STORE.  token identity: id -> e^x with x = ln(id+1); decode: id = round(e^x) - 1.   line: context tokens => next token @ sector start-end (degrees of the 360 degree wheel)"]
    used = set(); big = sorted(S.wheels.items(), key=lambda kv: -sum(kv[1].values()))
    for ctx, w in big:
        tot = sum(w.values()); a0 = 0.0; secs = []
        for u, c in sorted(w.items(), key=lambda kv: -kv[1]):
            a1 = a0 + 360.0 * c / tot; secs.append(f"{exp_text(u)}@{a0:.2f}-{a1:.2f}"); a0 = a1; used.add(u)
        used.update(t for t in ctx if t != BOS)
        lines.append((",".join(exp_text(t) if t != BOS else "START" for t in ctx) or "ANY") + " => " + " ".join(secs))
    store = "\n".join(lines); open(os.path.join(EXP, "student_store.txt"), "w", encoding="utf-8").write(store)
    assert all(exp_decode(exp_text(t)) == t for t in used), "exponent identity does not decode exactly"
    res["store"] = {"wheels": len(S.wheels), "wheel_sectors": sum(len(w) for w in S.wheels.values()), "file_bytes": len(store.encode()), "ids_decode_exactly": True, "distinct_tokens": len(used),
                    "teacher_reply_tokens": sum(len(r["reply_ids"]) for r in recs), "teacher_reply_text_bytes": sum(len(r["reply"].encode()) for r in recs), "teacher_minutes": round(sum(r["seconds"] for r in recs) / 60, 1)}
    wheels_view = []
    for ctx, w in big[:60]:
        if len(ctx) < 1 or all(t == BOS for t in ctx): continue
        tot = sum(w.values()); wheels_view.append({"ctx": " ".join(("[start]" if t == BOS else repr(dec([t]))) for t in ctx), "n": tot, "next": [[repr(dec([u])) if u != END else "[end]", exp_text(u), c / tot] for u, c in sorted(w.items(), key=lambda kv: -kv[1])[:8]]})
    start = S.wheels.get(tuple([BOS] * 3)); wheels_view.insert(0, {"ctx": "[start] [start] [start]  (first word of a reply)", "n": sum(start.values()), "next": [[repr(dec([u])) if u != END else "[end]", exp_text(u), c / sum(start.values())] for u, c in sorted(start.items(), key=lambda kv: -kv[1])[:8]]})
    json.dump({"rows": rows, "res": res}, open(os.path.join(EXP, "student_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    page(rows, res, wheels_view, lines[:9])
    print(json.dumps(res, indent=1), flush=True)


def page(rows, res, wheels_view, excerpt):
    N = len(rows); W, H = 560, 200
    # learning curve: running mean (window 8) of the student's first-try F1, and the share answered from an exact repeat
    def run(vals, w=8): return [sum(vals[max(0, i - w + 1):i + 1]) / len(vals[max(0, i - w + 1):i + 1]) for i in range(len(vals))]
    a = run([x["f1_wheel"] for x in rows]); b = run([1.0 if x["f1_lock"] is not None else 0.0 for x in rows])
    pts = lambda v: " ".join(f"{20 + i * (W - 40) / max(1, N - 1):.1f},{H - 20 - y * (H - 40):.1f}" for i, y in enumerate(v))
    svg = (f"<svg width='{W}' height='{H}' style='background:#fff;border:1px solid #999'><polyline fill='none' stroke='#0A6C80' stroke-width='2' points='{pts(a)}'/><polyline fill='none' stroke='#c0392b' stroke-width='2' points='{pts(b)}'/>"
           f"<text x='24' y='14' font-size='11' fill='#0A6C80'>student answer vs teacher answer, word overlap (running average of 8)</text><text x='24' y='{H - 4}' font-size='11' fill='#c0392b'>red: share of prompts that were an exact repeat of something already learned</text>"
           f"<text x='{W - 150}' y='{H - 24}' font-size='11'>interaction 1 ... {N}</text></svg>")
    q = res["by_quarter"]; cs = res["cascade"]; nt = res["next_token_accuracy"]; sv = res["seen_vs_new_topic"]; st = res["store"]; lo = res["lock_only"]
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    esc = html.escape
    samples = "".join(f"<tr><td>{x['i'] + 1}</td><td>{esc(x['prompt'])}</td><td>{esc(x['teacher'][:150])}</td><td>{esc(x['student'][:150]) or '(nothing)'}</td><td>{pc(x['f1_wheel'])}</td><td>{'exact repeat' if x['f1_lock'] is not None else ('known topic' if x['topic_seen'] else 'new topic')}</td></tr>" for x in rows[-14:])
    wv = json.dumps(wheels_view, ensure_ascii=False)
    verdict = [
        f"The student never saw an answer before it tried: for every prompt it answered first, then learned the teacher's answer. Overlap with the teacher's answer rose from {pc(q[0]['wheel_f1'])} in the first quarter to {pc(q[3]['wheel_f1'])} in the last quarter.",
        f"Prompts that were exact repeats of something learned (same content words) are answered from memory: {pc(lo['share'])} of all prompts, with {pc(lo['mean_f1'])} overlap with what the teacher would have said.",
        f"A topic with other wordings learned: wheel answer overlap {pc(sv['seen_topic_f1'])} ({sv['seen_topic_n']} prompts). A topic never learned: {pc(sv['new_topic_f1'])} ({sv['new_topic_n']} prompts). So the wheels re-use wording they have seen, but they cannot invent facts: for a new topic the answer has to come from the 3B model.",
        f"Next word right when the wheel is given the true history: {pc(nt['seen']['acc'])} for topics it had learned, {pc(nt['new']['acc'])} for topics it had not (trained on the first two thirds of the answers, tested on the last third).",
        f"Memory: {st['wheels']:,} wheels with {st['wheel_sectors']:,} sectors, {st['file_bytes']:,} bytes as text, against {st['teacher_reply_text_bytes']:,} bytes of the teacher's reply text. The token identities as e^x decode back exactly."]
    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>ZYGON student</title><style>
body{{font:14px/1.5 Consolas,monospace;margin:16px;max-width:1150px;background:#fff;color:#111}}h2{{font:700 17px system-ui;margin:22px 0 6px;border-bottom:1px solid #999}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #999;padding:3px 7px;vertical-align:top;text-align:left}}th{{background:#e6f2f4}}
pre{{background:#f4f4f4;padding:8px;overflow-x:auto;font-size:12px}}canvas{{background:#fff;border:1px solid #999}}.n{{text-align:right}}</style></head><body>
<h1 style="font:700 20px system-ui">ZYGON student: exponent identities and angle wheels, learning from the 3B model</h1>
<p>Teacher: the original 3B model (greedy answers, up to 60 tokens each), {N} prompts = 12 topics x 4 wordings, shuffled; {st['teacher_reply_tokens']:,} reply tokens recorded in {st['teacher_minutes']} minutes. Student: wheels only, no matrices.</p>
<h2>1. What the student is</h2>
<p>Every token is its own identity: id -> e^x with x = ln(id+1), stored as text and decoded exactly back (checked for all {st['distinct_tokens']:,} tokens used). Every context (the last 0 to 3 tokens) owns a wheel of 360 degrees cut into sectors, one per next token; sector width = how often that token followed. Choosing the next word = turning the wheel to the widest sector. The prompt's own content words add a pull towards tokens that earlier replies to those words used.</p>
<pre>{esc(chr(10).join(excerpt))}</pre>
<h2>2. Learning while interacting (answer first, learn after)</h2>{svg}
<table><tr><th>quarter of the interactions</th><th>student answer overlap with teacher</th><th>prompts that were exact repeats</th></tr>{''.join(f"<tr><td>{r['q']}</td><td class='n'>{pc(r['wheel_f1'])}</td><td class='n'>{pc(r['lock_share'])}</td></tr>" for r in q)}</table>
<h2>3. Student in front of the 3B model: how often can the teacher be skipped?</h2>
<p>Rule: an exact repeat is answered from memory; otherwise the student answers if the winning sector of its wheels is at least this share of the circle; below that the 3B model is asked (and counts as perfect).</p>
<table><tr><th>student must be at least this sure</th><th>prompts answered without the 3B model</th><th>quality of those answers</th><th>quality overall</th></tr>{''.join(f"<tr><td class='n'>{c['threshold']:.1f}</td><td class='n'>{pc(c['served_without_teacher'])}</td><td class='n'>{pc(c['mean_f1_of_served'])}</td><td class='n'>{pc(c['mean_f1_overall_if_teacher_fills_rest'])}</td></tr>" for c in cs)}</table>
<p>Same, with the rule used in the live AI: if the question contains a word the student has never seen, the 3B model is asked.</p>
<table><tr><th>student must be at least this sure</th><th>prompts answered without the 3B model</th><th>quality of those answers</th><th>quality overall</th></tr>{''.join(f"<tr><td class='n'>{c['threshold']:.1f}</td><td class='n'>{pc(c['served_without_teacher'])}</td><td class='n'>{pc(c['mean_f1_of_served'])}</td><td class='n'>{pc(c['mean_f1_overall_if_teacher_fills_rest'])}</td></tr>" for c in res["cascade_new_word_rule"])}</table>
<p>The rule now used in the live AI: follow the nearest learned question when its content words overlap at least this much, the main (rarest) word of the question is in it, and no word is new (the new wording is hooked to it); otherwise ask the 3B model.</p>
<table><tr><th>minimum word overlap</th><th>prompts answered without the 3B model</th><th>quality of those answers</th><th>quality overall</th></tr>{''.join(f"<tr><td class='n'>{c['min_word_overlap']:.2f}</td><td class='n'>{pc(c['served_without_teacher'])}</td><td class='n'>{pc(c['mean_f1_of_served'])}</td><td class='n'>{pc(c['mean_f1_overall_if_teacher_fills_rest'])}</td></tr>" for c in res["replay_nearest"])}</table>
<h2>4. Last prompts: teacher vs student (the student answered before learning each one)</h2>
<table><tr><th>#</th><th>prompt</th><th>3B model</th><th>student</th><th>overlap</th><th>kind</th></tr>{samples}</table>
<h2>5. Look at a wheel</h2><p>Pick a context. The pointer turns to the widest sector.</p>
<select id="sel"></select> <button id="spin">Turn the wheel</button><br><canvas id="cv" width="420" height="420"></canvas><div id="leg"></div>
<h2>6. Verdict</h2><ul>{''.join(f'<li>{esc(v)}</li>' for v in verdict)}</ul>
<script>
const WV={wv};const sel=document.getElementById('sel'),cv=document.getElementById('cv'),cx=cv.getContext('2d');
WV.forEach((w,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=w.ctx+'  ('+w.n+' times)';sel.appendChild(o);}});
const COL=['#0A6C80','#c0392b','#e6a817','#6a4c93','#2e8b57','#d35400','#7f8c8d','#2980b9'];let rot=0,target=null;
function draw(){{const w=WV[sel.value];cx.clearRect(0,0,420,420);let a=-Math.PI/2+rot;const tot=w.next.reduce((s,x)=>s+x[2],0);let acc=0;
w.next.forEach((x,i)=>{{const sw=x[2]/tot*Math.PI*2;cx.beginPath();cx.moveTo(210,210);cx.arc(210,210,170,-Math.PI/2+acc,-Math.PI/2+acc+sw);cx.closePath();cx.fillStyle=COL[i%8];cx.globalAlpha=.8;cx.fill();cx.globalAlpha=1;cx.strokeStyle='#fff';cx.stroke();
const m=-Math.PI/2+acc+sw/2;if(sw>.25){{cx.fillStyle='#fff';cx.font='12px Consolas';cx.textAlign='center';cx.fillText(x[0],210+Math.cos(m)*110,210+Math.sin(m)*110);}}acc+=sw;}});
const ang=-Math.PI/2+rot;cx.beginPath();cx.moveTo(210,210);cx.lineTo(210+Math.cos(ang)*185,210+Math.sin(ang)*185);cx.strokeStyle='#000';cx.lineWidth=3;cx.stroke();cx.lineWidth=1;
document.getElementById('leg').innerHTML='<table><tr><th>next</th><th>identity</th><th>share</th><th>sector (degrees)</th></tr>'+(()=>{{let c=0;return w.next.map(x=>{{const r='<tr><td>'+x[0]+'</td><td>'+x[1]+'</td><td>'+(100*x[2]).toFixed(1)+'%</td><td>'+c.toFixed(1)+' - '+(c+360*x[2]).toFixed(1)+'</td></tr>';c+=360*x[2];return r;}}).join('')}})()+'</table>';}}
function spin(){{const w=WV[sel.value];const mid=(w.next[0][2]*Math.PI*2)/2;target=mid+Math.PI*4;rot=0;const t0=performance.now();(function f(t){{const u=Math.min(1,(t-t0)/1500);rot=target*(1-Math.pow(1-u,3));draw();if(u<1)requestAnimationFrame(f);}})(t0);}}
sel.onchange=()=>{{rot=0;draw();}};document.getElementById('spin').onclick=spin;draw();
</script></body></html>"""
    open(os.path.join(ROOT, "angle_student_report.html"), "w", encoding="utf-8").write(out)
    print("written angle_student_report.html", flush=True)


if __name__ == "__main__":
    main()
