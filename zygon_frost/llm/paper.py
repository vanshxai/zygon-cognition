"""Worked-by-hand maths on ruled paper (LaTeX, pen-blue handwriting) from the recorded numbers of board.py (model/frost_work/board_data.json).
    python paper.py   ->  capital_of_france_paper.html     (no model needed)
"""
import html, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
D = json.load(open(os.path.join(ROOT, "model", "frost_work", "board_data.json"), encoding="utf-8"))
DIM = D["dims"]; T = D["T"]


def n(x):
    if x == 0: return "0"
    a = abs(x)
    if a >= 1e5 or a < 1e-3:
        m, e = f"{x:.3e}".split("e"); return f"{float(m):.4g}\\times10^{{{int(e)}}}"
    return f"{x:.4g}"


def p(x):
    return f"({n(x)})" if x < 0 else n(x)


def tx(s):
    """a token piece as LaTeX text: quoted, spaces shown as _, newline as \\n"""
    s = s.replace("\\", "\x00").replace("\n", "\x01")
    for a, b in (("{", "\\{"), ("}", "\\}"), ("_", "\\_"), ("#", "\\#"), ("%", "\\%"), ("&", "\\&"), ("$", "\\$"), ("~", "\\~{}"), ("^", "\\^{}"), (" ", "\\_")):
        s = s.replace(a, b)
    return "\\text{``" + s.replace("\x00", "\\backslash ").replace("\x01", "\\backslash n") + "''}"


def bm(rows):
    return "\\begin{bmatrix}" + " \\\\ ".join(" & ".join(r) for r in rows) + "\\end{bmatrix}"


def col(vals, name=None, tail=True):
    r = [[v] for v in vals] + ([["\\vdots"]] if tail else [])
    return bm(r)


def disp(s):
    return "<div class='d'>$$" + html.escape(s, quote=False) + "$$</div>"


def say(s):
    return "<p class='w'>" + html.escape(s) + "</p>"


def receipt(sym, idx, wsym, vsym, rc, bias=True):
    """sum of 2048 products written out with the real numbers"""
    t = rc["terms"]
    first = " + ".join(f"{p(a)}\\cdot{p(b)}" for a, b, _ in t[:3]); second = " + ".join(f"{p(a)}\\cdot{p(b)}" for a, b, _ in t[3:])
    prods = " + ".join(p(c) for _, _, c in t)
    return disp(f"{sym}_{{{idx}}}=\\sum_{{j=0}}^{{{rc['n'] - 1}}} {wsym}_{{{idx},j}}\\,{vsym}_j" + (f"+b_{{{idx}}}" if bias else "")) + disp(
        "\\begin{aligned}" + f"{sym}_{{{idx}}} &= {first} \\\\ &\\quad + {second} + \\cdots \\\\ &= {prods} + \\underbrace{{{p(rc['rest'])}}}_{{\\text{{the other {rc['n'] - 6} terms}}}}"
        + (f" + \\underbrace{{{p(rc['bias'])}}}_{{b_{{{idx}}}}}" if bias else "") + f" \\\\ &= \\mathbf{{{p(rc['out'])}}}" + "\\end{aligned}")


def rowvec(vals):
    return bm([[p(v) for v in vals] + ["\\cdots"]])


def norm_block(tag, nrm, gname):
    e = nrm["e"]
    lines = " \\\\ ".join(f"{tag}_{{{i}}} &= \\frac{{{p(a)}}}{{{n(nrm['rms'])}}}\\cdot{p(g)} = {p(h)}" for i, (a, g, h) in enumerate(e))
    return (disp(f"\\mathrm{{ms}}=\\frac{{1}}{{{DIM['dim']}}}\\sum_i x_i^2={n(nrm['ms'])},\\qquad \\mathrm{{rms}}=\\sqrt{{\\mathrm{{ms}}+\\varepsilon}}=\\sqrt{{{n(nrm['ms'])}+{n(nrm.get('eps', DIM['eps']))}}}={n(nrm['rms'])}")
            + disp(f"{tag}_i=\\frac{{x_i}}{{\\mathrm{{rms}}}}\\,g_i\\quad(g=\\texttt{{{gname}}})") + disp("\\begin{aligned}" + lines + "\\end{aligned}")
            + disp("h=" + col([p(h) for _, _, h in e]) + "\\ \\text{(first 5 of 2048)}"))


def mm_block(M, wname, hname, oname, rowtag, ctx_note):
    """four real output numbers of one matrix multiplication: weights x input = result"""
    W_ = bm([[p(v) for v in r] + ["\\cdots"] for r in M["W"]]); H_ = col([p(v) for v in M["h"]])
    o = [say(ctx_note)]
    o.append(disp(f"\\underbrace{{{wname}_{{[0:4],\\,[0:8]}}}}_{{\\text{{weights: 4 rows, first 8 of 2048 columns}}}}" + "=" + W_ + "\\qquad \\underbrace{" + hname + "_{[0:8]}}_{\\text{input, first 8 of 2048}}=" + H_))
    o.append(disp("\\text{using only those 8 terms of each row:}\\quad " + bm([[p(v)] for v in M["part"]]) + "\\ (\\text{partial sums})"))
    o.append(disp(bm([[p(v)] for v in M["part"]]) + "+" + bm([[p(v)] for v in M["rest"]]) + ("+" + bm([[p(v)] for v in M["b"]]) if any(M["b"]) else "") + "="
                  + bm([[p(v)] for v in M["out"]]) + f"=\\begin{{bmatrix}}{oname}_0\\\\{oname}_1\\\\{oname}_2\\\\{oname}_3\\end{{bmatrix}}"))
    o.append(say("Left: the first 8 terms of each row. Next: the other 2040 terms of each row added together" + (", and the bias" if any(M["b"]) else "") + ". Right: the finished output numbers. Each weight in the table above was used once, in one product, for one output."))
    return "".join(o)


def before_after(L):
    v = L["v8"]; rows = [("x", "layer input", "x"), ("h", "after normalise (step 1)", "h1"), ("q", "query, head 0 (step 2)", "q"), ("k", "key, head 0 (step 2)", "k"), ("v", "value, head 0 (step 2)", "v"),
                         ("c", "mixed values (step 5)", "ctx"), ("a", "after W_o (step 5)", "ao"), ("x'", "x + a (step 6)", "xm"), ("h", "after normalise (step 7)", "h2"),
                         ("g", "gate of neurons 0-7 (step 8)", "g"), ("u", "up of neurons 0-7 (step 8)", "u"), ("a_n", "neuron strengths 0-7 (step 8)", "act"), ("m", "after W_down (step 9)", "mo"), ("x_{out}", "layer output = x' + m (step 9)", "xo")]
    body = " \\\\ ".join(f"{sym} & \\text{{{txt}}} & " + " & ".join(p(a) for a in v[key]) for sym, txt, key in rows)
    return say("Before and after, step by step: the first 8 numbers of the vector at every stage of this layer (each vector has 2048 numbers, the neuron ones 11008).") + disp("\\begin{array}{ll|" + "r" * 8 + "}\\text{vector} & \\text{stage} & \\multicolumn{8}{c}{\\text{numbers 0 to 7}}\\\\ \\hline " + body + "\\end{array}") + \
        "<h4>0. one real matrix multiplication of this layer, before and after</h4>" + mm_block(L["mm"], "W^{Q}", "h", "q", "Q", f"The query matrix W^Q of layer {L['l']} (2048 x 2048) times the normalised vector h (2048 numbers). Four of the 2048 outputs, with real numbers:")


def layer(L):
    l = L["l"]; o = [f"<details data-l='{l}'><summary>Layer {l}</summary><div class='ly'>"]
    o.append(say("The 9 steps of this layer, in order:") + disp("\\begin{array}{ll}\\text{1. normalise} & h=\\dfrac{x}{\\mathrm{rms}(x)}\\odot g\\\\ \\text{2. Q, K, V} & q=W^Qh+b^Q,\\ k=W^Kh+b^K,\\ v=W^Vh+b^V\\\\ \\text{3. RoPE} & q,k\\leftarrow R_{\\theta(\\text{position})}\\,q,\\ R_{\\theta}\\,k\\\\ \\text{4. attention weights} & p_j=\\mathrm{softmax}_j\\!\\left(q\\cdot k_j/\\sqrt{128}\\right)\\\\ \\text{5. mix values} & c=\\sum_j p_jv_j,\\ a=W^Oc\\\\ \\text{6. residual} & x\\leftarrow x+a\\\\ \\text{7. normalise} & h=\\dfrac{x}{\\mathrm{rms}(x)}\\odot g'\\\\ \\text{8. neurons} & m=W^D\\left(\\mathrm{SiLU}(W^Gh)\\odot W^Uh\\right)\\\\ \\text{9. residual} & x\\leftarrow x+m\\end{array}") + before_after(L) + "<h4>1. normalise the vector</h4>" + say(f"x is the 2048 numbers going into layer {l}. Rescale them, then multiply by the learned gain g (weights blk.{l}.attn_norm.weight)."))
    o.append(norm_block("h", L["norm1"], f"blk.{l}.attn\\_norm"))
    o.append("<h4>2. queries, keys, values: one output number each, with its exact weight row</h4>")
    for a, nm, tn in (("q", "Q", "attn_q"), ("k", "K", "attn_k"), ("v", "V", "attn_v")):
        e = L[a]; r = e["row"]
        o.append(say(f"{nm}: output number {r}, using row {r} of blk.{l}.{tn}.weight."))
        if a == "q":
            rows = [[p(v) for v in row] + ["\\cdots"] for row in e["blk"]]
            o.append(disp(f"W^{{Q}}_{{{e['blk0']}..{e['blk0'] + 4},\\ 0..7}}=" + bm(rows) + f"\\quad\\text{{(real weights; row {r} is the middle one)}}"))
        else:
            o.append(disp(f"W^{{{nm}}}_{{{r},\\,0..7}}=" + rowvec(e["w"])))
        o.append(receipt(a, r, f"W^{{{nm}}}", "h", e["rc"]))
    at = L["att"]
    o.append("<h4>3. rotate q and k by the token position (RoPE)</h4>" + say(f"Head {at['head']}, position {T - 1}. Each pair (x_1, x_2) turns by the angle theta = position times a frequency."))
    for r in at["rope"]:
        o.append(disp(f"\\theta={n(r['a'])}:\\quad" + bm([["\\cos\\theta", "-\\sin\\theta"], ["\\sin\\theta", "\\cos\\theta"]]) + bm([[p(r["x1"])], [p(r["x2"])]]) + "=" + bm([[p(r["r1"])], [p(r["r2"])]])))
    o.append("<h4>4. attention: how much does the last token look at each token?</h4>" + say(f"Head {at['head']} (the one looking hardest at a token other than the first). score_j = q . k_j / sqrt(128), then softmax."))
    qk = at["qk"]
    o.append(say(f"4a. the score of one token, written out: token {qk['j']} {D['pieces'][qk['j']]!r}, the one this head looks at most. q is the rotated query, k_j the rotated key of token j (128 numbers each)."))
    o.append(disp(f"s_{{{qk['j']}}}=\\frac{{q\\cdot k_{{{qk['j']}}}}}{{\\sqrt{{128}}}}=\\frac{{" + " + ".join(f"{p(a_)}\\cdot{p(b_)}" for a_, b_, _ in qk["terms"]) + f" + \\cdots}}{{11.31}}=\\frac{{" + " + ".join(p(c_) for _, _, c_ in qk["terms"]) + f" + \\underbrace{{{p(qk['rest'])}}}_{{\\text{{other {qk['n'] - 6} terms}}}}}}{{11.31}}=\\mathbf{{{p(qk['score'])}}}"))
    import math
    ex = [math.exp(v - max(at["scores"])) for v in at["scores"]]; Z = sum(ex)
    o.append(say("4b. the same for all tokens, then softmax: divide each e^(s - biggest s) by their total Z (the biggest score is subtracted first so nothing overflows)."))
    o.append(disp(f"p_j=\\frac{{e^{{s_j-s_{{\\max}}}}}}{{Z}},\\qquad s_{{\\max}}={p(max(at['scores']))},\\qquad Z=\\sum_k e^{{s_k-s_{{\\max}}}}={n(Z)}"))
    rows = " \\\\ ".join(f"{j} & {tx(D['pieces'][j])} & {p(sj)} & {n(e_)} & \\frac{{{n(e_)}}}{{{n(Z)}}}={n(e_ / Z)}" for j, (sj, e_) in enumerate(zip(at["scores"], ex)))
    o.append(disp("\\begin{array}{r|l|r|r|l} j & \\text{token} & s_j & e^{s_j-s_{\\max}} & p_j \\\\ \\hline " + rows + "\\end{array}"))
    big = max(range(T), key=lambda j: at["probs"][j])
    o.append(say(f"Biggest weight: token {big} {D['pieces'][big]!r} with p = {n(at['probs'][big])}. All {T} weights add up to 1."))
    mx = at["mix"]
    o.append("<h4>5. mix the values (weights times values), then W_o</h4>" + say(f"Each token j has a value vector v_j. This head's output is c = sum of p_j times v_j. Written out for output number {mx['d']} of this head (the 6 biggest weights, then the rest):"))
    o.append(disp(f"c_{{{mx['d']}}}=\\sum_{{j=0}}^{{{T - 1}}} p_j\\,v_{{j,{mx['d']}}}=" + " + ".join(f"{n(pj)}\\cdot{p(vj)}" for _, pj, vj, _ in mx["terms"]) + f" + \\cdots=" + " + ".join(p(c_) for *_, c_ in mx["terms"]) + f" + \\underbrace{{{p(mx['rest'])}}}_{{\\text{{other {T - 6} tokens}}}}=\\mathbf{{{p(mx['c'])}}}"))
    o.append(disp("a=W^{O}c\\quad(c=\\text{all 16 heads side by side, 2048 numbers})"))
    o.append(disp(f"W^{{O}}_{{{L['o']['row']},\\,0..7}}=" + rowvec(L["o"]["w"])) + receipt("a", L["o"]["row"], "W^{O}", "c", L["o"]["rc"], bias=False))
    o.append("<h4>6. add it to the original vector (residual)</h4>" + disp("x'=x+a:\\quad" + col([p(r[0]) for r in L["res1"]]) + "+" + col([p(r[1]) for r in L["res1"]]) + "=" + col([p(r[2]) for r in L["res1"]])))
    o.append("<h4>7. normalise again</h4>" + norm_block("h", L["norm2"], f"blk.{l}.ffn\\_norm"))
    o.append("<h4>8. feed-forward: 11008 neurons, shown for the three loudest</h4>" + say(f"Neuron n has a gate row and an up row. gate = W_gate[n] . h, up = W_up[n] . h, strength = SiLU(gate) * up, where SiLU(z) = z / (1 + e^-z)."))
    for i, nr in enumerate(L["neurons"]):
        g = nr["gate"]; s = g / (1 + __import__("math").exp(-g))
        o.append(disp(f"\\text{{neuron }}{nr['n']}:\\quad W^{{g}}_{{{nr['n']},0..7}}=" + rowvec(nr["wg"]) + f",\\quad W^{{u}}_{{{nr['n']},0..7}}=" + rowvec(nr["wu"])))
        if i == 0:
            o.append(receipt("g", nr["n"], "W^{g}", "h", nr["rg"], bias=False)); o.append(receipt("u", nr["n"], "W^{u}", "h", nr["ru"], bias=False))
        o.append(disp(f"a_{{{nr['n']}}}=\\mathrm{{SiLU}}({p(g)})\\cdot{p(nr['up'])}=\\frac{{{p(g)}}}{{1+e^{{{n(-g)}}}}}\\cdot{p(nr['up'])}={n(s)}\\cdot{p(nr['up'])}=\\mathbf{{{p(nr['act'])}}}"))
    o.append(say(f"{L['down']['active']} of {DIM['ffn']} neurons are at least 10% as strong as the loudest one."))
    o.append("<h4>9. down projection and residual</h4>" + disp(f"W^{{D}}_{{{L['down']['row']},\\,0..7}}=" + rowvec(L["down"]["w"])) + receipt("m", L["down"]["row"], "W^{D}", "a", L["down"]["rc"], bias=False))
    o.append(disp("x_{\\text{out}}=x'+m:\\quad" + col([p(r[0]) for r in L["res2"]]) + "+" + col([p(r[1]) for r in L["res2"]]) + "=" + col([p(r[2]) for r in L["res2"]])))
    o.append(say("If the model stopped here it would say: " + ",  ".join(f"{a!r} {100 * b:.1f}%" for a, b in L["lens"])))
    o.append("</div></details>")
    return "".join(o)


def hl(x):
    return "\\colorbox{#ffe27a}{$" + x + "$}"


def path_block(g, k):
    P = g["path"]; names, cw, cr = P["comp_names"], P["cw"], P["cr"]
    o = [f"<details><summary>Word {k + 1}: input id {g['input_id']} {g['input_piece']!r} \u2192 output id {g['id']} {g['piece']!r}  ({g['p'] * 100:.1f}%)</summary><div class='ly'>"]
    o.append(say(f"This trip started with token id {g['input_id']} ({g['input_piece']!r}), went through all 36 layers, and the winner is id {g['id']} ({g['piece']!r}) with probability {g['p'] * 100:.2f}%."))
    o.append(disp(f"{g['input_id']}\\xrightarrow{{E[{g['input_id']},:]}}x^{{(0)}}\\xrightarrow{{\\text{{layer 0}}}}x^{{(1)}}\\to\\cdots\\to x^{{(36)}}\\xrightarrow{{\\text{{norm}}}}h\\xrightarrow{{E\\cdot h}}z_0\\ldots z_{{{DIM['vocab'] - 1}}}\\xrightarrow{{\\text{{softmax, argmax}}}}" + hl(str(g["id"])) + "=" + tx(g["piece"])))
    o.append("<h4>a. the winning score is one sum</h4>" + say(f"After the final norm, the score of the winning word is z = sum over the 2048 numbers i of E[{g['id']},i] times g_i times x_i, divided by rms = {n(P['rms'])}. Here are the 6 biggest terms (dimension i), then all the others together:"))
    o.append(disp("z_{" + str(g["id"]) + "}=\\sum_{i}\\frac{E_{" + str(g["id"]) + ",i}\\,g_i\\,x_i}{\\mathrm{rms}}=" + " + ".join(f"\\frac{{{p(e)}\\cdot{p(gg)}\\cdot{p(x)}}}{{{n(P['rms'])}}}" for _, e, gg, x, _ in P["top_terms"][:3]) + "+\\cdots"))
    o.append(disp("=" + " + ".join(p(t) for *_, t in P["top_terms"]) + f"+\\underbrace{{{p(P['rest_terms'])}}}_{{\\text{{all other dimensions}}}}=" + hl(p(P["logit_w"]))))
    o.append("<h4>b. where each part of the sum comes from</h4>" + say("x is itself a sum: the embedding plus what every layer added (attention part and neuron part). So the score splits exactly into 73 contributions; the running total ends at the winning score. Yellow = the 8 biggest pushes (up or down)."))
    big = set(sorted(range(len(cw)), key=lambda i: -abs(cw[i]))[:8]); run = 0.0; rows = []
    for i, (nm, c) in enumerate(zip(names, cw)):
        run += c; cs = p(c)
        rows.append(f"{tx(nm)} & {hl(cs) if i in big else cs} & {n(run)} & {p(c - cr[i])}")
    o.append(disp("\\begin{array}{l|r|r|r}\\text{part} & \\text{adds to } z_{" + str(g["id"]) + "} & \\text{running total} & \\text{vs runner-up}\\\\ \\hline " + " \\\\ ".join(rows) + " \\\\ \\hline \\text{sum} & " + hl(p(sum(cw))) + " & & " + p(sum(cw) - sum(cr)) + "\\end{array}"))
    o.append(say(f"Runner-up word: id {P['runner']} {P['runner_piece']!r} with score {n(P['logit_r'])}. The last column is how much each part pushed the winner above it; it adds up to the gap {n(P['logit_w'] - P['logit_r'])}."))
    o.append("<h4>c. the attention heads that pushed hardest, and what they read</h4>" + say("head contribution = how much of the winning score came through that head; it read the token shown, with that attention weight."))
    o.append(disp("\\begin{array}{r|r|r|l|r}\\text{layer}&\\text{head}&\\text{adds}&\\text{token it read}&\\text{weight}\\\\ \\hline " + " \\\\ ".join(f"{h['l']} & {h['h']} & {p(h['c'])} & {tx(D['pieces'][h['j']]) if h['j'] < T else tx('(earlier answer word ' + str(h['j'] - T + 1) + ')')} \\ (\\text{{pos }}{h['j']}) & {n(h['pj'])}" for h in P["heads"]) + "\\end{array}"))
    o.append("<h4>d. the neurons that pushed hardest</h4>" + say("neuron contribution = its activation times (its column of W_down dotted with the winner's direction)."))
    o.append(disp("\\begin{array}{r|r|r|r}\\text{layer}&\\text{neuron}&\\text{activation}&\\text{adds to }z\\\\ \\hline " + " \\\\ ".join(f"{h['l']} & {h['n']} & {p(h['act'])} & {p(h['c'])}" for h in P["neurons"]) + "\\end{array}"))
    ev = P["evo"]
    o.append("<h4>e. one number's journey through the 36 layers</h4>" + say(f"Dimension {P['dim']} of the vector (the one that adds the most to the winning score). It starts as the token's embedding and gets added to by every attention block and every neuron block:"))
    rows = " \\\\ ".join(f"{l} & {p(ev[1 + 2 * l])} & {p(ev[2 + 2 * l])}" for l in range(DIM["layers"]))
    o.append(disp(f"x^{{(0)}}_{{{P['dim']}}}={p(ev[0])}\\quad(\\text{{embedding}}),\\qquad x^{{(36)}}_{{{P['dim']}}}={hl(p(ev[-1]))}") + disp("\\begin{array}{r|r|r}\\text{layer}&\\text{after attention}&\\text{after neurons}\\\\ \\hline " + rows + "\\end{array}"))
    o.append("<h4>f. the answer id written as text</h4>" + disp(f"{hl(str(g['id']))}\\ \\Rightarrow\\ E[{g['id']},0..7]=" + rowvec(P["emb_w"]) + f"\\ \\Rightarrow\\ \\text{{bytes}}\\ \\Rightarrow\\ " + tx(g["piece"])))
    o.append("</div></details>")
    return "".join(o)


def main():
    o = []
    o.append("<h1>Frost 1.0, worked by hand</h1>" + say(f'Question: "{D["question"]}"') +
             say(f"The model has {DIM['layers']} layers. A token is a vector of {DIM['dim']} numbers. {DIM['heads']} attention heads ({DIM['kv']} key/value heads of {DIM['hd']} numbers), {DIM['ffn']} neurons per layer, vocabulary {DIM['vocab']}. ({D['weights_note']}.)") +
             say(f"I follow the LAST prompt token (position {T - 1}), the one that produces the first answer word. Every number is a real value from the weights, written with 4 digits. Each sum has 2048 terms (or 11008): I write the first 6, the total of the rest, and the result; the totals were checked against the model's own value. Open a layer to see it."))
    o.append("<h2>1. Words become numbers</h2>" + say("The model reads this chat template, not the bare question:") + disp("\\begin{array}{l}" + " \\\\ ".join(tx(s + "\n") for s in D["template"].split("\n")[:-1]) + "\\end{array}"))
    rows = " \\\\ ".join(f"{i} & {tx(D['pieces'][i])} & {D['ids'][i]} & {' '.join(map(str, D['bytes'][i]))}" for i in range(T))
    o.append(disp("\\begin{array}{r|l|r|l} i & \\text{piece} & \\text{token id} & \\text{bytes}\\\\ \\hline " + rows + "\\end{array}"))
    o.append("<h2>2. Each id picks one row of the embedding table</h2>" + disp(f"E\\in\\mathbb{{R}}^{{{DIM['vocab']}\\times{DIM['dim']}}},\\qquad x_i=E[\\mathrm{{id}}_i,\\,:]"))
    rows = [[str(D["ids"][i]), tx(D["pieces"][i])] + [p(v) for v in D["emb"][i]] + ["\\cdots"] for i in range(T)]
    o.append(disp("\\begin{array}{rl|" + "r" * 8 + "l} \\text{id} & \\text{piece} & \\multicolumn{8}{c}{x_i\\ \\text{(first 8 of 2048)}} & \\\\ \\hline " + " \\\\ ".join(" & ".join(r) for r in rows) + "\\end{array}"))
    L0 = D["layers"][0]
    o.append("<h2>2b. What a matrix multiplication does, on real numbers</h2>" + say("A matrix has rows (one per output number) and columns (one per input number). Output number i = row i times the input vector: multiply each weight in the row by the input number above it, then add. Every weight is used once; every input number is used by every row.")
             + disp("y_i=\\sum_{j=0}^{2047}W_{i,j}\\,h_j\\qquad(i=0\\ldots2047)") + mm_block(L0["mm"], "W^{Q}", "h", "q", "Q", "Real example from layer 0: four of the 2048 query numbers, before (weights and input) and after (result):")
             + say("Same thing with 11008 outputs in the feed-forward gate matrix (layer 0), four of them:") + mm_block(L0["mmg"], "W^{G}", "h", "g", "G", "Weights W^G (11008 x 2048) times the normalised vector h:"))
    o.append("<h2>3. The 36 layers</h2>" + say("Every layer does the same nine steps with its own weights; the vector keeps being updated by adding to it."))
    o.append("".join(layer(L) for L in D["layers"]))
    F = D["final"]
    o.append("<h2>4. After layer 35: final norm</h2>" + norm_block("h", F, "output\\_norm"))
    o.append("<h2>5. Scores for every word-piece (logits)</h2>" + disp(f"z_t=E[t,\\,:]\\cdot h,\\qquad t=0\\ldots{DIM['vocab'] - 1}") + say(f"Shown for the winner, id {F['first']} {F['top'][0]['piece']!r}:") +
             disp(f"E[{F['first']},0..7]=" + rowvec(F["top"][0]["w"])) + receipt("z", F["first"], "E", "h", F["rc"], bias=False))
    o.append("<h2>6. Softmax</h2>" + disp(f"p_t=\\frac{{e^{{z_t-z_{{\\max}}}}}}{{\\sum_s e^{{z_s-z_{{\\max}}}}}},\\quad z_{{\\max}}={n(F['max_logit'])},\\quad \\sum_s e^{{z_s-z_{{\\max}}}}={n(F['sum_exp'])}"))
    rows = " \\\\ ".join(f"{a['id']} & {tx(a['piece'])} & {p(a['logit'])} & {a['p'] * 100:.2f}\\%" for a in F["top"])
    o.append(disp("\\begin{array}{r|l|r|r}\\text{id} & \\text{piece} & z & p \\\\ \\hline " + rows + "\\end{array}"))
    o.append("<h2>7. Pick the biggest</h2>" + disp(f"\\arg\\max_t p_t={F['first']}\\ \\Rightarrow\\ {tx(F['top'][0]['piece'])}"))
    G = D["gen"]
    o.append("<h2>8. The winning path of every answer word</h2>" + say("Each answer word is a separate trip through all 36 layers: the previous word's id goes in, the next word's id comes out. Yellow = the winning number of each trip."))
    o.append(disp("\\begin{array}{r|r|l|r|l|r}\\text{trip}&\\text{id in}&\\text{word in}&\\text{id out}&\\text{word out}&p\\\\ \\hline " + " \\\\ ".join(f"{i + 1} & {g['input_id']} & {tx(g['input_piece'])} & {hl(str(g['id']))} & {tx(g['piece'])} & {g['p'] * 100:.1f}\\%" for i, g in enumerate(G)) + "\\end{array}"))
    o.append(disp(f"{G[0]['input_id']}\\ \\to\\ " + "\\ \\to\\ ".join(hl(str(g["id"])) for g in G)))
    o.append(say("Open a trip to see the exact maths of why that word won, part by part:"))
    o.append("".join(path_block(g, i) for i, g in enumerate(G)))
    o.append("<h2>9. The whole answer, one trip per word</h2>")
    rows = " \\\\ ".join(f"{i + 1} & {g['id']} & {tx(g['piece'])} & {g['p'] * 100:.1f}\\% & " + ",\\ ".join(f"{tx(a[1])}\\,{a[2] * 100:.1f}\\%" for a in g["top5"][1:4]) for i, g in enumerate(D["gen"]))
    o.append(disp("\\begin{array}{r|r|l|r|l}\\text{step}&\\text{id}&\\text{piece}&p&\\text{runners-up}\\\\ \\hline " + rows + "\\end{array}"))
    o.append("<h2>10. Numbers back to text</h2>" + disp(" ".join(str(i) for i in D["answer_ids"]) + "\\ \\Rightarrow\\ " + " + ".join(tx(g["piece"]) for g in D["gen"][:len(D["answer_ids"])])) + say("Answer: " + D["answer"]))
    page = PAGE.replace("<!--BODY-->", "".join(o))
    open(os.path.join(ROOT, "capital_of_france_paper.html"), "w", encoding="utf-8").write(page)
    print("written", len(page) // 1024, "KB")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frost 1.0 - worked by hand</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap">
<style>
body{margin:0;background:#d9d4c7;font-family:'Caveat','Segoe Print','Bradley Hand',cursive;color:#1b3a9c}
#sheet{max-width:1000px;margin:18px auto;padding:30px 40px 60px 96px;background-color:#fdfbf2;
 background-image:linear-gradient(90deg,transparent 78px,#e0786f 78px,#e0786f 80px,transparent 80px),repeating-linear-gradient(#fdfbf2 0,#fdfbf2 31px,#c5d6ea 31px,#c5d6ea 32px);box-shadow:0 2px 14px #0004}
h1{font-size:46px;margin:0 0 8px;font-weight:700}h2{font-size:34px;margin:34px 0 4px;font-weight:700;border-bottom:2px solid #1b3a9c55}h4{font-size:25px;margin:22px 0 0;font-weight:700}
p.w{font-size:23px;line-height:32px;margin:4px 0;font-weight:500}
.d{margin:2px 0;overflow-x:auto;overflow-y:hidden;color:#1b3a9c}
.katex{font-size:1.12em;color:#1b3a9c}.katex-display{margin:.5em 0}
.d .katex-display{filter:url(#pen)}
.d:nth-of-type(odd) .katex-display{transform:rotate(-.12deg)}.d:nth-of-type(even) .katex-display{transform:rotate(.1deg)}
details{margin:6px 0}summary{cursor:pointer;font-size:28px;font-weight:700;padding:2px 0}
.ly{border-left:3px solid #1b3a9c33;padding-left:16px}
.nok .d{font:14px Consolas,monospace;white-space:pre-wrap}
</style></head><body>
<svg width="0" height="0" style="position:absolute"><filter id="pen"><feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="2" seed="4" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="1.5"/></filter></svg>
<div id="sheet"><!--BODY--></div>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script>
window.addEventListener('load', () => {
  if (!window.renderMathInElement) { document.getElementById('sheet').classList.add('nok'); return; }   // offline: the raw LaTeX stays readable
  const opt = {delimiters: [{left: '$$', right: '$$', display: true}], throwOnError: false};
  const done = new WeakSet();
  const doIt = el => { if (!done.has(el)) { done.add(el); renderMathInElement(el, opt); el.querySelectorAll('.katex-display').forEach(k => { if (k.offsetHeight > 110) k.style.filter = 'none'; }); } };   // the pen wobble is too slow on tall tables
  document.querySelectorAll('#sheet > .d').forEach(doIt);
  document.querySelectorAll('#sheet > details').forEach(d => d.addEventListener('toggle', () => { if (d.open) doIt(d); }));
});
</script></body></html>
"""

if __name__ == "__main__":
    main()
