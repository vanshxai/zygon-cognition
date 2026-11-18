"""Layer 1 of 36, for one word, written on paper with plain-English labels.
    python plain_layer.py the      (needs model/frost_work/layer1_the.json from the_layer1.py)   ->  the_layer1_paper.html
"""
import html, json, math, os, sys
sys.stdout.reconfigure(encoding="utf-8")
from paper import n, p, tx, bm, col, disp, say, hl, PAGE, ROOT

word = sys.argv[1] if len(sys.argv) > 1 else "the"
D = json.load(open(os.path.join(ROOT, "model", "frost_work", f"layer1_{word}.json"), encoding="utf-8"))
DIM, P, T = D["dims"], D["P"], D["T"]
tw = lambda s: "\\text{" + s.replace(chr(0x2019), chr(39)).replace(chr(0xF7), "/").replace(chr(0xD7), "x") + "}"


def h(s):
    return f"<h4>{html.escape(s)}</h4>"


def mmn(M, wname, inname, outname, cols_note=None, bias=True):
    """four real outputs of one matrix multiplication, every part named in words"""
    r, c = M["rows_total"], M["cols"]
    Wb = bm([[p(v) for v in row] + ["\\cdots"] for row in M["W"]]); Ib = col([p(v) for v in M["h"]])
    o = [disp("\\underbrace{" + tw(wname) + "}_{" + tw(f"matrix: {r} rows x {c} columns (showing 4 rows, 8 columns)") + "}=" + Wb)]
    o.append(disp("\\underbrace{" + tw(inname) + "}_{" + tw(f"{c} numbers (showing the first 8)") + "}=" + Ib))
    o.append(say("Each output number = its row of the matrix, multiplied number by number with the input, then added up. First the 8 terms we can see, then all the other terms of the row together" + (", then the bias (a fixed number added at the end)" if bias and any(M["b"]) else "") + ":"))
    o.append(disp(tw(outname + " 0..3") + "=" + bm([[p(v)] for v in M["part"]]) + tw(" (first 8 terms) ") + "+" + bm([[p(v)] for v in M["rest"]]) + tw(f" (other {c - 8} terms) ")
                  + (("+" + bm([[p(v)] for v in M["b"]]) + tw(" (bias) ")) if bias and any(M["b"]) else "") + "=" + hl_col(M["out"])))
    return "".join(o)


def hl_col(vals):
    return bm([["\\colorbox{#ffe27a}{$" + p(v) + "$}"] for v in vals])


def norm_table(nm, title, unit_in, unit_out):
    rows = " \\\\ ".join(f"{i} & {p(a)} & {p(a / nm['rms'])} & {p(g)} & {hl(p(o))}" for i, (a, g, o) in enumerate(nm["e"]))
    return (disp(tw("average of the squares of all 2048 numbers") + f"={n(nm['ms'])},\\qquad " + tw("typical size") + f"=\\sqrt{{{n(nm['ms'])}+{n(nm['eps'])}}}={n(nm['rms'])}")
            + say(f"Each number is divided by the typical size ({n(nm['rms'])}) and multiplied by its own learned gain (a stored weight). First 8 of 2048:")
            + disp("\\begin{array}{r|r|r|r|r}" + tw("number #") + "&" + tw(unit_in) + "&" + tw("÷ typical size") + "&" + tw("× gain") + "&" + tw(unit_out) + "\\\\ \\hline " + rows + "\\end{array}"))


def nrec(label, wlabel, inlabel, rc, bias=False):
    t = rc["terms"]
    first = " + ".join(f"{p(a)}\\cdot{p(b)}" for a, b, _ in t[:3]); second = " + ".join(f"{p(a)}\\cdot{p(b)}" for a, b, _ in t[3:])
    prods = " + ".join(p(c) for _, _, c in t)
    return disp("\\begin{aligned}" + tw(label) + f"&= {first}\\\\ &\\quad+ {second}+\\cdots\\\\ &= {prods}+\\underbrace{{{p(rc['rest'])}}}_{{{tw(f'the other {rc['n'] - 6} terms')}}}" + (f"+{p(rc['bias'])}" if bias else "") + f"\\\\ &= {hl(p(rc['out']))}" + "\\end{aligned}")


def main():
    pcs = D["pieces"]; o = []
    o.append(f"<h1>Frost 1.0 and the word “{html.escape(word)}”</h1>")
    # ---------------------------------------------------------------- A. the reply
    g = D["greedy"]
    o.append("<h2>A. What Frost answers to the one word “" + html.escape(word) + "”</h2>")
    o.append(say(f"Always-the-top-word run (no randomness), {g['tokens']} words" + ("" if g["finished"] else " (cut off here at the limit)") + ":"))
    o.append("<pre class='rep'>" + html.escape(g["text"]) + "</pre>")
    o.append(say("The chat app picks randomly among the likely words (temperature 0.7), so the reply can differ. Three such runs, first 50 tokens each:"))
    for s_ in D["samples"]:
        o.append("<pre class='rep'>" + html.escape(s_["text"]) + "</pre>")
    # ---------------------------------------------------------------- B. the prompt as numbers
    o.append("<h2>B. The prompt as numbers</h2>" + say(f"Frost never sees the bare word. It sees this chat text cut into {T} pieces, each piece a whole number (its id). The word “{word}” is piece number {P}, id {D['ids'][P]}. This page follows that one piece through layer 1."))
    rows = " \\\\ ".join((f"{i} & {tx(pcs[i])} & {hl(str(D['ids'][i]))}" if i == P else f"{i} & {tx(pcs[i])} & {D['ids'][i]}") for i in range(T))
    o.append(disp("\\begin{array}{r|l|r}" + tw("piece #") + "&" + tw("text") + "&" + tw("id") + "\\\\ \\hline " + rows + "\\end{array}"))
    # ---------------------------------------------------------------- C. layer 1
    o.append("<h2>C. Layer 1 of 36</h2>" + say(f"Layer 1 receives the word’s 2048 numbers (its embedding, looked up from the table of 151,936 rows) and hands 2048 new numbers to layer 2. Inside are 9 steps. The first 8 of the 2048 numbers going in:")
             + disp(tw("numbers coming in") + "=" + bm([[p(v) for v in D["emb8"]] + ["\\cdots"]])))
    o.append(disp("\\begin{array}{ll}" + tw("1. tidy") + "&" + tw("rescale the numbers to a typical size of 1") + "\\\\" + tw("2. three questions") + "&" + tw("build a query, a key and a value from them (3 matrices)") + "\\\\"
                  + tw("3. twist by position") + "&" + tw("so the model knows where in the text this word is") + "\\\\" + tw("4. look back") + "&" + tw("score every earlier word, turn scores into shares") + "\\\\"
                  + tw("5. blend") + "&" + tw("mix the earlier words’ values by those shares, then one more matrix") + "\\\\" + tw("6. add back") + "&" + tw("new numbers = old numbers + what attention found") + "\\\\"
                  + tw("7. tidy again") + "&" + tw("same as step 1") + "\\\\" + tw("8. neurons") + "&" + tw("11,008 neurons each read the numbers (2 matrices) and fire") + "\\\\" + tw("9. combine + add back") + "&" + tw("1 matrix, then new numbers = old + neurons’ result") + "\\end{array}"))
    o.append(h("Step 1. Tidy the numbers") + say("Some of the 2048 numbers are big and some tiny. Tidying rescales them so the typical size is 1, then multiplies each by its own learned gain.") + norm_table(D["norm1"], "", "number coming in", "tidied number"))
    o.append(h("Step 2. Ask three questions: what am I looking for, what do I contain, what can I give?"))
    o.append(say(f"Three matrices turn the 2048 tidied numbers into: a QUERY (what this word is looking for, {DIM['heads']} search features of {DIM['hd']} numbers = 2048 numbers), a KEY (a label saying what this word contains, {DIM['kv']} sets of {DIM['hd']} = 256 numbers) and a VALUE (what this word will hand over if another word pays attention to it, 256 numbers)."))
    o.append(say("The QUERY matrix:") + mmn(D["mm_q"], "query matrix", "tidied numbers", "query numbers"))
    o.append(say("The KEY matrix (256 rows, so fewer outputs):") + mmn(D["mm_k"], "key matrix", "tidied numbers", "key numbers"))
    o.append(say("The VALUE matrix:") + mmn(D["mm_v"], "value matrix", "tidied numbers", "value numbers"))
    at = D["att"]
    o.append(h("Step 3. Twist query and key by the word’s position") + say(f"Queries and keys are rotated in pairs of numbers. The angle depends on the position of the word in the text (here position {P}). Two of the 64 pairs of head {at['head']}, before and after turning:"))
    for r in at["rope"]:
        o.append(disp(tw("angle") + f"={n(r['a'])}\\ " + tw("radians") + ":\\quad" + bm([["\\cos", "-\\sin"], ["\\sin", "\\cos"]]) + bm([[p(r["x1"])], [p(r["x2"])]]) + "=" + bm([[hl(p(r["r1"]))], [hl(p(r["r2"]))]])))
    o.append(h("Step 4. Look back: how much does this word look at each earlier word?") + say(f"This word can look at the {at['n']} pieces up to and including itself. There are {DIM['heads']} independent “heads”, each with its own opinion. The one piece each head looks at most:"))
    hrows = " \\\\ ".join(f"{x['h']} & {tx(pcs[x['top']])} & {n(x['w'] * 100)}\\% & {n(x['self'] * 100)}\\%" for x in at["heads"])
    o.append(disp("\\begin{array}{r|l|r|r}" + tw("head") + "&" + tw("looks most at") + "&" + tw("how much") + "&" + tw("looks at itself") + "\\\\ \\hline " + hrows + "\\end{array}"))
    qk = at["qk"]
    o.append(say(f"Head {at['head']} (the head that looks hardest at something other than the very first piece) in detail. First, the score for one earlier piece, {pcs[qk['j']]!r}: multiply the query by that piece’s key number by number, add, and divide by √128 = 11.31:"))
    o.append(disp(tw("score") + f"_{{{qk['j']}}}=\\frac{{" + " + ".join(f"{p(a)}\\cdot{p(b)}" for a, b, _ in qk["terms"]) + f" + \\cdots}}{{11.31}}=\\frac{{" + " + ".join(p(c) for *_, c in qk["terms"]) + f"+{p(qk['rest'])}}}{{11.31}}={hl(p(qk['score']))}"))
    ex = [math.exp(v - max(at["scores"])) for v in at["scores"]]; Z = sum(ex)
    o.append(say(f"Do that for all {at['n']} pieces. Then turn scores into shares that add up to 100%: take e to the power (score minus the biggest score), add them all up (total {n(Z)}), and divide:"))
    rows = " \\\\ ".join(f"{j} & {tx(pcs[j])} & {p(sj)} & {n(e_)} & {n(e_ / Z * 100)}\\%" for j, (sj, e_) in enumerate(zip(at["scores"], ex)))
    o.append(disp("\\begin{array}{r|l|r|r|r}" + tw("piece #") + "&" + tw("text") + "&" + tw("score") + "&" + tw("e^{score-biggest}") + "&" + tw("share (attention)") + "\\\\ \\hline " + rows + "\\end{array}"))
    mx = at["mix"]
    o.append(h("Step 5. Blend: mix what each earlier word offers, by those shares") + say(f"Every piece has a value (what it hands over). This head’s result is the shares times the values, added up. Written out for output number {mx['d']} of the head (the 6 biggest shares, then the rest):"))
    o.append(disp(tw("blended number") + "=" + " + ".join(f"{n(s)}\\cdot{p(vv)}" for _, s, vv, _ in mx["terms"]) + "+\\cdots=" + " + ".join(p(c) for *_, c in mx["terms"]) + f"+{p(mx['rest'])}={hl(p(mx['c']))}"))
    o.append(say("All 16 heads do this; their results are put side by side (2048 numbers) and one more matrix mixes them:") + mmn(D["mm_o"], "output-mixing matrix", "blended numbers (all heads)", "attention result", bias=False))
    r1 = D["res1"]
    o.append(h("Step 6. Add back what attention found") + say("The old numbers are kept and the attention result is simply added. First 8 numbers:"))
    o.append(disp("\\begin{array}{r|r|r|r}" + tw("number #") + "&" + tw("old number") + "&" + tw("+ attention result") + "&" + tw("= new number") + "\\\\ \\hline " + " \\\\ ".join(f"{i} & {p(a)} & {p(b)} & {hl(p(c))}" for i, (a, b, c) in enumerate(r1)) + "\\end{array}"))
    o.append(h("Step 7. Tidy again") + norm_table(D["norm2"], "", "number", "tidied number"))
    o.append(h("Step 8. The neurons") + say(f"The layer has {DIM['ffn']} neurons. Each neuron owns one row in the GATE matrix and one row in the UP matrix. Each row is dotted with the tidied numbers. Four neurons (numbers 0 to 3) in detail:"))
    o.append(say("The GATE matrix (11008 rows):") + mmn(D["mm_g"], "gate matrix", "tidied numbers", "gate of neurons", bias=False))
    o.append(say("The UP matrix (11008 rows):") + mmn(D["mm_u"], "up matrix", "tidied numbers", "up of neurons", bias=False))
    o.append(say("A neuron’s strength = SiLU(gate) × up, where SiLU(z) = z / (1 + e^-z): a smooth switch that is almost 0 for negative z and about z for positive z. The three loudest neurons of this layer for this word, with their real weights:"))
    for i, nr in enumerate(D["neurons"]):
        gz = nr["gate"]; s = gz / (1 + math.exp(-gz))
        o.append(disp(tw(f"neuron {nr['n']}") + ":\\ " + tw("gate row (first 8 weights)") + "=" + bm([[p(v) for v in nr["wg"]] + ["\\cdots"]]) + ",\\ " + tw("up row") + "=" + bm([[p(v) for v in nr["wu"]] + ["\\cdots"]])))
        if i == 0:
            o.append(nrec("gate", "", "", nr["rg"])); o.append(nrec("up", "", "", nr["ru"]))
        o.append(disp(tw(f"strength of neuron {nr['n']}") + f"=\\mathrm{{SiLU}}({p(gz)})\\cdot{p(nr['up'])}=\\frac{{{p(gz)}}}{{1+e^{{{n(-gz)}}}}}\\cdot{p(nr['up'])}={n(s)}\\cdot{p(nr['up'])}={hl(p(nr['act']))}"))
    o.append(say(f"{D['active']} of the {DIM['ffn']} neurons ({100 * D['active'] / DIM['ffn']:.1f}%) are at least 10% as strong as the loudest one; the rest are close to silent."))
    o.append(h("Step 9. Combine the neurons and add back") + say("The DOWN matrix turns the 11,008 neuron strengths back into 2048 numbers: each of its 2048 rows is dotted with all the strengths."))
    o.append(mmn(D["mm_d"], "down matrix", "neuron strengths", "neurons’ result", bias=False))
    r2 = D["res2"]
    o.append(say("Finally the neurons’ result is added to the numbers from step 6. This is what leaves layer 1 and enters layer 2:"))
    o.append(disp("\\begin{array}{r|r|r|r}" + tw("number #") + "&" + tw("numbers after step 6") + "&" + tw("+ neurons’ result") + "&" + tw("= goes to layer 2") + "\\\\ \\hline " + " \\\\ ".join(f"{i} & {p(a)} & {p(b)} & {hl(p(c))}" for i, (a, b, c) in enumerate(r2)) + "\\end{array}"))
    o.append("<h2>D. Before and after layer 1</h2>" + disp(tw("in") + "=" + bm([[p(v) for v in D["emb8"]] + ["\\cdots"]]) + "\\ \\Longrightarrow\\ " + tw("out") + "=" + bm([[p(c) for *_, c in r2] + ["\\cdots"]])) +
             say("The same nine steps repeat in layers 2 to 36, each with its own matrices. After layer 36 the numbers are compared with all 151,936 word-pieces to choose the next word."))
    body = "".join(o)
    page = PAGE.replace("<!--BODY-->", body).replace("</style>", "pre.rep{font:20px/1.4 'Caveat','Segoe Print',cursive;white-space:pre-wrap;background:#fffbe0;border-left:4px solid #1b3a9c;margin:6px 0;padding:6px 12px}</style>", 1)
    page = page.replace("#sheet > .d", "#sheet .d").replace("#sheet > details", "#sheet > details")      # every formula is on the page, no dropdowns
    page = page.replace("Frost 1.0 - worked by hand", f"Frost 1.0 - layer 1 for '{word}'")
    open(os.path.join(ROOT, f"{word}_layer1_paper.html"), "w", encoding="utf-8").write(page)
    print("written", len(page) // 1024, "KB")


main()
