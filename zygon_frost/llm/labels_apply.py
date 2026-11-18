"""Cleans the 3B model's draft labels into Frost Math's own plain names.   python labels_apply.py  ->  model/math_solo/labels.json + model/math_solo/labels_review.txt
Rules: a title is a short name, not a sentence; aliases are short ways to ask; variable labels are 1 to 4 plain words and must belong to the formula.
Anything that breaks a rule is replaced by a safe fallback (the old name in plain words) and listed in the review file. Nothing here changes a formula or a calculation."""
import json, os, re, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frostmath as fm
STORE = fm.STORE; DRAFT = os.path.join(STORE, "labels_draft.jsonl"); OUT = os.path.join(STORE, "labels.json"); REV = os.path.join(STORE, "labels_review.txt")
CURATED = os.path.join(STORE, "labels_curated.json")                 # hand-made fixes win over the draft
m = fm.Math(); draft = {}
for line in open(DRAFT, encoding="utf-8"):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("ok"): draft[r["name"]] = r
cur = json.load(open(CURATED, encoding="utf-8")) if os.path.exists(CURATED) else {}
plain = lambda n: n.replace("_", " ")
out, review = {}, []
for name, rec in m.F.items():
    d = draft.get(name, {}); flags = []
    t = (d.get("title") or "").strip().rstrip(".")
    t = re.sub(r"^(the )?[\w' -]{3,40} (equation|formula|law):\s*", lambda mo: mo.group(0).split(":")[0] + " - ", t) if t.count(":") > 1 else t
    if not t: flags.append("no title")
    elif len(t.split()) > 11 or "=" in t or re.search(r"\b(equals|equal to)\b", t.lower()): flags.append("title is a sentence: " + t); t = ""
    title = t or plain(name)
    al = []
    for a in d.get("aliases") or []:
        a = str(a).strip().rstrip(".").lower()
        if 1 <= len(a.split()) <= 12 and a != title.lower() and a not in al and "=" not in a: al.append(a)
    if not al: flags.append("no aliases")
    vs = {}
    for k, v in (d.get("vars") or {}).items():
        v = str(v).strip().lower()
        if k in rec["vars"] and 1 <= len(v.split()) <= 4 and v != k.lower() and "=" not in v: vs[k] = v
    out[name] = {"title": title, "aliases": al[:6], "vars": vs, "source": "draft by the local 3B model, cleaned by rules", "flags": flags}
    if name in cur: out[name].update({k: v for k, v in cur[name].items()}); out[name]["source"] = "curated"
    if flags or not d: review.append(f"{name:36} | {title:50} | {'; '.join(flags) or ('NO DRAFT' if not d else '')}")
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
open(REV, "w", encoding="utf-8").write("\n".join(review)); print(len(out), "formulas,", sum(1 for k in out if draft.get(k)), "from the draft,", len(review), "need a look ->", REV)
