"""Run the 3B model (the teacher) on a stream of prompts and record every answer, one JSON line per prompt, as they are produced.
    python teacher_collect.py            -> model/exp_store/teacher.jsonl
The student (angle_student.py) can read the file while this is still running.
12 topics x 4 wordings, shuffled: later wordings of a topic can reuse what the student already learned.
"""
import json, os, random, sys, time
import numpy as np
from engine import Model, chatml

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "model", "exp_store", "teacher.jsonl")
SYSTEM = "You are Frost 1.0, a helpful assistant."
MAX_NEW = 60
GROUPS = {
    "capital of France": ["What is the capital of France?", "capital of France?", "Tell me the capital city of France.", "Which city is the capital of France?"],
    "capital of Japan": ["What is the capital of Japan?", "capital of Japan?", "Tell me the capital city of Japan.", "Which city is the capital of Japan?"],
    "Romeo and Juliet": ["Who wrote Romeo and Juliet?", "Romeo and Juliet was written by whom?", "Name the author of Romeo and Juliet.", "Who is the author of the play Romeo and Juliet?"],
    "boiling point": ["At what temperature does water boil?", "What is the boiling point of water?", "Tell me the boiling point of water in Celsius.", "How hot does water have to get to boil?"],
    "largest planet": ["What is the largest planet in the solar system?", "Which planet is the biggest in our solar system?", "Name the largest planet.", "Tell me the biggest planet of the solar system."],
    "speed of light": ["What is the speed of light?", "How fast does light travel?", "Tell me the speed of light in a vacuum.", "Speed of light in km per second?"],
    "photosynthesis": ["What is photosynthesis?", "Explain photosynthesis briefly.", "Tell me what photosynthesis means.", "How do plants make food using sunlight?"],
    "symbol of gold": ["What is the chemical symbol for gold?", "Chemical symbol of gold?", "Which symbol represents gold in the periodic table?", "Tell me gold's symbol in chemistry."],
    "leap year": ["How many days are in a leap year?", "Number of days in a leap year?", "Tell me how long a leap year is in days.", "A leap year has how many days?"],
    "tallest mountain": ["What is the tallest mountain on Earth?", "Which mountain is the highest in the world?", "Name the tallest mountain.", "Tell me the highest mountain on Earth."],
    "Mona Lisa": ["Who painted the Mona Lisa?", "Mona Lisa was painted by whom?", "Name the painter of the Mona Lisa.", "Tell me who created the Mona Lisa painting."],
    "algorithm": ["What is an algorithm?", "Explain what an algorithm is.", "Tell me the meaning of algorithm.", "Define algorithm in simple words."],
}


def main():
    items = [(t, p) for t, ps in GROUPS.items() for p in ps]
    random.Random(3).shuffle(items)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["prompt"] for l in open(OUT, encoding="utf-8") if l.strip()}
    m = Model(verbose=False, patch=False)                     # the original 3B weights
    end = m.tok.id_of["<|im_end|>"]
    print(f"teacher ready; {len(items) - len(done)} prompts to do", flush=True)
    for n, (topic, prompt) in enumerate(items):
        if prompt in done: continue
        t0 = time.time()
        ids = m.tok.encode(chatml([("system", SYSTEM), ("user", prompt)]))
        cache = m.new_cache(len(ids) + MAX_NEW + 8); lg = m.forward(ids, cache); out, top3, fin = [], [], False
        for _ in range(MAX_NEW):
            p = np.exp(lg - lg.max()); p /= p.sum(); tk = np.argsort(-lg)[:3]; t = int(tk[0])
            top3.append([[int(a), round(float(p[a]), 4)] for a in tk])
            out.append(t)
            if t in (end, m.tok.eos): fin = True; break
            lg = m.forward([t], cache)
        rec = {"n": n, "topic": topic, "prompt": prompt, "prompt_ids": ids, "reply_ids": out, "reply": m.tok.decode([t for t in out if t != end]), "finished": fin, "top3": top3, "seconds": round(time.time() - t0, 1)}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{n + 1}/{len(items)} {prompt!r} -> {rec['reply'][:70]!r} ({len(out)} tokens, {rec['seconds']}s)", flush=True)


main()
