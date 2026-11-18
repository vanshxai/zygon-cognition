# Usage

Python 3.12, `pip install numpy sympy`. Run from the repository root. All services bind to 127.0.0.1.

## ZYGON chat (`llm/zygon_door.py`, port 8890)
One screen. Jobs: **invoice** (systems, price, hours, rate, material lines `name qty price`, travel, discount %, tax %), **warehouse** (length, width, height, pallet size, level height, aisle %), **HVAC sizing** (BTU/h and delta-T -> tons, CFM, watts), and **any of the 333 formulas** (it asks for each value; units like `12 kW` are read).
- Fix a value: say it again, e.g. `hours 12`. Abandon: `cancel`.
- Chats are saved in `model/zygon_door.json`; pick earlier ones from the dropdown.
- Router floor `FLOOR = 0.45` in `zygon_door.py`: below it, questions go to Lang (`http://127.0.0.1:8840/api/ask`).

## Routing test
```bash
python llm/qkv_router.py "current through a resistor"   # top 3 capabilities with scores
python llm/qkv_router.py --eval                          # 47 test questions vs the keyword baseline
```

## Frost Math (port 8860)
`python llm/frost_forms.py --kind math`. API: `/api/find?q=`, `/api/formula?name=`, `POST /api/solve_one`, `POST /api/solve_chain`. Evaluate: `python llm/math_eval.py` (hide one value, solve it back).

## Frost Engg (port 8870)
`python llm/frost_forms.py --kind engg`. Upload a `.ply` point cloud or a structure `.json`; built-in examples give known sizes. Evaluate: `python llm/engg_eval.py`.

## Frost Lang (port 8840)
`python llm/si_app.py --engine 3 --port 8840`. Teach it by dropping in a PDF/DOCX/TXT or typing lessons; ask "find ..." to look something up on the web (polite, no CAPTCHA bypass). Lessons live in `llm/si_lessons.py`.

## Dashboard
`python llm/frost_hub.py` (port 8800) lists and starts every app.

## Exponent Qwen
Needs the Qwen2.5-3B-Instruct Q4_K_M GGUF at `model/original/qwen2.5-3b-instruct-q4_k_m.original.gguf`.
```bash
python llm/exp_weights.py convert                 # ~2 min, writes model/exp_weights (2.9 GB)
python llm/exp_weights.py run "what is hvac" 40   # offline generation
python llm/exp_compare.py                         # token-by-token vs the original (needs llama.cpp in llm/llama/bin)
python llm/block_diff.py                          # how different are the 36 blocks
```
Memory stays under 4 GB: bytes are memory-mapped and one matrix is decoded at a time.

## Making your own capability for the router
Add a key text to `JOBS` in `llm/qkv_router.py` and a handler in `llm/zygon_door.py`. A formula needs only an entry in `model/math_solo/formulas.jsonl` and a plain-language title in `labels_draft.jsonl`.
