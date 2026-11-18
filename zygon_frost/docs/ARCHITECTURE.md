# ZYGON architecture

## Idea
Modern LLMs store knowledge as huge matrices and guess the next word. ZYGON keeps knowledge as **named, exact records** and reaches them by **direct lookup**. A keyword is bound to its function; calling it is a click, not a calculation over a million numbers. Numbers carry their own instructions, and comparison is done with exponents and angles, not learned weight matrices.

## Components
```
 user text
    |
 qkv_router.py  --- angle-sum of the words vs. every capability's key text
    |-- job:invoice | job:warehouse | job:hvac_size      (zygon_door.py, exact Fractions)
    |-- formula:<name>   (333 formulas -> frostmath.py: exact sympy solve, units, put-back check)
    |-- engg             (frostengg.py: planes/cylinders from point clouds)
    '-- lang             (lang.py: sentence memory, directional gate, chat thread)
```
- **State sheet:** each chat keeps a keyed sheet (`key -> value`). New values overwrite old ones; a hidden log keeps every change. One screen, never scrolls; reopening a chat restores the last state (`model/zygon_door.json`).
- **Frost Math:** formulas live in `model/math_solo/formulas.jsonl`; plain-language names/aliases in `labels_draft.jsonl`; units in `units.json`. Solving is exact (fractions), checked by substituting the answer back (60-digit check), with numeric fallback when there is no closed form.
- **Frost Lang:** sentences are stored with word identities and linked on a path; a directional gate decides which neighbouring sentence belongs in the answer (paragraph F1 0.62 -> 0.92 with paragraph ends).
- **Meaning table** (`model/meaning/meaning.npz`): each of ~20,000 words carries 64 whole-number angles; two words mean alike when their angles line up. Built offline by `meaning_build.py` (PCA over the 3B model's embedding table).
- **Angle-QKV router:** Query = sum of word angle-vectors in the question (rarity-weighted). Key = same sum over each capability's description (formula title, aliases, variable names). Score = how well the sums line up. Value = the function to call. Add, compare, call; no matrices, no tokenizer, no training. Below score 0.45 the question goes to Lang.

## Exponent Qwen (experiment)
`exp_weights.py convert` rewrites every 2-D weight of Qwen2.5-3B-Instruct (Q4_K_M GGUF) as one byte: 1 sign bit + a 7-bit integer exponent step, value = +/- 2^((step-100)/4). That is 4 steps per octave, about 5% average error per weight, 5.71 bits of information per weight (entropy). The forward pass (RMSNorm, RoPE, attention, SiLU feed-forward) is unchanged; one matrix is widened at a time through a lookup table so the run fits in 4 GB.

Measured: coherent answers; same top token as the original 90.8% (teacher-forced, 5 repair prompts x 48 tokens).
Honest status: same model, different storage. It is slower (13 s/token) and bigger (2.9 GB) than Q4 because the multiplications are still done in float after lookup. The benefit appears only if the maths is rewritten as exponent adds and shifts.

`block_diff.py` shows the 36 blocks have different numbers entirely (pairwise similarity ~0), so they cannot be replaced by one repeated block.

## Experiments and tools
- `trace_model.py` traces which matrix cells did the work for a prompt; `build_exploded.py` builds the interactive exploded model view.
- `weights_to_hexonet_try.py`, `dir_experiment.py`, `ctx_experiment.py`: earlier directional/exponent experiments (results in `model/*.json`).
- Earlier Frost SI 1.0/2.0 apps (`si_app.py`, `si2.py`): word identities on angle wheels.

## What is deliberately not here
The private blueprint folders, credentials, model weights and training books. Rebuild model-dependent files with the scripts above.
