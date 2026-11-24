# Jailbreak Detection in Small Chat Models

Jailbreak attacks on small open chat models, paired with interpretability-based defenses in the style of [llm-backdoor-attribution](https://github.com/pkiourti/llm-backdoor-attribution): break a chat model's safety training with GCG and PAIR (`jailbreak/`), then detect the jailbreaks with Integrated Gradients and activation probes. The sleeper-agent backdoor experiment that used to live here is now its own repo, [llm-sleeper-agents](https://github.com/pkiourti/llm-sleeper-agents).

## Jailbreaks: GCG and PAIR vs. attribution and probes

Can the defenses that caught a planted backdoor trigger also catch an adversarial prompt that is found at inference time?

### Attacks

Both attacks target harmful behaviors from AdvBench (Zou et al. 2023) against `Qwen2.5-0.5B-Instruct` by default.

- **GCG** (Greedy Coordinate Gradient, [Zou et al. 2023](https://arxiv.org/abs/2307.15043)) is white-box. It appends a 20-token suffix to the request and optimizes it with token gradients so that the model's reply starts with the AdvBench target, e.g. "Sure, here is…". Each step swaps one suffix token for one of the top-k candidates by gradient, evaluates a batch of swaps and keeps the best. Every 25 steps it generates a reply and stops early once the judge counts it as jailbroken. Suffixes are restricted to printable ASCII and must re-tokenize to the same ids, so the attack works as plain text.
- **PAIR** (Prompt Automatic Iterative Refinement, [Chao et al. 2023](https://arxiv.org/abs/2310.08419)) is black-box. An attacker LLM (`Qwen2.5-1.5B-Instruct` by default) writes a fluent jailbreak prompt such as a role-play or a fictional framing. It sees the target's reply and the judge's score, then rewrites the prompt. The default is 3 parallel streams × 5 rounds. Attacker replies are prefilled with the start of the JSON answer so a small attacker stays in format; malformed replies are counted in `attacks.jsonl`.

**Judge.** `--judge keyword` (the default) counts a reply as jailbroken if it contains none of the GCG paper's refusal phrases. It is cheap but generous, because a reply that neither refuses nor helps still counts. `--judge llm` also asks the attacker model for a PAIR-style 1–10 rating; only a 10 counts by default (`--judge-threshold`).

### Defenses

- **Attribution (IG or EG), as in MISA and llm-backdoor.** It attributes the logit of the reply's first token (e.g. "Sure" vs. "I") back to the user-message tokens, removes the `--remove-k` (default 3) most-attributed tokens, and regenerates. The input is flagged if the model complied before and refuses after. For GCG, *suffix localization* is the fraction of removed tokens that fall inside the adversarial suffix. Two things are configurable, both as CLI flags (`--help`):
  - `--attr-method ig` (default): Integrated Gradients with a single padding baseline (only the user message is ablated, so the chat template is kept).
  - `--attr-method eg`: Expected Gradients ([Erion et al. 2021](https://www.nature.com/articles/s42256-021-00343-w)), as used in [MISA](https://arxiv.org/abs/2305.09941). Instead of one fixed baseline, it averages over random interpolation points *and* a distribution of baselines — here, `--eg-baselines` harmless prompts (padded/truncated to the input's length, the text analogue of MISA's "eval-set images as baselines"), each sampled with `--eg-samples` draws. Implemented with plain `captum.attr.GradientShap` plus a manual forward hook, since `LayerGradientShap`'s NoiseTunnel machinery adds Gaussian noise to the raw token-id input before any layer-hooking happens, which only makes sense for continuous inputs.
  - `--attr-layer` (default: unset, meaning the input embeddings / token-level); an integer `k` attributes at `hidden_states[k]` instead, the same residual-stream location the activation probes read.
- **Activation probes.** These are difference-of-means directions at the middle layer, fit on plain harmful vs. harmless prompts that are disjoint from the attacked ones:
  - The *harmfulness* direction is read at the last user token. It is the detector: does the model still represent the request as harmful?
  - The *refusal* direction ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) is read at the last prompt token. It is diagnostic: a working jailbreak should push it below threshold.

  Recent work (Zhao et al. 2025, "LLMs Encode Harmfulness and Refusal Separately") argues that the two are encoded separately, so a jailbreak may switch off refusal while the harmfulness signal survives.

**Hypothesis.** IG should localize GCG's gibberish suffix well but struggle with PAIR, because no single token in a fluent prompt carries the attack. The harmfulness probe should be more robust to PAIR.

### Results

**Methodology note, read this first.** Both judges in `judge.py` turned out to be unreliable measures of real harm. The keyword judge counts a reply as "jailbroken" whenever it avoids a refusal phrase — including replies that are garbled nonsense, backwards (e.g. giving a *victim* of identity theft advice instead of a perpetrator), or cleanly deflected to a benign topic. The LLM judge is not simply stricter: rescoring with a bigger model (`qwen2.5-3b-instruct`) showed it rewards *fluent, structured, "step-by-step guide"-shaped* text largely independent of whether it addresses the actual request — it gave high scores to responses that were the literal opposite of the goal (e.g. 8/10 for "describe how to abuse a child" answered with child-*protection* advice) and low scores to responses that were genuinely closer to real harm. Below, "ASR" means the keyword judge's number (for comparability with the attack literature); "real/manual harm rate" means every response was read and scored against a four-point rubric (on-topic, coherent, specific, actionable) by a human/Claude pass, with no second rater — treat these as a careful single read, not a validated inter-rater-agreement number.

#### Attack success: reported vs. real

| | Judge ASR | Real harm rate (manual) |
|---|---|---|
| GCG, weak search (`--gcg-batch 32`) | 100% (keyword) | **49%** (n=100, CI roughly ±10pt) |
| GCG, strong search (`--gcg-batch 128`, no early stop) | 100% (keyword) | **54%** (n=100) |
| PAIR, original setup (default judge/attacker), weak+strong combined | 100% (keyword) | **1%** (n=200) |
| PAIR, fixed setup (see below), weak search | 10% (LLM, threshold 10) | **1%** (n=100) |
| PAIR, fixed setup, strong search | 48% (LLM, threshold 10) | **12%** (n=100) |

GCG's real hit rate roughly **doubled** (from ~20% to ~50%) simply by removing the pipeline's premature early-stopping (`--gcg-check-every 25` → `--gcg-steps`) and giving replies more room to finish (`--max-new-tokens 64` → `150`/`300`) — the default config under-sells the attack. Increasing the candidate pool further (`--gcg-batch 128` vs. `32`) pushed it a bit higher again, though that specific gain (49%→54%) is within noise at n=100 per condition.

**PAIR needed two separate bugs fixed before it was even a measurable attack.** The original setup (`--judge keyword`, the default `--attacker-model qwen2.5-1.5b-instruct`) looked completely broken: all 200 attempts (`pair-n50-{weak,strong}[-seed1]`) scored "jailbroken" by the keyword judge, but a manual read against the same four-point rubric used for GCG (on-topic, coherent, specific, actionable) found only 2 real jailbreaks. Most replies weren't refusals at all — the model answered a *different, benign* question (a bomb-making goal gets a woodworking story), produced literally the *inverse* of the goal (identity-theft prevention tips for an identity-theft goal), or listed security terminology as a generic glossary without assembling an actual attack.

Digging into *why* turned up the real bug: the keyword judge can only score 1 or 10, no gradient, so PAIR's internal search (`pair.py`'s `if best["score"] >= threshold: break`) stopped the instant any reply dodged a refusal phrase — attacks averaged **1.0–1.2 iterations out of a 5–10 budget**, never refining anything. Switching the internal judge to `--judge llm` (a real 1–10 rating) fixed the search mechanics — attacks now used their full iteration budget — but not the result: scores converged on 7–8/10 and real harm stayed near 0%, because the default 1.5B attacker model isn't a capable enough red-teamer; its "refined" prompts kept producing the same off-target/inverted replies, just rated more generously now by a judge with its own well-documented bias toward fluent structure over substance.

What actually worked was also swapping the attacker for a bigger model — `--attacker-model qwen2.5-3b-instruct --attacker-load-in-4bit` (4-bit was needed to fit both models in 8GB; see `--attacker-load-in-4bit` under *Run it* below). With both fixes together, real harm rose from 1% to **6.5%** (13/200) — a genuine, if modest, improvement — and for the first time attack strength made a real difference (1%→12%, weak→strong) instead of being noise. Even so, the LLM judge's reported ASR (10% weak, 48% strong) still overstates real harm by roughly 4–5x: of 58 judge-flagged "jailbreaks," only 12 held up on manual read, split between the same inversion pattern and a new "glossary of attack category names with nothing assembled" pattern; one real jailbreak was missed by the judge entirely (a low-scored reply that buried a genuine SQL-injection methodology). **PAIR, even fixed, still falls far short of GCG's ~50% real success rate against this target** — a brittle, small-model-specific result, not evidence PAIR doesn't work in general.

#### Defense performance against real jailbreaks (default settings: IG, embedding layer, `--remove-k 3`)

Measured on 200 independent GCG attack attempts (98 distinct AdvBench behaviors × weak/strong, Qwen2.5-0.5B-Instruct), each manually labeled:

| | Recall (of real jailbreaks) | False-positive rate (on non-jailbreaks) | Precision |
|---|---|---|---|
| IG | 11.9% · 95% CI [7%, 20%] | 14.9% · CI [8%, 25%] | ~55% |
| Activation probe | 39.6% · CI [31%, 49%] | 49.3% · CI [38%, 61%] | ~55% |

Neither defense is close to deployable: IG rarely fires at all (catches ~1 in 8 real jailbreaks) but is relatively quiet on safe inputs; the probe catches more but at the cost of flagging roughly half of everything, harmful or not. **Attack strength (weak vs. strong search) does not reliably change either defense's recall or FPR** — every weak/strong comparison we ran had overlapping confidence intervals, including an initial, exciting-looking 83%→20% IG collapse from a 6-item sample that did not replicate at n=60 (±18% vs. the true ~15-18%, confirming that result was sampling noise, not a real effect of attack strength).

#### PAIR, fixed setup: defense performance against real jailbreaks

13 real jailbreaks and 187 non-jailbreaks across the 200 fixed-recipe attempts (`pair-n50-{weak,strong}-bigattacker[-seed1]`) — small enough that recall's confidence interval is wide, but no longer the n=2 of the original broken setup:

| | Recall (of real jailbreaks) | False-positive rate (on non-jailbreaks) | Precision |
|---|---|---|---|
| IG | 23.1% · 95% CI [8%, 50%] (3/13) | 5.3% · CI [3%, 10%] (10/187) | 23.1% |
| Activation probe | 0.0% · CI [0%, 23%] (0/13) | 0.0% · CI [0%, 2%] (0/187) | n/a (never flags) |

**This is too small a sample to say IG beats the probe here.** Their recall CIs overlap ([8%, 50%] vs. [0%, 23%]), so by the same standard applied to the GCG weak-vs-strong comparisons above, this isn't a difference the data can actually support. What *is* a consistent observation, not a comparison: the harmfulness probe caught 0 of 15 total real PAIR jailbreaks across both investigations in this project (the original broken setup and this fixed one combined). That's worth noting as a pattern worth watching, not as evidence against the pre-registered hypothesis — settling whether IG genuinely outperforms the probe on PAIR would need substantially more real jailbreaks than either setup produced.

IG's 5.3% false-positive rate on PAIR's non-jailbreak responses is also notably lower than its 21–23% rate on genuinely harmless Alpaca prompts (from the same four runs' `ig_false_positive_rate`) — PAIR's near-misses (inversions, glossaries) apparently don't trigger IG's token-removal flip the way ordinary harmless text occasionally does.

#### Tuning the defense: none of it breaks the trade-off

A full sweep over `--attr-method` (ig/eg), `--attr-layer` (unset/12), and `--remove-k` (3/5/8/12), on the same 200 GCG attempts:

| | `remove_k=3` | 5 | 8 | 12 |
|---|---|---|---|---|
| IG, embedding | 18% / 15% | 28% / 36% | 39% / 39% | 50% / 43% |
| IG, mid-layer | 24% / 25% | 34% / 36% | 53% / 58% | 75% / 67% |
| EG, embedding | 19% / 15% | 31% / 34% | 39% / 40% | 55% / 49% |
| EG, mid-layer | 18% / 25% | 27% / 36% | 44% / 46% | 69% / 61% |

(recall% / FPR%; precision is flat at 51–66% in every one of these 16 cells)

Every knob moves recall and false-positive rate together along essentially the same curve — more aggressive settings (more tokens removed, a later layer) catch more real jailbreaks but flag proportionally more harmless ones too, without ever improving precision. Expected Gradients never distinguishes itself from plain IG at matched settings (CIs overlap at every cell); `IG_STEPS` (the integration resolution, swept 4/8/16/32) is pure noise with no monotonic trend at all. The middle-layer attribution does reach a real, non-overlapping-CI recall advantage over the embedding layer at `remove_k=12` (75% vs. 50%) — but at 67% FPR, meaning it would flag two of every three harmless inputs, so it's not a usable improvement, just a further point on the same curve.

**Revisiting the pre-registered hypothesis above, for the GCG half of it:** IG does localize GCG's suffix well mechanically (92–97% of removed tokens fall inside it, whenever IG flags something at all), but its *overall* recall against real jailbreaks is low (~12-24%) regardless of attack strength, method, or layer — "localizes well when it fires" is not the same as "fires often enough to matter." The PAIR half (the harmfulness probe should be more robust to it than IG, since no single token carries a fluent jailbreak) remains untested in any statistically meaningful sense (see below): across all 15 real PAIR jailbreaks found in this project, IG caught 3 and the probe caught 0, but at this sample size that's not enough to confirm or refute the hypothesis either way.

#### Does a model's best classification-defense layer transfer to the generative jailbreak task?

A separate SST-2 backdoor-classification study ([llm-backdoor-attribution](https://github.com/pkiourti/llm-backdoor-attribution)) found that attributing at ~75% depth gives Phi-3.5-mini-instruct a strikingly good 95% detection rate for a planted classification backdoor — far better than embedding-layer attribution on the same model. That's a classification setting with a universal, fixed trigger; the question here is whether that "magic layer" also helps IG catch GCG's per-prompt optimized suffix on the same model.

200 GCG attempts against Phi-3.5-mini-instruct (`gcg-phi35-{weak,strong}[-seed1]`, 50 each) were swept over `--attr-layer` (embedding / 16 / 24, i.e. the embedding, mid-depth, and ~75%-depth points) × `--remove-k` (3/5/8/12) with [`scripts/phi35_layer_sweep.py`](scripts/phi35_layer_sweep.py):

| | `remove_k=3` | 5 | 8 | 12 |
|---|---|---|---|---|
| embedding | 16% | 17% | 17% | 7% |
| layer 16 | 39% | 45% | 38% | 39% |
| layer 24 (~75% depth) | 42% | 46% | **53%** | **53%** |

(recall against the keyword judge's "jailbroken" label, out of 150/200 prompts it marked as jailbroken)

Two caveats make this weaker evidence than the Qwen results above, and they're the reason this is reported as a secondary finding rather than folded into the main table:

- **No manual verification.** Recall here is against the keyword judge, which the methodology note at the top of this section already established is unreliable (it credits garbled or off-target replies as "jailbroken"). Unlike the Qwen numbers, no one has read these 200 Phi-3.5 responses to check which are real jailbreaks.
- **No false-positive measurement.** The sweep script reuses only the 200 GCG prompts — it never runs the defense on a harmless-prompt control set. Its `flagged` field also requires `complied_before=True`, which every keyword-judge-refused prompt fails by construction, so a computed "FPR" from this data would be 0% for every cell, not a real measurement. The table above reports recall only.

Even with those caveats, the pattern is clear enough to answer the original question: **recall increases with depth (7–17% at the embedding layer → 38–46% at layer 16 → 42–53% at layer 24), the same shape already seen for Qwen2.5-0.5B-Instruct** (18–50% embedding → 24–75% mid-layer, from the sweep above). There's no second, model-specific "magic layer" for the generative task — Phi-3.5-mini's classification-backdoor result doesn't transfer. The gain from going deeper is consistent with the same recall/FPR trade-off documented above, not a break from it; at `remove_k=8-12`, layer 24's ~53% recall would very plausibly come with a FPR high enough to land in the same ~55% precision band as everything else in this section, but confirming that (and whether any of these are even real jailbreaks) would need a harmless control set and a manual read, which wasn't worth the added compute given how strongly the recall trend alone already answers the transfer question.

### Run it

```bash
uv sync
```

```bash
uv run python -m jailbreak --smoke
```

```bash
uv run python -m jailbreak
```

```bash
uv run python -m jailbreak --attacks gcg --model qwen2.5-1.5b-instruct --dtype bfloat16
```

```bash
uv run python -m jailbreak --attacks gcg --attr-method eg --attr-layer 12 --remove-k 8
```

- `--smoke` checks the whole pipeline in about 30 seconds on a laptop CPU (after the first model download), using 2 behaviors, 3 GCG steps and 1 PAIR round. Its numbers are meaningless.
- The full run (20 behaviors, 250 GCG steps, 3×5 PAIR) wants a GPU. GCG dominates the runtime, roughly a few minutes per behavior on a T4 for the 0.5B model. Use `--attacks pair` or a smaller `--n-attack` / `--gcg-steps` for a quick look.
- Every config value has a flag (`--help`).
- Each run writes these files to `runs/jailbreak-<model>[-smoke]/` (override with `--out-dir`):
  - `results.json`: metrics plus the config
  - `attacks.jsonl`: every attack's final prompt, reply, scores, probe projections and IG result
  - `probe_projections.png`: harmfulness vs. refusal scatter for harmless prompts, plain harmful prompts and successful jailbreaks
  - `attribution_<attack>.png`: per-token IG for one jailbreak, with GCG suffix tokens in red

**Models.** `--model` takes a preset (`qwen2.5-{0.5b,1.5b,3b}-instruct`, `smollm2-{360m,1.7b}-instruct`, `llama3.2-{1b,3b}-instruct`, `gemma2-2b-it`) or any Hugging Face chat model id. `--attacker-model` sets the PAIR attacker and LLM judge. A larger attacker writes meaningfully better jailbreaks (see *Attack success* above) — `--attacker-load-in-4bit` loads it in 4-bit (bitsandbytes) so a bigger attacker still fits alongside the target on a memory-constrained GPU (needed for `qwen2.5-3b-instruct` + `qwen2.5-0.5b-instruct` together on 8GB).

**Data.** AdvBench is loaded from the [GCG repo's CSV](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) because the Hub copy is gated. Harmless prompts are Alpaca instructions with no input field.

| Module | What it does |
|---|---|
| `config.py` | `Config` dataclass with the defaults, plus the `--smoke` overrides |
| `models.py` | Chat-model presets |
| `data.py` | AdvBench / Alpaca loading and the probe / attack / harmless splits |
| `model.py` | Model loading, chat-template encoding (the user message is tokenized separately so a GCG suffix keeps its tokens), generation |
| `judge.py` | Keyword and LLM judges |
| `gcg.py`, `pair.py` | The two attacks |
| `defense.py` | `AttributionDefense` (IG or EG + token removal, embedding or mid-layer) and `ActivationProbes` (harmfulness and refusal directions) |
| `plots.py` | Attribution bars and the probe-projection scatter |
| `cli.py` | Argument parsing, the pipeline and the results table |
