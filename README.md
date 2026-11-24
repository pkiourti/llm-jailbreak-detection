# Jailbreak Detection in Small Chat Models

Can a [MISA](https://arxiv.org/abs/2305.09941)-style attribution defense, built to catch a fixed backdoor trigger, also catch a jailbreak prompt discovered at inference time? This repo attacks small open chat models with GCG and PAIR (`jailbreak/`), then tests Integrated Gradients and activation probes against them.

## Jailbreaks: GCG and PAIR vs. attribution and probes

Can the defenses that caught a planted backdoor trigger also catch an adversarial prompt that is found at inference time?

### Attacks

Both attacks target harmful behaviors from AdvBench (Zou et al. 2023) against `Qwen2.5-0.5B-Instruct` by default.

- **GCG** (Greedy Coordinate Gradient, [Zou et al. 2023](https://arxiv.org/abs/2307.15043)) is white-box. It appends a 20-token suffix to the request and optimizes it with token gradients so that the model's reply starts with the AdvBench target, e.g. "Sure, here is…". Each step swaps one suffix token for one of the top-k candidates by gradient, evaluates a batch of swaps and keeps the best. Every 25 steps it generates a reply and stops early once the judge counts it as jailbroken. Suffixes are restricted to printable ASCII and must re-tokenize to the same ids, so the attack works as plain text.
- **PAIR** (Prompt Automatic Iterative Refinement, [Chao et al. 2023](https://arxiv.org/abs/2310.08419)) is black-box. An attacker LLM (`Qwen2.5-1.5B-Instruct` by default) writes a fluent jailbreak prompt such as a role-play or a fictional framing. It sees the target's reply and the judge's score, then rewrites the prompt. The default is 3 parallel streams × 5 rounds. Attacker replies are prefilled with the start of the JSON answer so a small attacker stays in format; malformed replies are counted in `attacks.jsonl`.

**Judge.** `--judge keyword` (the default) counts a reply as jailbroken if it contains none of the GCG paper's refusal phrases. It is cheap but generous, because a reply that neither refuses nor helps still counts. `--judge llm` also asks the attacker model for a PAIR-style 1–10 rating; only a 10 counts by default (`--judge-threshold`).

### Defenses

- **Attribution (IG or EG), as in [MISA](https://arxiv.org/abs/2305.09941).** It attributes the logit of the reply's first token (e.g. "Sure" vs. "I") back to the user-message tokens, removes the `--remove-k` (default 3) most-attributed tokens, and regenerates. The input is flagged if the model complied before and refuses after. For GCG, *suffix localization* is the fraction of removed tokens that fall inside the adversarial suffix. Two things are configurable, both as CLI flags (`--help`):
  - `--attr-method ig` (default): Integrated Gradients with a single padding baseline (only the user message is ablated, so the chat template is kept).
  - `--attr-method eg`: Expected Gradients ([Erion et al. 2021](https://www.nature.com/articles/s42256-021-00343-w)), as used in [MISA](https://arxiv.org/abs/2305.09941). Instead of one fixed baseline, it averages over random interpolation points *and* a distribution of baselines: here, `--eg-baselines` harmless prompts (padded/truncated to the input's length, the text analogue of MISA's "eval-set images as baselines"), each sampled with `--eg-samples` draws. Implemented with plain `captum.attr.GradientShap` plus a manual forward hook, since `LayerGradientShap`'s NoiseTunnel machinery adds Gaussian noise to the raw token-id input before any layer-hooking happens, which only makes sense for continuous inputs.
  - `--attr-layer` (default: unset, meaning the input embeddings / token-level); an integer `k` attributes at `hidden_states[k]` instead, the same residual-stream location the activation probes read.
- **Activation probes.** These are difference-of-means directions at the middle layer, fit on plain harmful vs. harmless prompts that are disjoint from the attacked ones:
  - The *harmfulness* direction is read at the last user token. It is the detector: does the model still represent the request as harmful?
  - The *refusal* direction ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) is read at the last prompt token. It is diagnostic: a working jailbreak should push it below threshold.

  Recent work (Zhao et al. 2025, "LLMs Encode Harmfulness and Refusal Separately") argues that the two are encoded separately, so a jailbreak may switch off refusal while the harmfulness signal survives.

**Hypothesis.** IG should localize GCG's gibberish suffix well but struggle with PAIR, because no single token in a fluent prompt carries the attack. The harmfulness probe should be more robust to PAIR.

### Results

**Methodology note.** Both judges in `judge.py` are unreliable: the keyword judge counts any non-refusing reply as "jailbroken," even garbled or deflected ones; the LLM judge instead rewards fluent, structured text regardless of whether it's on-target. Below, "ASR" is the keyword judge's number (for comparability with the literature); "real harm rate" is every response read against a four-point rubric (on-topic, coherent, specific, actionable), a single human/Claude pass rather than a validated inter-rater score.

#### Attack success: reported vs. real

| | Judge ASR | Real harm rate |
|---|---|---|
| GCG, weak search | 100% (keyword) | **49%** (n=100) |
| GCG, strong search | 100% (keyword) | **54%** (n=100) |

GCG's real hit rate roughly doubled (~20% to ~50%) just by removing the pipeline's premature early stopping and giving replies more room to finish; the default config under-sells the attack.

#### Defense performance against real jailbreaks

| | Recall | FPR | Precision |
|---|---|---|---|
| IG | 11.9% (CI 7-20%) | 14.9% (CI 8-25%) | ~55% |
| Activation probe | 39.6% (CI 31-49%) | 49.3% (CI 38-61%) | ~55% |

IG rarely fires (catches ~1 in 8 real jailbreaks) but stays quiet on safe inputs; the probe catches more but flags roughly half of everything, harmful or not. Attack strength doesn't reliably change either defense's recall or FPR.

#### Tuning the defense: none of it breaks the trade-off

A full sweep over `--attr-method` (ig/eg), `--attr-layer` (unset/12), and `--remove-k` (3/5/8/12), on the same 200 GCG attempts (recall% / FPR%; precision flat at 51-66% in every cell):

| | `remove_k=3` | 5 | 8 | 12 |
|---|---|---|---|---|
| IG, embedding | 18% / 15% | 28% / 36% | 39% / 39% | 50% / 43% |
| IG, mid-layer | 24% / 25% | 34% / 36% | 53% / 58% | 75% / 67% |
| EG, embedding | 19% / 15% | 31% / 34% | 39% / 40% | 55% / 49% |
| EG, mid-layer | 18% / 25% | 27% / 36% | 44% / 46% | 69% / 61% |

Every knob moves recall and FPR together along the same curve: more aggressive settings catch more real jailbreaks but flag proportionally more harmless ones too, never improving precision. EG never distinguishes itself from IG at matched settings; `IG_STEPS` is pure noise. Mid-layer attribution reaches a real recall advantage at `remove_k=12` (75% vs. 50%), but at 67% FPR, not a usable improvement.

Revisiting the hypothesis: IG localizes GCG's suffix well mechanically (92-97% of removed tokens fall inside it, when it flags anything), but its overall recall stays low (~12-24%) regardless of attack strength, method, or layer. The PAIR half of the hypothesis is addressed below.

#### Does a classification-defense "magic layer" transfer to jailbreaks?

A separate SST-2 study found Phi-3.5-mini-instruct gets 95% detection for a planted classification backdoor at ~75% depth, far better than at the embedding layer. Testing whether that transfers to GCG on the same model (200 attempts, swept over `--attr-layer` and `--remove-k`, [`scripts/phi35_layer_sweep.py`](scripts/phi35_layer_sweep.py); recall against the unverified keyword judge, no harmless control so no real FPR here):

| | `remove_k=3` | 5 | 8 | 12 |
|---|---|---|---|---|
| embedding | 16% | 17% | 17% | 7% |
| layer 16 | 39% | 45% | 38% | 39% |
| layer 24 (~75% depth) | 42% | 46% | **53%** | **53%** |

Recall increases with depth (7-17% at embedding to 42-53% at layer 24), the same shape already seen for Qwen2.5-0.5B-Instruct. There's no second, model-specific magic layer for the generative task: Phi-3.5-mini's classification result doesn't transfer, it's just a further point on the same recall/FPR trade-off documented above.

#### PAIR: not enough data to draw a conclusion

| | Judge ASR | Real harm rate |
|---|---|---|
| PAIR, original setup | 100% (keyword) | **1%** (n=200) |
| PAIR, fixed setup, weak | 10% (LLM) | **1%** (n=100) |
| PAIR, fixed setup, strong | 48% (LLM) | **12%** (n=100) |

PAIR's original setup (default `--judge keyword`, `--attacker-model qwen2.5-1.5b-instruct`) was broken two ways: the keyword judge's binary scoring let its search stop after ~1 iteration out of a 5-10 budget the instant any reply dodged a refusal phrase, and the default 1.5B attacker wasn't a capable enough red-teamer even once that was fixed. Switching the judge alone left real harm near 0%; also swapping to a bigger 4-bit attacker (`--attacker-model qwen2.5-3b-instruct --attacker-load-in-4bit`) raised real harm to **6.5%** (13/200), with attack strength finally mattering (1%→12%, weak→strong). Even fixed, PAIR falls far short of GCG's ~50%, and the LLM judge still overstates real harm by about 4-5x.

Defense numbers for the fixed setup:

| | Recall | FPR | Precision |
|---|---|---|---|
| IG | 23.1%, 3/13 (CI 8-50%) | 5.3%, 10/187 (CI 3-10%) | 23.1% |
| Activation probe | 0.0%, 0/13 (CI 0-23%) | 0.0%, 0/187 (CI 0-2%) | n/a |

**There isn't enough data here to draw a conclusion.** 13 real jailbreaks (15 across both investigations) is far too small a sample: IG's and the probe's recall CIs overlap, so this can't say IG beats the probe, and the pre-registered hypothesis (the probe should be more robust than IG on PAIR) remains untested either way. One pattern worth watching, not confirming: the probe caught 0 of the 15 real PAIR jailbreaks found across both setups. IG's FPR on PAIR's non-jailbreaks (5.3%) is notably lower than its FPR on genuinely harmless prompts (21-23%).

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

**Models.** `--model` takes a preset (`qwen2.5-{0.5b,1.5b,3b}-instruct`, `smollm2-{360m,1.7b}-instruct`, `llama3.2-{1b,3b}-instruct`, `gemma2-2b-it`) or any Hugging Face chat model id. `--attacker-model` sets the PAIR attacker and LLM judge. A larger attacker writes meaningfully better jailbreaks (see *Attack success* above). `--attacker-load-in-4bit` loads it in 4-bit (bitsandbytes) so a bigger attacker still fits alongside the target on a memory-constrained GPU (needed for `qwen2.5-3b-instruct` + `qwen2.5-0.5b-instruct` together on 8GB).

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
