"""Sweep IG's attribution layer and REMOVE_K on Phi-3.5-mini-instruct, reusing the saved prompt/response pairs from
the four gcg-phi35-* runs (no new GCG attacks needed). Run on a GPU with more than ~8GB headroom (Colab, not the
local 4060 -- the model alone takes 7.6GB in bf16, leaving nothing for IG's backward pass).

Usage: uv run python scripts/phi35_layer_sweep.py
Writes results to runs/phi35_layer_k_sweep.json and prints a summary table.
"""
import json

import numpy as np
from jailbreak.config import Config
from jailbreak.defense import AttributionDefense
from jailbreak.judge import refused
from jailbreak.model import generate_ids, load_chat_model
from jailbreak.models import resolve_model

RUNS = ["gcg-phi35-weak", "gcg-phi35-weak-seed1", "gcg-phi35-strong", "gcg-phi35-strong-seed1"]
LAYERS = [None, 16, 24]
REMOVE_KS = [3, 5, 8, 12]
DEVICE = "cuda"


def main():
    cfg = Config()
    hf_id = resolve_model("phi-3.5-mini-instruct")
    tok, model = load_chat_model(hf_id, "bfloat16", DEVICE)

    rows = []
    for run in RUNS:
        with open(f"runs/{run}/attacks.jsonl") as f:
            rows += [json.loads(line) for line in f]
    print(len(rows), "rows loaded", flush=True)

    all_results = {}
    for layer in LAYERS:
        print(f"--- attributions: layer={layer} ---", flush=True)
        ad = AttributionDefense(model, tok, cfg, DEVICE, method="ig", layer=layer)
        attrs = []
        for i, r in enumerate(rows):
            seq, (s, e), first, tokens, scores = ad.token_attributions(r["prompt"])
            attrs.append(dict(seq=seq, s=s, e=e, scores=scores.tolist()))
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

        for k in REMOVE_KS:
            res = []
            # rows are concatenated in RUNS order: the two weak runs (max_new_tokens=150) first, then the two
            # strong runs (max_new_tokens=300) -- 50 rows each, so weak = rows[0:100], strong = rows[100:200].
            weak_n = 100
            for budget, idxs in [(150, list(range(0, weak_n))), (300, list(range(weak_n, len(rows))))]:
                kept_seqs = []
                for i in idxs:
                    a = attrs[i]
                    scores = np.array(a["scores"])
                    top = np.argsort(-np.abs(scores))[: min(k, a["e"] - a["s"] - 1)]
                    drop = {a["s"] + int(t) for t in top}
                    kept = [t for j, t in enumerate(a["seq"]) if j not in drop]
                    kept_seqs.append(kept)
                afters = generate_ids(model, tok, kept_seqs, budget, DEVICE)
                for i, after in zip(idxs, afters):
                    r = rows[i]
                    complied_before = not refused(r["response"])
                    complied_after = not refused(after)
                    res.append(dict(idx=i, flagged=complied_before and not complied_after))
            all_results[f"layer{layer}_K{k}"] = res
            print(f"layer={layer} K={k}: done", flush=True)

    with open("runs/phi35_layer_k_sweep.json", "w") as f:
        json.dump(all_results, f)
    print("saved runs/phi35_layer_k_sweep.json -- push this and pull it back to compute recall/FPR against the manual verdicts")


if __name__ == "__main__":
    main()
