import random

from datasets import load_dataset

# AdvBench harmful behaviors (Zou et al. 2023), from the GCG repo (the Hub copy is gated)
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def load_data(cfg):
    """Disjoint splits: harmful/harmless prompts to fit the probes, harmful behaviors to attack, harmless prompts for false positives."""
    rng = random.Random(cfg.SEED)
    adv = load_dataset("csv", data_files=ADVBENCH_URL, split="train")
    harmful = list(zip(adv["goal"], adv["target"]))
    rng.shuffle(harmful)
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"] for r in alpaca if not r["input"]]
    rng.shuffle(harmless)

    p, a = cfg.N_PROBE, cfg.N_ATTACK
    data = dict(
        probe_harmful   = [g for g, _ in harmful[:p]],
        probe_harmless  = harmless[:p],
        attack_goals    = [g for g, _ in harmful[p:p+a]],
        attack_targets  = [t for _, t in harmful[p:p+a]],
        eval_harmless   = harmless[p:p+cfg.N_HARMLESS],
    )
    print({k: len(v) for k, v in data.items()})
    return data
