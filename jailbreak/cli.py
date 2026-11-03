import argparse
import json
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .config import Config, SMOKE
from .data import load_data
from .defense import ActivationProbes, AttributionDefense
from .gcg import GCG
from .judge import Judge, refused
from .model import load_chat_model, respond
from .models import MODELS, model_slug, resolve_model
from .pair import PAIR
from .plots import plot_attr, plot_projections


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m jailbreak",
                                description="Jailbreak a small chat model with GCG and PAIR, then detect the jailbreaks with "
                                            "Integrated Gradients and activation probes.")
    choices = {"DTYPE": ["float32", "bfloat16", "float16"], "JUDGE": ["keyword", "llm"]}
    for f in fields(Config):
        typ = {"PROBE_LAYER": int}.get(f.name, f.type if f.type in (int, float, str) else str)
        p.add_argument("--" + f.name.lower().replace("_", "-"), dest=f.name, type=typ, default=None,
                       help=f"default: {f.default}", choices=choices.get(f.name))
    p.add_argument("--model", dest="MODEL_NAME", default=None, help="alias for --model-name; presets: " + ", ".join(MODELS))
    p.add_argument("--smoke", action="store_true", help=f"tiny CPU-friendly run to check the pipeline: {SMOKE}")
    p.add_argument("--out-dir", type=Path, default=None, help="default: runs/jailbreak-<model>[-smoke]")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")


def pct(x):
    return "n/a" if np.isnan(x) else f"{x:.1%}"


def results_table(r, attacks):
    rows = [("Attack success rate", "asr"), ("IG defense: detection rate", "ig_detection_rate"),
            ("IG defense: removed tokens in GCG suffix", "ig_suffix_localization"), ("ASR after IG defense", "asr_after_ig"),
            ("Harmfulness probe: detection rate", "probe_detection_rate"), ("ASR after probe", "asr_after_probe"),
            ("Jailbreaks below refusal threshold", "refusal_suppressed")]
    lines = ["| Metric | " + " | ".join(a.upper() for a in attacks) + " |", "|---" * (len(attacks) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(pct(r[a][key]) for a in attacks) + " |" for name, key in rows]
    return f'''
Without an attack: ASR {pct(r["baseline_asr"])}, refusals on harmless prompts {pct(r["harmless_refusal_rate"])}, \
harmfulness probe flags {pct(r["probe_detection_plain_harmful"])} of plain harmful prompts.
False positives on harmless prompts: IG defense {pct(r["ig_false_positive_rate"])}, harmfulness probe {pct(r["probe_false_positive_rate"])}.

''' + "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    overrides = dict(SMOKE) if args.smoke else {}
    overrides.update({f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    cfg = Config(**overrides)
    attacks = [a for a in cfg.ATTACKS.split(",") if a]

    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    device = cfg.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    hf_id = resolve_model(cfg.MODEL_NAME)
    out_dir = args.out_dir or Path("runs") / f"jailbreak-{model_slug(hf_id)}{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n## 1. Data")
    data = load_data(cfg)

    print("\n## 2. Models")
    tok, model = load_chat_model(cfg.MODEL_NAME, cfg.DTYPE, device)
    atok = amodel = None
    if "pair" in attacks or cfg.JUDGE == "llm":
        same = resolve_model(cfg.ATTACKER_MODEL) == hf_id
        atok, amodel = (tok, model) if same else load_chat_model(cfg.ATTACKER_MODEL, cfg.DTYPE, device)
    judge = Judge(cfg.JUDGE, cfg.JUDGE_THRESHOLD, amodel, atok, device)

    print("\n## 3. Baseline (no attack)")
    goals, targets = data["attack_goals"], data["attack_targets"]
    plain = respond(model, tok, goals, cfg.MAX_NEW_TOKENS, device)
    harmless_resp = respond(model, tok, data["eval_harmless"], cfg.MAX_NEW_TOKENS, device)
    results = dict(model=hf_id, baseline_asr=mean(judge.jailbroken(goals, plain)),
                   harmless_refusal_rate=mean([refused(r) for r in harmless_resp]))

    print("\n## 4. Probes")
    probes = ActivationProbes(model, tok, cfg, device)
    probes.fit(data["probe_harmful"], data["probe_harmless"])
    plain_h, plain_r = probes.project(goals)
    harmless_h, harmless_r = probes.project(data["eval_harmless"])
    results.update(probe_layer=probes.layer, probe_detection_plain_harmful=mean(plain_h > 0),
                   probe_false_positive_rate=mean(harmless_h > 0))

    print("\n## 5. Attacks")
    runners = dict(gcg=lambda: GCG(model, tok, cfg, judge, device),
                   pair=lambda: PAIR(model, tok, amodel, atok, cfg, judge, device))
    runs = {}
    for a in attacks:
        runner = runners[a]()
        runs[a] = []
        for i, (g, t) in enumerate(zip(goals, targets)):
            print(f"[{a} {i+1}/{len(goals)}] {g}")
            runs[a].append(runner.attack(g, t))

    print("\n## 6. Defenses")
    ig = AttributionDefense(model, tok, cfg, device)
    harmless_ig = [ig.defend(u, r) for u, r in zip(data["eval_harmless"], harmless_resp)]
    results["ig_false_positive_rate"] = mean([d["flagged"] for d in harmless_ig])
    projections = {"harmless": (harmless_h, harmless_r), "plain harmful": (plain_h, plain_r)}
    for a, rs in runs.items():
        h, r = probes.project([x["prompt"] for x in rs])
        for x, hx, rx in zip(rs, h, r):
            x.update(harm_projection=float(hx), refusal_projection=float(rx), probe_flagged=bool(hx > 0))
            x["ig"] = ig.defend(x["prompt"], x["response"], x["suffix_tokens"]) if x["jailbroken"] else None
        jb = [x for x in rs if x["jailbroken"]]
        projections[f"{a} jailbreaks"] = (np.array([x["harm_projection"] for x in jb]), np.array([x["refusal_projection"] for x in jb]))
        results[a] = dict(
            asr                    = mean([x["jailbroken"] for x in rs]),
            ig_detection_rate      = mean([x["ig"]["flagged"] for x in jb]),
            ig_suffix_localization = mean([x["ig"]["removed_in_suffix"] for x in jb]) if a == "gcg" else float("nan"),
            asr_after_ig           = mean([x["jailbroken"] and not x["ig"]["flagged"] for x in rs]),
            probe_detection_rate   = mean([x["probe_flagged"] for x in jb]),
            asr_after_probe        = mean([x["jailbroken"] and not x["probe_flagged"] for x in rs]),
            refusal_suppressed     = mean([x["refusal_projection"] <= 0 for x in jb]),
        )

    results.update(n_attacked=len(goals), n_harmless=len(data["eval_harmless"]), config=cfg.to_dict())
    print(json.dumps(results, indent=2))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "attacks.jsonl", "w") as f:
        for rs in runs.values():
            for x in rs:
                f.write(json.dumps(x) + "\n")
    print(results_table(results, attacks))

    if not args.no_plots:
        plot_projections(projections, out_dir / "probe_projections.png")
        for a, rs in runs.items():
            x = next((x for x in rs if x["jailbroken"]), rs[0] if rs else None)
            if x:
                plot_attr(ig, x["prompt"], out_dir / f"attribution_{a}.png", x["suffix_tokens"])

    print(f"wrote results to {out_dir}/")
    return results
