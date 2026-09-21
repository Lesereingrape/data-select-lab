"""Full LoRA + LESS data-selection study -> results/selection.json (CPU).

Three seeds x four budgets x three selection strategies (LESS top-k / random /
bottom-k). For every cell we record held-out target accuracy (the capability the
selection targets) and easy accuracy (the base skill — we do NOT hide that
narrow selection causes catastrophic forgetting). Aggregates are mean +/- std
across seeds.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from datasel.pipeline import run_seed

SEEDS = (0, 1, 2)
BUDGETS = (32, 64, 128, 256)


def _ms(vals: list[float]) -> dict:
    return {"mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0}


def run_study(seeds=SEEDS, budgets=BUDGETS, out="results/selection.json") -> dict:
    t0 = time.time()
    runs = [run_seed(s, budgets=budgets) for s in seeds]

    agg: dict[str, dict] = {}
    for k in budgets:
        key = str(k)
        agg[key] = {
            "target": {m: _ms([r["budgets"][k][m]["target_acc"] for r in runs])
                       for m in ("top", "random", "bottom")},
            "easy": {m: _ms([r["budgets"][k][m]["easy_acc"] for r in runs])
                     for m in ("top", "random", "bottom")},
            "top_frac_target": _ms([r["budgets"][k]["top_frac_target"] for r in runs]),
            "rand_frac_target": _ms([r["budgets"][k]["rand_frac_target"] for r in runs]),
        }

    result = {
        "config": {"seeds": list(seeds), "budgets": list(budgets), "n_easy": 1500,
                   "n_pool": 800, "pretrain_steps": 800, "ft_steps": 150,
                   "lora_r": 4, "proj_dim": 256, "n_target_eval": 600,
                   "target_lo": 150},
        "no_ft": {"target": _ms([r["no_ft"]["target_acc"] for r in runs]),
                  "easy": _ms([r["no_ft"]["easy_acc"] for r in runs])},
        "target_rate_in_pool": _ms([r["target_rate_in_pool"] for r in runs]),
        "budgets": agg,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    print("LESS top-k vs random-k target accuracy (mean over seeds):")
    for k in budgets:
        tp = agg[str(k)]["target"]["top"]["mean"]
        rd = agg[str(k)]["target"]["random"]["mean"]
        bo = agg[str(k)]["target"]["bottom"]["mean"]
        print(f"  k={k:>4}: top {tp:.3f}  random {rd:.3f}  bottom {bo:.3f}")
    print(f"wrote {path} in {result['runtime_sec']}s")
    return result


if __name__ == "__main__":
    run_study()
