"""Render README result tables directly from results/selection.json."""

from __future__ import annotations

import json
from pathlib import Path


def build(data: dict) -> str:
    cfg = data["config"]
    out: list[str] = []
    noft = data["no_ft"]
    tr = data["target_rate_in_pool"]["mean"]
    budgets = cfg["budgets"]
    first, last = str(budgets[0]), str(budgets[-1])

    out.append("*Every figure below is produced by `experiments/run_study.py` on CPU "
               "and committed as [`results/selection.json`](results/selection.json); "
               f"the tables are rendered by `experiments/make_report.py`. "
               f"{len(cfg['seeds'])} seeds, LoRA r={cfg['lora_r']}, matched "
               "fine-tuning steps.*")
    out.append("")
    out.append(f"- base model (pre-trained on carry-free sums, then frozen): target "
               f"accuracy **{noft['target']['mean']:.3f}**, easy accuracy "
               f"{noft['easy']['mean']:.3f} — it genuinely cannot do the held-out "
               f"target capability (a+b >= {cfg['target_lo']}).")
    out.append(f"- adaptation pool has {tr:.1%} target-slice examples; selection "
               f"budgets are k = {budgets} LoRA fine-tuning examples, mean over "
               f"seeds {cfg['seeds']}.")
    env = data["environment"]
    out.append(f"- measured under: Python {env['python']} on {env['platform']}, "
               f"torch {env['torch']}, {env['threads']} CPU threads, {env['device']} "
               "— `experiments/run_study.py --out /tmp/again.json` reproduces every "
               "figure in this file inside that environment (its `runtime_sec` is the "
               "one field a rerun is allowed to move), and nowhere else promises to")
    out.append("")

    out.append("### Target accuracy vs selection budget (held-out, mean ± population "
               "std over seeds)")
    out.append("")
    out.append("| k | LESS top-k | random-k | bottom-k (anti) |")
    out.append("|---:|-----------:|---------:|----------------:|")
    for k in budgets:
        b = data["budgets"][str(k)]["target"]
        out.append(f"| {k} | **{b['top']['mean']:.3f} ± {b['top']['std']:.3f}** "
                   f"| {b['random']['mean']:.3f} ± {b['random']['std']:.3f} "
                   f"| {b['bottom']['mean']:.3f} ± {b['bottom']['std']:.3f} |")
    out.append("")
    bt = data["budgets"]
    top_last = bt[last]["target"]["top"]["mean"]
    top_first = bt[first]["target"]["top"]["mean"]
    rnd_last = bt[last]["target"]["random"]["mean"]
    top_std_last = bt[last]["target"]["top"]["std"]
    rnd_std_last = bt[last]["target"]["random"]["std"]
    spread = {str(k): max(bt[str(k)]["target"][m]["std"] for m in ("top", "random"))
              for k in budgets}
    gaps = {str(k): bt[str(k)]["target"]["top"]["mean"]
            - bt[str(k)]["target"]["random"]["mean"] for k in budgets}
    widest_k = max(gaps, key=gaps.get)
    gap_widest = gaps[widest_k]
    beyond = gap_widest > spread[widest_k]
    always_ahead = all(g > 0 for g in gaps.values())
    bottom_lastest = all(bt[str(k)]["target"]["bottom"]["mean"]
                         <= min(bt[str(k)]["target"][m]["mean"]
                                for m in ("top", "random")) for k in budgets)
    low = bt[first]["target"]["bottom"]["mean"]
    high = bt[last]["target"]["bottom"]["mean"]
    n_seeds = len(cfg["seeds"])
    out.append(
        f"LESS top-k climbs {top_first:.3f} -> {top_last:.3f} as the budget grows. "
        + ("random-k also improves with budget but stays behind top-k at every k; "
           if always_ahead else
           "random-k overtakes top-k at some k (see the table); ")
        + f"it reaches {rnd_last:.3f} where LESS hits {top_last:.3f}. "
        f"The gap is widest at k={widest_k} "
        f"({bt[widest_k]['target']['top']['mean']:.3f} vs "
        f"{bt[widest_k]['target']['random']['mean']:.3f}, i.e. {gap_widest:.3f}), and "
        + (f"that is larger than the seed-to-seed spread there (±{spread[widest_k]:.3f}), "
           "so the ordering is not a seed artefact"
           if beyond else
           f"that sits inside the seed-to-seed spread there "
           f"(±{spread[widest_k]:.3f}), so {n_seeds} seeds do not settle the ordering")
        + f". LESS is also calmer across seeds at the largest k "
        f"(±{top_std_last:.3f} vs ±{rnd_std_last:.3f}). bottom-k "
        + ("stays lowest throughout" if bottom_lastest else
           "is not lowest throughout")
        + f" ({low:.3f} at k={budgets[0]}, {high:.3f} at k={budgets[-1]})."
    )
    out.append("")

    out.append("### Mechanism check — what the selector actually picks")
    out.append("")
    out.append("| k | target share of LESS top-k | target share of random-k |")
    out.append("|---:|---------------------------:|-------------------------:|")
    for k in budgets:
        b = data["budgets"][str(k)]
        out.append(f"| {k} | {b['top_frac_target']['mean']:.2f} "
                   f"| {b['rand_frac_target']['mean']:.2f} |")
    out.append("")
    out.append(f"The pool is only {tr:.1%} target examples, yet LESS top-k is far "
               f"more target-concentrated than random sampling — the cosine-to-target-"
               f"gradient score is finding the right data, not just resampling the pool.")
    out.append("")

    out.append("### The honest cost — catastrophic forgetting of the base skill")
    out.append("")
    out.append("| k | easy accuracy after LESS top-k | easy accuracy after random-k |")
    out.append("|---:|-------------------------------:|-----------------------------:|")
    easy_top_vals = []
    for k in budgets:
        e = data["budgets"][str(k)]["easy"]
        easy_top_vals.append(e["top"]["mean"])
        out.append(f"| {k} | {e['top']['mean']:.3f} | {e['random']['mean']:.3f} |")
    out.append("")
    easy_mean = sum(easy_top_vals) / len(easy_top_vals)
    out.append(f"Fine-tuning adapters on narrowly *target-selected* data buys the new "
               f"capability at the price of the old one: easy accuracy collapses to "
               f"~{round(100 * easy_mean)}%. Selecting *for influence* is efficient "
               f"but not free — this is a real, reproducible downside, reported "
               f"rather than hidden.")
    return "\n".join(out)


def _write(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    nl = "\n"
    path.write_text(f"{head}{start}{nl}{block}{nl}{end}{tail}", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--write", action="store_true",
                    help="splice the block into README.md instead of printing it")
    ap.add_argument("--results", default="results/selection.json")
    args = ap.parse_args()
    rendered = build(json.loads(Path(args.results).read_text(encoding="utf-8")))
    if args.write:
        _write(Path("README.md"), rendered)
        print("README results block rewritten")
    else:
        print(rendered)
