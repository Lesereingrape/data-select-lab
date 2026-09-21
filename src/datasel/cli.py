"""Command line entry point: ``datasel``.

    datasel demo    # one seeded run at two budgets, prints the selection comparison
    datasel study   # full multi-seed study -> results/selection.json
"""

from __future__ import annotations

import argparse

from .pipeline import run_seed


def _demo(seed: int) -> None:
    r = run_seed(seed, budgets=(64, 256))
    print(f"base (no fine-tune): target {r['no_ft']['target_acc']:.3f} "
          f"| easy {r['no_ft']['easy_acc']:.3f}")
    print(f"target share in pool: {r['target_rate_in_pool']:.3f}\n")
    print(f"{'k':>4} {'LESS top-k':>12} {'random-k':>10} {'bottom-k':>10} "
          f"{'top#target':>11}")
    for k, v in r["budgets"].items():
        print(f"{k:>4} {v['top']['target_acc']:>12.3f} {v['random']['target_acc']:>10.3f} "
              f"{v['bottom']['target_acc']:>10.3f} {v['top_frac_target']:>11.2f}")


def _study() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
    from run_study import run_study

    run_study()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datasel")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="single seeded selection run")
    d.add_argument("--seed", type=int, default=0)
    sub.add_parser("study", help="full study -> results/selection.json")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        _demo(args.seed)
    else:
        _study()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
