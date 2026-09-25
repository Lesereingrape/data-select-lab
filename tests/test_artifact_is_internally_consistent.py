"""The committed aggregates must be recomputable from the committed per-seed cells.

``results/selection.json`` stores both the mean/std summary and the raw per-seed
accuracy behind every cell, so this test re-derives each published aggregate from
those cells and fails if the two disagree — which is the difference between an
artifact that can be audited and one that has to be believed.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "selection.json").read_text(encoding="utf-8"))
STRATEGIES = ("top", "random", "bottom")


def _ms(vals):
    return {"mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0}


def _seeds():
    return [DATA["per_seed"][str(s)] for s in DATA["config"]["seeds"]]


def test_config_lists_the_seeds_and_budgets_the_artifact_contains():
    assert sorted(DATA["per_seed"]) == sorted(str(s) for s in DATA["config"]["seeds"])
    assert sorted(DATA["budgets"]) == sorted(str(k) for k in DATA["config"]["budgets"])


def test_every_budget_cell_matches_its_per_seed_rows():
    runs = _seeds()
    for k in DATA["config"]["budgets"]:
        published, raw = DATA["budgets"][str(k)], runs[0]["budgets"][str(k)]
        assert set(published) == {"target", "easy", "top_frac_target", "rand_frac_target"}
        assert set(raw) == {*STRATEGIES, "top_frac_target", "rand_frac_target"}
        for metric in ("target", "easy"):
            field = "target_acc" if metric == "target" else "easy_acc"
            for strat in STRATEGIES:
                assert published[metric][strat] == _ms(
                    [r["budgets"][str(k)][strat][field] for r in runs]
                ), (k, metric, strat)
        for frac in ("top_frac_target", "rand_frac_target"):
            assert published[frac] == _ms([r["budgets"][str(k)][frac] for r in runs]), (k, frac)


def test_headline_aggregates_match_their_per_seed_rows():
    runs = _seeds()
    for name, field in (("target", "target_acc"), ("easy", "easy_acc")):
        assert DATA["no_ft"][name] == _ms([r["no_ft"][field] for r in runs]), name
    assert DATA["target_rate_in_pool"] == _ms([r["target_rate_in_pool"] for r in runs])


def test_environment_records_where_the_numbers_came_from():
    env = DATA["environment"]
    for key in ("python", "platform", "torch", "device"):
        assert isinstance(env[key], str) and env[key], key
