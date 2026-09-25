"""Guard the README's hand-written size and cost claims.

The results block is byte-pinned to the committed JSON by
``test_readme_matches_results.py``; the figures in the prose above it ("~103k
parameters", "~11k of ~112k trainable", "only ~14% target-slice", "~2 CPU-minutes")
are the ones a reader takes on trust, so each is measured here instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasel.model import TinyTransformer
from datasel.train import attach_lora

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "results" / "selection.json").read_text(encoding="utf-8"))
TOLERANCE = 2000


def _model_pair():
    model = TinyTransformer()
    base = sum(p.numel() for p in model.parameters())
    trained = attach_lora(model, r=DATA["config"]["lora_r"])
    return base, trained, base + trained


def test_readme_base_parameter_claim():
    claim = int(re.search(r"~(\d+)k-parameter", README).group(1))
    base, _, _ = _model_pair()
    assert abs(base / 1000 - claim) * 1000 <= TOLERANCE, (
        f"README says ~{claim}k parameters, the model is {base:,}")


def test_readme_trainable_fraction_claim():
    m = re.search(r"~([\d.]+)k of ~(\d+)k params\s*\n?\s*\(~(\d+)%\) are trainable",
                  README)
    assert m, "the LoRA size sentence was reworded; update this test with it"
    trained_k, total_k, pct = float(m.group(1)), int(m.group(2)), int(m.group(3))
    _, trained, total = _model_pair()
    assert abs(trained / 1000 - trained_k) * 1000 <= TOLERANCE, (
        f"README says ~{trained_k}k trainable, attaching LoRA r="
        f"{DATA['config']['lora_r']} gives {trained:,}")
    assert abs(total / 1000 - total_k) * 1000 <= TOLERANCE
    assert abs(100 * trained / total - pct) <= 1.5, (
        f"README says ~{pct}% trainable, it is {100 * trained / total:.1f}%")


def test_readme_target_share_claim():
    claim = int(re.search(r"only ~(\d+)% are target-slice", README).group(1))
    actual = DATA["target_rate_in_pool"]["mean"] * 100
    assert abs(actual - claim) <= 1.0, (
        f"README says ~{claim}% of the pool is target-slice; the artifact measured "
        f"{actual:.1f}%")


def test_readme_runtime_claim_holds_for_the_committed_run():
    claim = re.search(r"in ~(\d+) CPU-minutes", README)
    assert claim, "the runtime sentence was reworded; update this test with it"
    assert DATA["runtime_sec"] <= int(claim.group(1)) * 60, (
        f"README promises ~{claim.group(1)} CPU-minutes but the published run took "
        f"{DATA['runtime_sec']}s")
