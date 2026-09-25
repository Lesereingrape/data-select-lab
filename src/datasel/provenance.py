"""Provenance for the published ``results/selection.json``.

The study's numbers are deterministic given a fixed torch build and thread count, so
"reproducible" is only a meaningful claim if the artifact says what produced it. This
module owns that record; the study script, the README renderer and the tests all read
it from here rather than restating it.
"""

from __future__ import annotations

import platform
import sys

import torch


def environment() -> dict:
    """The parts of a machine that change CPU float reduction order."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }
