"""Tests for the addition task, verifier, and difficulty slices."""

from __future__ import annotations

import random

from datasel.data import (
    TARGET_LO,
    Example,
    digit_token,
    gen_easy,
    gen_pool,
    gen_target,
    parse_answer,
    verify,
)


def test_verify_accepts_correct_chain():
    ex = Example(57, 68)  # 125
    assert verify(ex, ex.cot())


def test_verify_rejects_wrong_answer():
    ex = Example(57, 68)
    bad = ex.cot()
    bad[-1] = digit_token(0)  # corrupt final carry -> answer 25, not 125
    assert not verify(ex, bad)


def test_verify_rejects_malformed():
    ex = Example(12, 30)
    assert not verify(ex, [1, 2, 3])  # too short
    assert parse_answer([999] * 5) is None


def test_is_target_matches_sum_threshold():
    assert Example(80, 80).is_target  # 160 >= 150
    assert not Example(20, 30).is_target  # 50


def test_generated_slices_respect_their_ranges():
    rng = random.Random(0)
    for e in gen_easy(50, rng):
        assert e.target < 90
    for e in gen_target(50, rng):
        assert e.target >= TARGET_LO
    for e in gen_pool(200, rng):
        assert 20 <= e.target <= 198


def test_pool_and_target_agree_on_is_target():
    rng = random.Random(1)
    for e in gen_pool(300, rng):
        assert e.is_target == (e.target >= TARGET_LO)
