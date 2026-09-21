"""Tests for the LESS gradient features and selection helpers."""

from __future__ import annotations

import random

import torch

from datasel.data import Example, gen_easy, gen_target
from datasel.less import (
    feature_matrix,
    grad_feature,
    random_projection,
    scores,
    select_bottomk,
    select_random,
    select_topk,
    target_direction,
)
from datasel.model import TinyTransformer
from datasel.pipeline import _adapter_dim
from datasel.train import attach_lora


def _tuned_base(seed: int = 0) -> TinyTransformer:
    torch.manual_seed(seed)
    model = TinyTransformer(d_model=32, n_head=4, n_layer=1)
    attach_lora(model, r=4, alpha=8.0)
    return model


def test_grad_feature_length_equals_adapter_dim():
    model = _tuned_base()
    g = grad_feature(model, Example(21, 33))
    assert g.numel() == _adapter_dim(model)


def test_grad_features_are_nonzero():
    model = _tuned_base()
    g = grad_feature(model, Example(21, 33))
    assert g.abs().sum() > 0


def test_random_projection_is_deterministic():
    w1 = random_projection(120, 32, seed=5)
    w2 = random_projection(120, 32, seed=5)
    assert torch.equal(w1, w2)
    assert w1.shape == (120, 32)


def test_select_topk_and_bottomk_are_extremes_of_scores():
    score = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7])
    top = select_topk(score, 2)
    bot = select_bottomk(score, 2)
    assert set(top) == {1, 4}  # the two largest
    assert set(bot) == {0, 3}  # the two smallest
    assert len(set(select_random(score, 3, seed=1))) == 3  # distinct indices


def test_target_examples_score_above_easy_examples():
    """Core property: LESS ranks on-target data above carry-free data."""
    model = _tuned_base(seed=2)
    rng = random.Random(2)
    pool = gen_easy(40, rng) + gen_target(40, rng)
    proj = random_projection(_adapter_dim(model), 64, seed=11)
    feats = feature_matrix(model, pool, proj)
    direction = target_direction(model, gen_target(20, random.Random(9)), proj)
    score = scores(feats, direction)
    is_target = torch.tensor([e.is_target for e in pool])
    top = select_topk(score, 20)
    frac_top = is_target[top].float().mean().item()
    # top-k should be clearly enriched in target examples (pool base rate 0.5)
    assert frac_top >= 0.6
