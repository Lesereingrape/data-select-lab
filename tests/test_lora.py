"""Tests for the LoRA adapter and injection helper."""

from __future__ import annotations

import torch
import torch.nn as nn

from datasel.lora import LoRALinear, add_lora, trainable_param_count
from datasel.model import TinyTransformer


def test_lora_output_shape_matches_base():
    base = nn.Linear(8, 6)
    layer = LoRALinear(base, r=3, alpha=6.0)
    x = torch.randn(4, 8)
    assert layer(x).shape == (4, 6)


def test_lora_starts_near_identity():
    base = nn.Linear(16, 16)
    layer = LoRALinear(base, r=4, alpha=8.0, init=0.01)
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), base(x), atol=0.05)


def test_add_lora_freezes_base_linears_and_makes_adapters_trainable():
    model = TinyTransformer(d_model=32, n_head=4, n_layer=1)
    before = trainable_param_count(model)
    created = add_lora(model, r=4, alpha=8.0)
    # adapters are a small fraction of the full parameter budget
    assert trainable_param_count(model) < before
    for m in created:
        assert m.A.requires_grad and m.B.requires_grad
        assert not m.base.weight.requires_grad  # wrapped base frozen
    # add_lora only touches Linear layers; embeddings are unaffected here
    assert model.tok.weight.requires_grad


def test_add_lora_wraps_every_linear():
    model = TinyTransformer(d_model=32, n_head=4, n_layer=2)
    n_linear = sum(isinstance(m, nn.Linear) for m in model.modules())
    created = add_lora(model, r=4, alpha=8.0)
    assert len(created) == n_linear
    # each wrapped module exposes trainable A and B
    for m in created:
        assert m.A.requires_grad and m.B.requires_grad


def test_attach_lora_freezes_everything_but_adapters():
    from datasel.train import attach_lora

    model = TinyTransformer(d_model=32, n_head=4, n_layer=1)
    trainable = attach_lora(model, r=4, alpha=8.0)
    assert not model.tok.weight.requires_grad  # embeddings frozen
    # every trainable param is a LoRA A or B matrix
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names
    assert all(n.endswith(".A") or n.endswith(".B") for n in trainable_names)
    assert trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)
