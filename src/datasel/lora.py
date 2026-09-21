"""A minimal, from-scratch LoRA (Hu et al., 2004 — arXiv 2106.09685).

LoRA freezes a base ``nn.Linear`` and adds a low-rank update
``y = W0 x + (alpha / r) * B A x`` with ``A: (r, in)``, ``B: (out, r)``. Only
``A`` and ``B`` train, so the trainable subspace is tiny — which is exactly what
makes per-example *gradient features* cheap enough to compute on CPU and use for
data selection in :mod:`datasel.less`.

We initialise both A and B small-but-nonzero (rather than B=0) so every adapter
parameter has a well-defined gradient at selection time; the product B@A is then
near zero, so the adapter starts as an identity on the frozen base.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0,
                 init: float = 0.01):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.empty(base.out_features, r))
        nn.init.normal_(self.A, std=init)
        nn.init.normal_(self.B, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.matmul(x @ self.A.t(), self.B.t())
        return self.base(x) + self.scaling * low

    def adapter_params(self):
        return (self.A, self.B)


def add_lora(root: nn.Module, r: int, alpha: float) -> list[LoRALinear]:
    """Replace every direct ``nn.Linear`` child of ``root`` (and its submodules)
    with a LoRA-wrapped equivalent. Returns the created adapters."""
    created: list[LoRALinear] = []
    for module in list(root.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, name, wrapped)
                created.append(wrapped)
    return created


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
