"""LESS-flavoured data selection (Xia et al., 2024 — arXiv 2402.04333), CPU-sized.

For every candidate example we take the gradient of its teacher-forced loss with
respect to the *trainable LoRA parameters only* (base frozen). That gradient is a
tune-aware influence signature. We cheapen it with a fixed random projection
(Johnson-Lindenstrauss), L2-normalise, and score each candidate by its cosine
similarity to the aggregate gradient over a small *target validation* batch.

Top-k by score should, at the same budget, teach a frozen model more about the
target capability than random-k (no signal) or bottom-k (anti-aligned). The whole
point of the study is to check whether that intuition survives at toy scale — and
report the honest number whether it does or not.
"""

from __future__ import annotations

import torch

from .data import Example
from .train import per_example_loss


def _adapter_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def grad_feature(model: torch.nn.Module, ex: Example) -> torch.Tensor:
    """Flat per-example gradient over trainable adapter params (base frozen)."""
    model.zero_grad(set_to_none=True)
    loss = per_example_loss(model, ex)
    params = _adapter_params(model)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    parts = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
             for g, p in zip(grads, params, strict=True)]
    return torch.cat(parts)


def random_projection(dim: int, out_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, out_dim, generator=g) / (out_dim ** 0.5)


def _normalize(m: torch.Tensor) -> torch.Tensor:
    return m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)


def feature_matrix(model, examples: list[Example], proj: torch.Tensor) -> torch.Tensor:
    raw = torch.stack([grad_feature(model, e) for e in examples])
    return _normalize(raw @ proj)


def target_direction(model, target_val: list[Example], proj: torch.Tensor) -> torch.Tensor:
    raw = torch.stack([grad_feature(model, e) for e in target_val]).mean(0)
    return raw @ proj


def scores(features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    d = direction / direction.norm().clamp_min(1e-12)
    return features @ d


def select_topk(score: torch.Tensor, k: int) -> list[int]:
    k = min(k, score.shape[0])
    return torch.topk(score, k).indices.tolist()


def select_bottomk(score: torch.Tensor, k: int) -> list[int]:
    k = min(k, score.shape[0])
    return torch.topk(score, k, largest=False).indices.tolist()


def select_random(score: torch.Tensor, k: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    k = min(k, score.shape[0])
    return torch.randperm(score.shape[0], generator=g)[:k].tolist()


def fraction_target(is_target: torch.Tensor, idx: list[int]) -> float:
    """Diagnostic: what share of a chosen subset actually lies on the target slice."""
    if not idx:
        return float("nan")
    picked = torch.as_tensor(idx)
    return float(is_target[picked].float().mean())
