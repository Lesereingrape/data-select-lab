"""End-to-end LoRA + LESS selection experiment for one seed.

The pipeline isolates the *selection* variable: the base model, the adapter init,
the fine-tuning step budget and the held-out target slice are all held fixed, and
only which pool examples you fine-tune on changes (LESS top-k vs random-k vs
bottom-k). Any accuracy gap on the target slice is therefore attributable to the
data-selection rule, not to extra compute.
"""

from __future__ import annotations

import copy
import random

import torch

from .data import Example, gen_easy, gen_pool, gen_target
from .less import (
    feature_matrix,
    fraction_target,
    random_projection,
    scores,
    select_bottomk,
    select_random,
    select_topk,
    target_direction,
)
from .model import TinyTransformer
from .train import attach_lora, evaluate, finetune_lora, pretrain


def _adapter_dim(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_seed(seed: int, *, budgets=(32, 64, 128, 256), n_easy=1500, n_pool=800,
             pretrain_steps=800, ft_steps=150, r=4, alpha=8.0, proj_dim=256,
             n_target_eval=600) -> dict:
    """Pre-train a base, then compare selection strategies at each budget."""
    rng = random.Random(seed)

    # 1. base model on easy, carry-free sums (this becomes the frozen PEFT base)
    torch.manual_seed(seed)
    base = TinyTransformer()
    pretrain(base, gen_easy(n_easy, rng), steps=pretrain_steps, seed=seed)

    # 2. freeze base, add near-identity LoRA adapters
    attach_lora(base, r=r, alpha=alpha)

    # 3. LESS scores over the broad pool against the target-capability direction
    pool = gen_pool(n_pool, rng)
    target_val = gen_target(64, random.Random(seed + 999))
    proj = random_projection(_adapter_dim(base), proj_dim, seed=seed + 7)
    feats = feature_matrix(base, pool, proj)
    direction = target_direction(base, target_val, proj)
    score = scores(feats, direction)
    is_target = torch.tensor([e.is_target for e in pool])

    # 4. held-out evaluations (never used for selection or training)
    target_test = gen_target(n_target_eval, random.Random(seed + 55))
    easy_test = gen_easy(300, random.Random(seed + 66))

    def measure(subset: list[Example], ft_seed: int) -> dict:
        model = copy.deepcopy(base)  # identical adapter init for every strategy
        finetune_lora(model, subset, steps=ft_steps, seed=ft_seed)
        return {"target_acc": round(evaluate(model, target_test)["target_acc"], 4),
                "easy_acc": round(evaluate(model, easy_test)["acc"], 4)}

    no_ft = {"target_acc": round(evaluate(base, target_test)["target_acc"], 4),
             "easy_acc": round(evaluate(base, easy_test)["acc"], 4)}

    per_budget = {}
    for k in budgets:
        top = select_topk(score, k)
        rnd = select_random(score, k, seed=seed)
        bot = select_bottomk(score, k)
        per_budget[k] = {
            "top": measure([pool[i] for i in top], ft_seed=seed),
            "random": measure([pool[i] for i in rnd], ft_seed=seed),
            "bottom": measure([pool[i] for i in bot], ft_seed=seed),
            "top_frac_target": round(fraction_target(is_target, top), 3),
            "rand_frac_target": round(fraction_target(is_target, rnd), 3),
        }
    return {"seed": seed, "no_ft": no_ft, "budgets": per_budget,
            "target_rate_in_pool": round(float(is_target.float().mean()), 3)}
