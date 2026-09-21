"""Pre-training, LoRA fine-tuning, and slice-wise evaluation (CPU PyTorch).

The lifecycle used by the study is:

1. ``pretrain`` a plain model on small carry-free sums (easy data) until it solves
   those — this is the frozen *base*.
2. ``attach_lora`` freezes the base and adds trainable low-rank adapters (reset to
   near-identity), so downstream work is pure PEFT.
3. ``finetune_lora`` trains **only** the adapters on a selected subset of the broad
   pool for a fixed number of steps — the same budget across selection strategies.
4. ``evaluate`` scores greedy self-generated CoT with the exact verifier and reports
   accuracy on the held-out *target* capability slice and overall.
"""

from __future__ import annotations

import torch

from .data import Example, verify
from .lora import add_lora, trainable_param_count
from .model import TinyTransformer


def _pair_tensors(examples: list[Example]):
    prompts = torch.tensor([e.prompt() for e in examples], dtype=torch.long)
    cots = torch.tensor([e.cot() for e in examples], dtype=torch.long)
    return prompts, cots


def pretrain(model: TinyTransformer, examples: list[Example], *, steps: int = 800,
             batch: int = 128, lr: float = 3e-3, seed: int = 0) -> list[float]:
    """Train the whole model on ``examples`` (easy sums). Returns logged losses."""
    torch.manual_seed(seed)
    prompts, cots = _pair_tensors(examples)
    n = prompts.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        loss = model.sft_loss(prompts[idx], cots[idx]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append(round(float(loss.item()), 4))
    return losses


def attach_lora(model: TinyTransformer, *, r: int = 4, alpha: float = 8.0) -> int:
    """Freeze the entire base, then wrap every Linear with a near-identity adapter.

    Embeddings, layer norms and all base weights are frozen first, so only the
    adapter ``A``/``B`` matrices remain trainable — a genuine PEFT subspace for
    both the LESS gradient features and the fine-tuning step.
    """
    for p in model.parameters():
        p.requires_grad = False
    add_lora(model, r=r, alpha=alpha)
    return trainable_param_count(model)


def finetune_lora(model: TinyTransformer, examples: list[Example], *, steps: int = 200,
                  lr: float = 5e-3, seed: int = 0) -> float:
    """Train only the adapter parameters on ``examples``; return final loss."""
    adapter = [p for p in model.parameters() if p.requires_grad]
    if not adapter:
        raise RuntimeError("no trainable parameters — call attach_lora first")
    torch.manual_seed(seed)
    prompts, cots = _pair_tensors(examples)
    n = prompts.shape[0]
    opt = torch.optim.AdamW(adapter, lr=lr)
    loss_val = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (min(32, n),))
        loss = model.sft_loss(prompts[idx], cots[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = float(loss.item())
    return loss_val


@torch.no_grad()
def evaluate(model: TinyTransformer, examples: list[Example], *, batch: int = 512) -> dict:
    model.eval()
    prompts, _ = _pair_tensors(examples)
    ok = 0
    tgt_ok = tgt_n = 0
    for start in range(0, len(examples), batch):
        chunk = prompts[start:start + batch]
        cots = model.generate_cot(chunk, n_tokens=5, greedy=True).tolist()
        for ex, cot in zip(examples[start:start + batch], cots, strict=True):
            good = verify(ex, cot)
            ok += int(good)
            if ex.is_target:
                tgt_ok += int(good)
                tgt_n += 1
    n = len(examples)
    return {
        "n": n,
        "acc": ok / n,
        "target_acc": (tgt_ok / tgt_n) if tgt_n else float("nan"),
        "target_n": tgt_n,
    }


def per_example_loss(model: TinyTransformer, ex: Example) -> torch.Tensor:
    """Scalar teacher-forced CE for a single example (used for gradient features)."""
    prompts, cots = _pair_tensors([ex])
    return model.sft_loss(prompts, cots).mean()
