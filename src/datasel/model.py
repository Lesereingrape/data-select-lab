"""A nanoGPT-shaped decoder-only transformer, built for LoRA injection.

Attention is implemented with explicit nn.Linear projections (not
nn.MultiheadAttention) so that :func:`datasel.lora.add_lora` can wrap every
projection with a low-rank adapter — the trainable subspace LESS scores over.

The model learns the column-addition CoT from ``data.py``; ``generate_cot``
samples the model's own chain so accuracy is measured on real generated
arithmetic graded by the exact verifier, never on teacher forcing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NVOCAB, PAD


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.hd = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(z):
            return z.view(b, t, self.n_head, self.hd).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd) + mask
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 64, n_head: int = 4, n_layer: int = 2,
                 max_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok = nn.Embedding(NVOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NVOCAB)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def _mask(self, t: int, device) -> torch.Tensor:
        return torch.triu(torch.full((t, t), float("-inf"), device=device),
                          diagonal=1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        mask = self._mask(t, ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def sft_loss(self, prompt: torch.Tensor, cot: torch.Tensor) -> torch.Tensor:
        """Teacher-forced next-token CE over the CoT span only."""
        x = torch.cat([prompt, cot[:, :-1]], dim=1)
        logits = self.forward(x)
        p = prompt.shape[1]
        preds = logits[:, p - 1: p - 1 + cot.shape[1], :]
        return F.cross_entropy(preds.reshape(-1, NVOCAB), cot.reshape(-1),
                               reduction="none")

    @torch.no_grad()
    def generate_cot(self, prompt: torch.Tensor, n_tokens: int = 5,
                     greedy: bool = True) -> torch.Tensor:
        self.eval()
        x = prompt.clone()
        for _ in range(n_tokens):
            nxt = self.forward(x)[:, -1, :]
            if greedy:
                tok = nxt.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(F.softmax(nxt, dim=-1), 1)
            x = torch.cat([x, tok], dim=1)
        return x[:, prompt.shape[1]:]
