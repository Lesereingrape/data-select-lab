"""Verifiable column-addition with difficulty tiers, for data-selection studies.

Same toy task family as a from-scratch arithmetic model: two 2-digit operands,
and the model emits the digit-by-digit carry chain

    prompt:  a1 a0 + b1 b0 =
    target:  o0 c1 o1 c2 o2   (answer = o2 o1 o0)

The twist that makes *data selection* meaningful here: we deliberately hold out
a **target capability** — sums that require a final hundreds-carry, and especially
heavy ones (a+b >= TARGET_LO). A base model is pre-trained only on small, carry-free
sums, so it is weak exactly on that slice. The study then asks whether a handful of
finely-*selected* fine-tuning examples can buy more target accuracy than the same
budget picked at random.

Because the answer is reconstructed and checked against a+b, ``verify`` is a pure,
exact function — accuracy numbers are trustworthy, not model-opinion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD, BOS, EOS, PLUS, EQ = 0, 1, 2, 3, 4
DIGIT0 = 5
VOCAB = ["<pad>", "<bos>", "<eos>", "+", "="] + [str(d) for d in range(10)]
NVOCAB = len(VOCAB)

# The held-out "target capability": sums at least this large (all need a final carry).
TARGET_LO = 150


def digit_token(d: int) -> int:
    return DIGIT0 + d


def token_to_digit(tok: int) -> int:
    return tok - DIGIT0


@dataclass(frozen=True)
class Example:
    a: int
    b: int

    @property
    def target(self) -> int:
        return self.a + self.b

    def prompt(self) -> list[int]:
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        return [digit_token(a1), digit_token(a0), PLUS,
                digit_token(b1), digit_token(b0), EQ]

    def cot(self) -> list[int]:
        """Ground-truth o0 c1 o1 c2 o2 (5 tokens, no EOS)."""
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        s0 = a0 + b0
        o0, c1 = s0 % 10, s0 // 10
        s1 = a1 + b1 + c1
        o1, c2 = s1 % 10, s1 // 10
        o2 = c2
        return [digit_token(x) for x in (o0, c1, o1, c2, o2)]

    @property
    def is_target(self) -> bool:
        return self.target >= TARGET_LO


def parse_answer(cot_tokens: list[int]) -> int | None:
    if len(cot_tokens) < 5:
        return None
    if any(not (DIGIT0 <= t <= DIGIT0 + 9) for t in cot_tokens[:5]):
        return None
    o0, _c1, o1, _c2, o2 = (token_to_digit(t) for t in cot_tokens[:5])
    return 100 * o2 + 10 * o1 + o0


def verify(ex: Example, cot_tokens: list[int]) -> bool:
    """True iff the reconstructed answer equals a+b (the exact reward)."""
    ans = parse_answer(cot_tokens)
    return ans is not None and ans == ex.target


def gen_easy(n: int, rng: random.Random) -> list[Example]:
    """Carry-free-ish sums for base pre-training: a,b in [10,44] so a+b < 90."""
    return [Example(a=rng.randint(10, 44), b=rng.randint(10, 44)) for _ in range(n)]


def gen_pool(n: int, rng: random.Random) -> list[Example]:
    """Broad adaptation pool: a,b in [10,99]; mixes carry-free and target-heavy sums."""
    return [Example(a=rng.randint(10, 99), b=rng.randint(10, 99)) for _ in range(n)]


def gen_target(n: int, rng: random.Random) -> list[Example]:
    """The held-out target capability: sums >= TARGET_LO."""
    out: list[Example] = []
    while len(out) < n:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if a + b >= TARGET_LO:
            out.append(Example(a, b))
    return out
