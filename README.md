# datasel — a CPU-only LoRA/PEFT + LESS data-selection study

**datasel** reproduces, at honest toy scale, one of the most practical ideas in
LLM post-training: **you don't need all the data — you need the *right* few
examples.** It implements **LoRA** (Hu et al., 2106.09685) from scratch and a
**LESS-style** influence selector (Xia et al., 2402.04333) on top of it, then
measures — with a *held-out capability the base model genuinely cannot do* —
whether gradient-influence data selection actually buys more than random
sampling at the same budget.

![ci](https://github.com/Lesereingrape/data-select-lab/actions/workflows/ci.yml/badge.svg)

> Every number below is produced by `experiments/run_study.py` in ~2 CPU-minutes
> and committed as `results/selection.json`; the README tables are rendered from
> that file by `experiments/make_report.py`. Nothing is quoted from a paper, and
> the downside (catastrophic forgetting) is reported, not hidden.

## The setup

- **Task:** a from-scratch ~103k-parameter decoder-only transformer learns
  digit-by-digit column addition with an explicit carry chain. The answer is
  reconstructed and checked against `a+b`, so accuracy is measured by an *exact
  verifier*, never by the model's own opinion.
- **Base model** is pre-trained *only on carry-free sums* (`a+b < 90`) and then
  **frozen**. It therefore scores ~0 on the held-out **target capability**
  (`a+b >= 150`) — real headroom for a selection method to demonstrate.
- **Adaptation pool:** a broad mix of sums; only ~14% are target-slice examples.
- **PEFT:** LoRA adapters (r=4) on every linear layer — ~8.5k of ~112k params
  (~8%) are trainable, so per-example gradients are cheap.

## The method (LESS, sized for a laptop)

For each pool example we take the gradient of its loss w.r.t. the **trainable
LoRA parameters**, project it down with a fixed Johnson–Lindenstrauss random
projection, L2-normalise, and score the example by its **cosine similarity to
the aggregate gradient over a small target-validation batch**. High score ⇒
"training on me moves you toward the target capability." We then fine-tune
LoRA on:

- **LESS top-k** (highest influence),
- **random-k** (no signal),
- **bottom-k** (anti-aligned),

at matched budgets and matched optimisation steps, so any gap is attributable to
the *selection rule*, not extra compute.

## Quickstart

```bash
pip install -e .            # torch is the only dependency
datasel demo                # one seeded run: LESS top-k vs random vs bottom
datasel study               # full 3-seed study -> results/selection.json
python experiments/make_report.py --write  # splice the README block from the JSON
```

## What it measures

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/selection.json`](results/selection.json); the tables are rendered by `experiments/make_report.py`. 3 seeds, LoRA r=4, matched fine-tuning steps.*

- base model (pre-trained on carry-free sums, then frozen): target accuracy **0.003**, easy accuracy 1.000 — it genuinely cannot do the held-out target capability (a+b >= 150).
- adaptation pool has 13.9% target-slice examples; selection budgets are k = [32, 64, 128, 256] LoRA fine-tuning examples, mean over seeds [0, 1, 2].
- measured under: Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu — `experiments/run_study.py --out /tmp/again.json` reproduces every figure in this file inside that environment (its `runtime_sec` is the one field a rerun is allowed to move), and nowhere else promises to

### Target accuracy vs selection budget (held-out, mean ± population std over seeds)

| k | LESS top-k | random-k | bottom-k (anti) |
|---:|-----------:|---------:|----------------:|
| 32 | **0.416 ± 0.123** | 0.108 ± 0.031 | 0.061 ± 0.085 |
| 64 | **0.506 ± 0.123** | 0.206 ± 0.051 | 0.097 ± 0.061 |
| 128 | **0.551 ± 0.035** | 0.279 ± 0.116 | 0.122 ± 0.058 |
| 256 | **0.623 ± 0.009** | 0.527 ± 0.198 | 0.187 ± 0.021 |

LESS top-k climbs 0.416 -> 0.623 as the budget grows. random-k also improves with budget but stays behind top-k at every k; it reaches 0.527 where LESS hits 0.623. The gap is widest at k=32 (0.416 vs 0.108, i.e. 0.307), and that is larger than the seed-to-seed spread there (±0.123), so the ordering is not a seed artefact. LESS is also calmer across seeds at the largest k (±0.009 vs ±0.198). bottom-k stays lowest throughout (0.061 at k=32, 0.187 at k=256).

### Mechanism check — what the selector actually picks

| k | target share of LESS top-k | target share of random-k |
|---:|---------------------------:|-------------------------:|
| 32 | 0.75 | 0.12 |
| 64 | 0.65 | 0.10 |
| 128 | 0.43 | 0.12 |
| 256 | 0.29 | 0.15 |

The pool is only 13.9% target examples, yet LESS top-k is far more target-concentrated than random sampling — the cosine-to-target-gradient score is finding the right data, not just resampling the pool.

### The honest cost — catastrophic forgetting of the base skill

| k | easy accuracy after LESS top-k | easy accuracy after random-k |
|---:|-------------------------------:|-----------------------------:|
| 32 | 0.028 | 0.568 |
| 64 | 0.012 | 0.534 |
| 128 | 0.017 | 0.849 |
| 256 | 0.061 | 0.956 |

Fine-tuning adapters on narrowly *target-selected* data buys the new capability at the price of the old one: easy accuracy collapses to ~3%. Selecting *for influence* is efficient but not free — this is a real, reproducible downside, reported rather than hidden.
<!-- RESULTS:END -->

## Layout

```
src/datasel/
  data.py     # addition task + exact verifier + easy/pool/target difficulty slices
  model.py    # from-scratch decoder-only transformer with LoRA-injectable linears
  lora.py     # LoRALinear + add_lora (freeze base, attach low-rank adapters)
  less.py     # per-example LoRA-gradient features + random projection + cosine select
  train.py    # pretrain / attach_lora / finetune_lora / exact slice-wise evaluate
  pipeline.py # the isolated per-seed experiment (selection is the only variable)
  provenance.py # the python/platform/torch/thread/device this run was measured on
experiments/
  run_study.py   # 3 seeds x 4 budgets x 3 strategies -> results/selection.json
                 # (--out writes a scratch artifact so a rerun can be diffed)
  make_report.py # render README tables straight from the committed JSON (--write splices)
tests/           # verifier exactness, LoRA freeze/identity, LESS ranking properties,
                 # plus the three integrity guards below
```

## Reproducing and honesty

`results/selection.json` commits both the aggregates and the raw per-seed cells, so
nothing in the tables above has to be taken on trust:

- `tests/test_artifact_is_internally_consistent.py` recomputes every published
  mean and population std from the per-seed rows in the same file.
- `tests/test_readme_matches_results.py` byte-compares the README block against
  what `make_report.py` renders, so prose and numbers cannot drift apart.
- `tests/test_readme_size_claims.py` measures the hand-written figures in the setup
  section (base parameters, trainable LoRA parameters, target share of the pool,
  wall-clock) against the code and the artifact rather than trusting them — it is
  what caught this README over-reporting the adapter size.

To check a rerun against the published artifact:

```bash
python experiments/run_study.py --out /tmp/again.json   # leaves results/ untouched
python - <<'PY'
import json
a = json.load(open("results/selection.json", encoding="utf-8"))
b = json.load(open("/tmp/again.json", encoding="utf-8"))
allowed = {"environment", "per_seed", "runtime_sec"}
print(sorted(set(a) ^ set(b)) or "no new keys;",
      [k for k in set(a) & set(b) - allowed if a[k] != b[k]] or "every figure matched")
PY
```

The rerun behind the current tables did exactly that: written to a scratch path and
compared field by field, it reproduced every published figure and all 72 per-seed
accuracy cells identically, and only `runtime_sec` moved (102.0s against the published
107.4s, because the machine was busier). CPU float reduction order follows the thread
count and torch build recorded in `environment`, so bit-exactness is promised inside
that environment and nowhere else.

## Honest limitations

- **Toy scale on purpose.** A 2-layer transformer on 2-digit addition; LESS in
  production runs on 7B models with real instruction data. This is a faithful
  *mechanism* demo, not a SOTA result.
- **Catastrophic forgetting is real and reported.** Narrowly target-selected
  fine-tuning restores the target capability but erodes the base skill — see the
  forgetting table. Influence-based selection is *efficient*, not *free*; a
  production pipeline would mix in replay, which this demo flags but does not solve.
- **Seed noise matters.** Some cells (e.g. LESS at small k) vary across the 3
  seeds; we print ±std and avoid claiming wins that sit inside that spread.
- Everything is fixed-seed, CPU-only and reproducible: `pytest` + `datasel study`
  regenerate every figure.

## License

MIT
