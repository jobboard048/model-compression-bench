# Model compression comparison bench

Compare **knowledge distillation**, **quantization**, and **pruning** on the same small CNN and dataset.

## Commands

From the repo root (Windows, macOS, or Linux):

```text
python scripts/setup.py
python scripts/test.py
python scripts/run.py
```

`run.py` is the experiment. The article is [`docs/blog-post.md`](docs/blog-post.md). [`notebooks/results.ipynb`](notebooks/results.ipynb) only plots the saved CSVs; it does not train. Open it from the repo root after `python scripts/run.py`. Jupyter is optional and is not installed by `setup.py`.

Artifacts in `results/fashion_mnist/` (or `results/cifar10/`):

| File | What it is |
| --- | --- |
| `run.log` | full terminal output (epoch lines, warnings, table) |
| `run_summary.json` | dataset, device, selected epoch count |
| `teacher_curve.csv` | teacher train loss and val accuracy each epoch |
| `student_ce_curve.csv` | small-student train loss each epoch |
| `student_kd_curve.csv` | distillation train loss each epoch |
| `prune_finetune_curve.csv` | prune fine-tune loss at the configured amount |
| `prune_sparsity_curve.csv` | test accuracy vs sparsity (`0.0` is fine-tune with no prune) |
| `comparison.csv` / `comparison.json` | accuracy / size / latency |
| `per_class_accuracy.csv` | test accuracy per class |
| `*.pt` | checkpoints |

```text
python scripts/run.py --dataset cifar10
python scripts/run.py --epochs 40
python scripts/run.py --prune-only
```

`--prune-only` reloads `teacher_fp32.pt` and reruns pruning only (mask held through fine-tune, plus a sparsity curve). It writes `prune_rerun.log` and does not overwrite `run.log`.

## What is compared

| Variant | Technique |
| --- | --- |
| `teacher_fp32` | baseline |
| `student_ce` | smaller model, hard labels |
| `student_kd` | distillation |
| `teacher_pruned` | unstructured L1 at `prune.amount`, mask held through fine-tune |
| `teacher_dynamic_int8` | dynamic INT8 on Linear layers |
| `teacher_static_int8` | static INT8 if the backend works |

Shared metrics: top-1 accuracy, params, sparsity, checkpoint size, batch latency.

## Layout

- `scripts/setup.py` — create `.venv` and install deps
- `scripts/test.py` — unit tests (no dataset)
- `scripts/run.py` — the experiment
- `notebooks/results.ipynb` — plots saved results (optional; needs Jupyter)
- `src/compression_bench/` — models, train, distill, prune, quantize, eval
