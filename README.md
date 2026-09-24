# Model compression comparison bench

Compare **knowledge distillation**, **quantization**, and **pruning** on the same small CNN and dataset.

## Commands

From the repo root (Windows, macOS, or Linux):

```text
python scripts/setup.py
python scripts/test.py
python scripts/run.py
```

`run.py` is the experiment. Draft of the article: [`docs/blog-post.md`](docs/blog-post.md). After a full run, replace the smoke-test table in that file with `results/fashion_mnist/comparison.csv`.

Artifacts in `results/fashion_mnist/` (or `results/cifar10/`):

| File | What it is |
| --- | --- |
| `samples.png` | labeled 2×5 class grid |
| `run_summary.json` | dataset, device, selected epoch count |
| `teacher_curve.csv` | teacher train loss and val accuracy each epoch |
| `student_ce_curve.csv` | small-student train loss each epoch |
| `student_kd_curve.csv` | distillation train loss each epoch |
| `prune_finetune_curve.csv` | prune fine-tune loss |
| `comparison.csv` / `comparison.json` | accuracy / size / latency |
| `per_class_accuracy.csv` | test accuracy per class |
| `*.pt` | checkpoints |

```text
python scripts/run.py --dataset cifar10
python scripts/run.py --epochs 40
```

## What is compared

| Variant | Technique |
| --- | --- |
| `teacher_fp32` | baseline |
| `student_ce` | smaller model, hard labels |
| `student_kd` | distillation |
| `teacher_pruned` | unstructured prune + fine-tune |
| `teacher_dynamic_int8` | dynamic INT8 on Linear layers |
| `teacher_static_int8` | static INT8 if the backend works |

Shared metrics: top-1 accuracy, params, sparsity, checkpoint size, batch latency.

## Layout

- `scripts/setup.py` — create `.venv` and install deps
- `scripts/test.py` — unit tests (no dataset)
- `scripts/run.py` — the experiment
- `src/compression_bench/` — models, train, distill, prune, quantize, eval
