# Model compression comparison bench

Compare **knowledge distillation**, **quantization**, and **pruning** on the same small CNN and dataset so results are comparable.

The default task is **CIFAR-10**, a standard 10-class image-classification benchmark: 50,000 training images and 10,000 test images, each 32 × 32 × 3. torchvision downloads it into `data/` (gitignored) when you run the experiment; the images are not committed to the repo.

Default YAML epochs (3 / 3 / 1) are a short local check, not a published accuracy number. Bump `epochs_teacher` / `epochs_student` / `epochs_finetune` before treating the table as a real result.

For a smaller CPU-only sweep, use `configs/fashion_mnist.yaml`.

## What is compared

| Variant | Technique | What it measures |
| --- | --- | --- |
| `teacher_fp32` | baseline | Accuracy / size / latency of the larger fp32 CNN |
| `student_ce` | smaller model, hard labels only | Capacity vs distillation |
| `student_kd` | distillation (KL + CE) | Whether soft teacher labels beat CE for the same student |
| `teacher_pruned` | global unstructured L1 prune (+ optional fine-tune) | Sparsity vs accuracy |
| `teacher_dynamic_int8` | dynamic INT8 on Linear layers | Easy quantization path |
| `teacher_static_int8` | post-training static quant (calibrate + convert) | Stronger INT8 path; skipped automatically if the backend fails |

Shared metrics: top-1 accuracy, parameter count, nonzero params, sparsity, serialized `state_dict` size (MB), batch latency p50/p95.

## Setup

```powershell
cd C:\Users\Admin\model-compression-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install pytest
```

## Sample grid

Save a labeled 2×5 class grid for the blog (downloads CIFAR-10 into `data/` if needed):

```powershell
python scripts/save_sample_grid.py --data-dir data --out results/cifar10_samples.png
```

## Run the comparison

From the project root (so `configs/` and `PYTHONPATH` via the editable install work):

```powershell
python scripts/run_comparison.py --config configs/default.yaml --out results
```

Fashion-MNIST (smaller, CPU-friendly):

```powershell
python scripts/run_comparison.py --config configs/fashion_mnist.yaml --out results
```

Static PTQ can fail on some Windows CPU builds. Use:

```powershell
python scripts/run_comparison.py --skip-static-quant
```

## Tests (no dataset download)

```powershell
pytest tests -q
```

## Layout

- `src/compression_bench/models.py` — teacher/student CNNs with quant stubs
- `src/compression_bench/distill.py` — temperature-scaled KD loss
- `src/compression_bench/prune.py` — global unstructured magnitude prune
- `src/compression_bench/quantize.py` — dynamic + static PTQ
- `src/compression_bench/evaluate.py` — accuracy, size, latency
- `scripts/run_comparison.py` — end-to-end sweep → `results/comparison.csv`
- `scripts/save_sample_grid.py` — labeled CIFAR-10 class grid → `results/cifar10_samples.png`
- `configs/default.yaml` — CIFAR-10 default
- `configs/fashion_mnist.yaml` — smaller CPU alternative

## Notes

- Unstructured pruning zeros weights but **does not shrink serialized size** unless you use a sparse format or structured prune. The `sparsity` and `nonzero_params` columns are the honest signal.
- Dynamic quant mainly shrinks **Linear** layers; conv compute stays fp32.
- Distillation quality depends on a decent teacher. Bump `epochs_teacher` before judging KD.
