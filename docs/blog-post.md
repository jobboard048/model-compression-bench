# Compressing a small CNN: distillation, pruning, and quantization on a laptop

This post walks through a reproducible experiment: train a small convolutional network on a standard image-classification benchmark, then compress it three ways and compare **accuracy, size, and speed** on the same test set.

The code is this repository. From the repo root:

```text
python scripts/setup.py
python scripts/test.py
python scripts/run.py
```

`run.py` is the whole experiment. It trains **one** teacher (and stops early if validation accuracy plateaus), then trains a smaller student, a distilled student, a pruned teacher, and INT8 variants. Intermediate curves, a class sample grid, per-class accuracy, and checkpoints are written under `results/fashion_mnist/` (or `results/cifar10/`).

---

## Why compress at all?

A trained classifier is a pile of floating-point weights. For a tiny CNN that already fits in a megabyte, compression is not about saving a data centre. It is about asking, on the same task:

- Can a **smaller network** match the big one?
- Do **soft labels** from a teacher help the small network more than ordinary labels?
- Does **zeroing weak weights** (pruning) keep accuracy?
- Does **INT8** shrink the file without a large accuracy drop?

Those questions only make sense if the dataset, architecture family, and metrics are shared. That is what a benchmark is for.

---

## The dataset

The default run uses **Fashion-MNIST** (Zalando): 70,000 grayscale 28×28 images of clothing, 10 classes, 60,000 train / 10,000 test. torchvision downloads it; the images are not committed to git.

Each image is a **28 × 28 × 1** array (one brightness value per pixel). The 10 classes are T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

**CIFAR-10** is the harder optional task (`python scripts/run.py --dataset cifar10`): 60,000 colour 32×32 images, 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 50,000 train / 10,000 test, shape **32 × 32 × 3**.

Both are standard research sets. Using them means someone else can rerun the same split and compare numbers. You do not spend the article explaining how you collected photos.

`run.py` writes `samples.png`: a labeled 2×5 grid of one training example per class (unnormalized, so colours look like the photos).

---

## The models

Teacher and student are the same CNN template with different width:

- **Teacher:** wider (`width=32`), about 220k parameters on Fashion-MNIST.
- **Student:** narrower (`width=8`), about 52k parameters.

Quantization stubs wrap the network so static INT8 can insert fake-quant ops. Training is SGD with a small learning rate, on CPU unless CUDA is available (`device: auto`).

---

## How long to train

Thirty epochs is a **maximum**, not a quota.

Each teacher epoch: train on 90% of the training set, then measure accuracy on a held-out 10% (the official test set is unused for this decision). Training **stops** when validation accuracy has not improved by at least 0.2 percentage points for four epochs, or when the cap is hit. The comparison uses the **best** teacher checkpoint. Students train for that same epoch count.

That avoids the mistake of ranking compression methods on a half-trained teacher.

---

## Three compression methods (plus a smaller model)

### 1. Distillation

The teacher produces a probability distribution over 10 classes. The student is trained to match those **soft** labels (KL divergence at temperature 4) and the true **hard** labels (cross-entropy). Weight on the soft term is `alpha = 0.7`.

A control student (`student_ce`) trains on hard labels only. If distillation works, `student_kd` should beat `student_ce` at the same size.

### 2. Pruning

Global unstructured L1 pruning zeros the smallest-magnitude conv and linear weights (config amount 50%). A short fine-tune follows. **Unstructured zeros do not shrink a dense `.pt` file.** The honest signals are `sparsity` and `nonzero_params`. If fine-tuning is done without keeping masks, zeros can fill back in (that happened on an early smoke run: ~6% sparsity instead of 50%).

### 3. Quantization

- **Dynamic INT8:** Linear layers in INT8 at runtime; convolutions stay fp32. Easy, portable.
- **Static INT8:** calibrate on training batches, convert the graph. Stronger (convs can quantize too). The bench auto-selects the engine this PyTorch build actually has (`x86`, `fbgemm`, `onednn`, or `qnnpack`). On this Windows CPU wheel that is **onednn**. Eager-mode PTQ APIs are deprecated in current PyTorch (warnings, still working); the replacement is TorchAO, which is a rewrite, not a flag.

---

## What gets saved (for figures)

After `python scripts/run.py`:

| File | Use in the post |
| --- | --- |
| `samples.png` | “Here is the dataset” |
| `teacher_curve.csv` | Val accuracy vs epoch; when we stopped |
| `student_ce_curve.csv`, `student_kd_curve.csv` | Training dynamics of the two students |
| `prune_finetune_curve.csv` | Fine-tune after prune |
| `comparison.csv` | Main result table |
| `per_class_accuracy.csv` | Did compression hurt shirts more than trousers? |
| `run_summary.json` | Dataset, device, selected epochs |
| `*.pt` | Reuse weights; not required to read the article |

---

## Results: 3-epoch smoke test (not the final story)

The numbers below are from an **early Fashion-MNIST run with 3 teacher/student epochs**, on CPU, **before** early-stopping `run.py` and **before** static INT8 used `onednn`. Treat them as a pipeline check. After you run `python scripts/run.py` to plateau, **replace this table** with `comparison.csv`.

| Variant | Test accuracy | Params | Size (MB) | Latency p50 (ms / batch 128) |
| --- | ---: | ---: | ---: | ---: |
| teacher_fp32 | 87.57% | 220,234 | 0.84 | 81 |
| student_ce | 88.15% | 52,138 | 0.20 | 16 |
| student_kd | 87.76% | 52,138 | 0.20 | 14 |
| teacher_pruned | 89.62% | 220,234 | 0.84 | 76 |
| teacher_dynamic_int8 | 87.63% | 18,816* | 0.27 | 47 |

\*Dynamic INT8 `params` is not comparable to fp32 rows (packed quantized tensors).

**What that smoke test actually showed**

- The **smaller student** was faster and smaller, and slightly *more* accurate than the undertrained teacher. Capacity was not the bottleneck at 3 epochs.
- **Distillation did not beat hard labels**, which is expected when the teacher is not clearly better than the student.
- **Prune + extra fine-tune** raised accuracy but barely increased sparsity: extra training, not successful 50% compression.
- **Dynamic INT8** kept accuracy and cut checkpoint size. That is the cleanest compression signal in this table.
- Static INT8 was skipped on that run because the code demanded an `x86` engine. Current `run.py` falls back to `onednn` and a convert smoke test succeeded on this laptop.

Per-class accuracy was not recorded on that smoke run. A full `run.py` writes `per_class_accuracy.csv` so you can say which clothing classes dropped under INT8 or pruning.

---

## How to read a *real* comparison

A fair ranking needs a teacher that has **plateaued** on validation. Then:

1. If `student_kd` > `student_ce` at the same size, distillation helped.
2. If pruned sparsity is near the configured 50% **and** accuracy holds, pruning compressed the *effective* weights (still check file size).
3. If INT8 accuracy ≈ teacher and `size_mb` drops, quantization did its job.
4. Per-class: look for classes that collapse (often shirt vs T-shirt on Fashion-MNIST). Overall accuracy can hide that.

---

## Reproduce

Same three commands on Windows, macOS, or Linux. `setup.py` creates `.venv` and installs CPU PyTorch if there is no NVIDIA GPU. `test.py` runs unit tests with no dataset download. `run.py` may take on the order of **one to a few hours** on a laptop CPU (shorter if val accuracy plateaus early).

Hardware for the smoke table: Windows laptop, Intel UHD Graphics 620, no CUDA, PyTorch 2.14 CPU.

---

## Footnotes

- CIFAR-10 comes from the Canadian Institute for Advanced Research; Fashion-MNIST from Zalando Research.
- Unstructured prune ≠ smaller file unless you store sparse formats or prune structure.
- PyTorch plans to delete `torch.ao.quantization` eventually (docs mentioned 2.10, then slipped; APIs still work on 2.14). Pin the wheel you used for the blog numbers.
- Static INT8 on this CPU warned that oneDNN without VNNI can be less accurate. Report the engine name from the log (`using quantized engine 'onednn'`).
