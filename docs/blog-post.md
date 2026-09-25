# Compressing a small CNN: distillation, pruning, and quantization on a laptop

I trained one small convolutional network on Fashion-MNIST, then compressed it three ways and measured accuracy, file size, and speed on the same 10,000-image test set. One seed, one laptop CPU, no claim that this ranks compression methods in general.

The code is this repository. From the repo root:

```text
python scripts/setup.py
python scripts/test.py
python scripts/run.py
```

`run.py` trains the teacher until validation accuracy plateaus, then trains a smaller student, a distilled student, a pruned teacher, and two INT8 variants. Curves, a class sample grid, per-class accuracy, and checkpoints land in `results/fashion_mnist/`.

Hardware for the numbers below: Windows laptop, Intel UHD Graphics 620, no CUDA, PyTorch CPU, seed 42.

---

## Why compress a model this small?

The teacher is already under a megabyte. Compression here is not about saving a data centre. It is four questions on one task:

- Can a smaller network match the big one?
- Do the teacher's soft labels help that small network more than ordinary labels?
- If you zero the weakest weights and keep them zero, does accuracy hold?
- Does INT8 shrink the file without a large accuracy drop?

Those questions only mean something when the dataset, the architecture family, and the metrics are shared.

---

## The dataset

Fashion-MNIST (Zalando): 70,000 grayscale 28×28 images of clothing, 10 classes, 60,000 train and 10,000 test. torchvision downloads it.

The classes are T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

`run.py` writes `samples.png`, a labeled 2×5 grid of one training example per class.

CIFAR-10 is the optional harder task (`python scripts/run.py --dataset cifar10`). The numbers in this post are Fashion-MNIST only.

---

## The models

Teacher and student are the same CNN with different width.

- Teacher: `width=32`, 220,234 parameters, 0.84 MB.
- Student: `width=8`, 52,138 parameters, 0.20 MB.

Training is SGD (learning rate 0.01, momentum 0.9, weight decay 1e-4) on CPU. Each teacher epoch trains on 90% of the training set and scores the held-out 10%. The official test set is not used to pick the epoch.

The cap was 30 epochs. Validation accuracy stopped improving by at least 0.2 points for four epochs, so training stopped at epoch 18 and the comparison used the best checkpoint, epoch 14 (validation accuracy 92.1%). Students then trained for 14 epochs.

---

## Three compression methods

### Distillation

The student matches the teacher's softened class probabilities (KL divergence, temperature 4) and the true labels (cross-entropy). The soft term has weight 0.7.

A control student trains on hard labels only, same size, same epoch count. If distillation helps, that student should win.

### Pruning

Global unstructured L1 pruning zeros the smallest-magnitude convolution and linear weights. The mask stays on during a one-epoch fine-tune at 0.1× the learning rate, then the zeros are baked into the checkpoint. The configured amount is 50%.

A curve from the same teacher checkpoint also fine-tunes copies at 0%, 30%, 50%, and 80% sparsity. The 0% point is the extra epoch with no zeros, so a gain from "just train a bit more" is visible on its own.

Unstructured zeros do not shrink a dense `.pt` file and do not speed up this CPU. The honest signals are sparsity and nonzero parameter count.

### Quantization

- Dynamic INT8 quantizes Linear layers at runtime. Convolutions stay fp32.
- Static INT8 calibrates on 20 training batches and converts the graph, so convolutions can quantize too. This PyTorch build has no `x86` engine, so the run used `onednn`.

Eager-mode post-training quantization is deprecated in current PyTorch. It still runs. The replacement is TorchAO, which would be a rewrite, not a flag.

---

## Results

Official Fashion-MNIST test set, batch 128, CPU. Latency is the median of 50 timed batches after 10 warmup batches.

| Variant | Test accuracy | Nonzero params | Sparsity | Size (MB) | Latency p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher fp32 | 92.07% | 220,234 | 0% | 0.84 | 146 |
| student, hard labels | 89.97% | 52,138 | 0% | 0.20 | 30 |
| student, distilled | 90.90% | 52,138 | 0% | 0.20 | 80 |
| teacher, 50% pruned | 92.46% | 110,202 | 50.0% | 0.84 | 189 |
| teacher, dynamic INT8 | 91.98% | — | — | 0.27 | 135 |
| teacher, static INT8 | 91.97% | — | — | 0.22 | 35 |

Dynamic INT8 parameter counts are packed quantized tensors, and the static checkpoint does not report a comparable parameter count, so those cells are left blank. Size and accuracy are the comparable columns.

### Distillation

The small student is about a quarter of the teacher in parameters and file size, and 2.1 points less accurate (89.97% vs 92.07%). Soft labels recover 0.93 points of that gap (90.90%) at the same size.

The two students are the same architecture, so the latency gap (30 ms vs 80 ms) is not a distillation effect. I would not quote it as one. The clean speed result is the hard-label student at 30 ms against the teacher's 146 ms.

### Pruning

Same teacher checkpoint, one fine-tune epoch each:

| Target sparsity | Measured sparsity | Test accuracy |
| --- | ---: | ---: |
| 0% (extra epoch only) | 0% | 92.33% |
| 30% | 30.0% | 92.41% |
| 50% | 50.0% | 92.46% |
| 80% | 79.9% | 92.10% |

Most of the lift over the original teacher (92.07% → 92.33%) is the extra epoch. Holding half the weights at zero does not hurt: 92.46% is 0.13 points above that no-prune control, inside the noise of a 10,000-image test set. At 80% sparsity, accuracy falls back to the dense baseline (92.10%) and sits 0.23 points under the fine-tune control.

The 50% checkpoint is still 0.84 MB, and the batch is slower (189 ms vs 146 ms). Zeroing weights counted. It did not compress the file or the runtime.

### Quantization

Static INT8 is the method that changes size and speed together: 91.97% accuracy, 0.22 MB, 35 ms. That is about 0.1 points under the fp32 teacher, a quarter of the file, and roughly four times faster on this CPU.

Dynamic INT8 lands in the same accuracy band (91.98%) and shrinks the file to 0.27 MB, with almost no speedup (135 ms), because the convolutions stay fp32.

oneDNN warned that its default config can be less accurate on a CPU without VNNI. The measured drop was still about 0.1 points.

### Per class

Overall accuracy hides the shirt class. Every variant is weakest there (about 75–78%). Trousers, sandals, bags, and ankle boots stay above 95%.

Under 50% pruning, the coat drops from 92.5% to 88.4%, while the pullover rises from 85.5% to 89.0%. Static INT8 shows the same coat dip (87.3%). A 0.1-point overall change can still move one clothing class by several points.

---

## What I would take from this run

On this model and this dataset:

- A 4× smaller student keeps most of the accuracy. Distillation adds about a point over training that student on hard labels.
- Unstructured pruning to 50% keeps accuracy if the zeros are not allowed to grow back. It does not make the checkpoint smaller or the CPU faster. 80% is where accuracy starts to give the extra training back.
- Static INT8 is the compression that shows up in both the file size and the latency, with a negligible accuracy change on this test set.

These are one seed on Fashion-MNIST. A 0.1–0.4 point gap is a few dozen images out of 10,000. I would rerun with more seeds before treating the prune curve as a ranking.

---

## Reproduce

```text
python scripts/setup.py
python scripts/test.py
python scripts/run.py
```

`setup.py` creates `.venv` and installs CPU PyTorch when there is no NVIDIA GPU. `test.py` runs the unit tests with no dataset download. On this laptop the full run took a few hours; the teacher stopped at epoch 14 of a 30-epoch cap.

To redo only the prune curve from the saved teacher, without retraining students or INT8:

```text
python scripts/run.py --prune-only
```

That writes `prune_rerun.log` and `prune_sparsity_curve.csv` and refreshes the pruned row. It does not overwrite `run.log`.

---

## Footnotes

- Fashion-MNIST is from Zalando Research. CIFAR-10 is from the Canadian Institute for Advanced Research.
- Unstructured prune is not a smaller file unless you store a sparse format or prune whole channels.
- `torch.ao.quantization` is on the way out in PyTorch. These numbers are from the eager API on a CPU wheel, engine `onednn`.
- Dynamic INT8 `params` and static INT8 `params` in `comparison.csv` are not comparable to the fp32 counts. Use `size_mb`.
