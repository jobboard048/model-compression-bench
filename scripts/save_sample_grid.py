from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from compression_bench.data import CLASS_NAMES, dataset_class_names

CELL_SCALE = 4
LABEL_HEIGHT = 22
PAD = 8
COLS = 5
ROWS = 2


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    t = img.clamp(0, 1)
    if t.ndim == 3 and t.shape[0] == 1:
        arr = (t * 255).to(torch.uint8).squeeze(0).numpy()
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (t * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def first_image_per_class(dataset, n_classes: int) -> dict[int, Image.Image]:
    found: dict[int, Image.Image] = {}
    for img, label in dataset:
        if label not in found:
            found[label] = _tensor_to_pil(img)
        if len(found) == n_classes:
            return found

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    for idx in perm:
        img, label = dataset[idx]
        if label not in found:
            found[label] = _tensor_to_pil(img)
        if len(found) == n_classes:
            break
    return found


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def make_labeled_grid(samples: dict[int, Image.Image], class_names: list[str]) -> Image.Image:
    missing = [name for i, name in enumerate(class_names) if i not in samples]
    if missing:
        raise RuntimeError(f"Missing classes: {', '.join(missing)}")

    font = load_font()
    sample = samples[0]
    cell_w = sample.width * CELL_SCALE
    cell_h = sample.height * CELL_SCALE + LABEL_HEIGHT
    canvas_w = COLS * cell_w + (COLS + 1) * PAD
    canvas_h = ROWS * cell_h + (ROWS + 1) * PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for class_id, name in enumerate(class_names):
        row, col = divmod(class_id, COLS)
        x = PAD + col * (cell_w + PAD)
        y = PAD + row * (cell_h + PAD)
        enlarged = samples[class_id].resize(
            (sample.width * CELL_SCALE, sample.height * CELL_SCALE),
            Image.NEAREST,
        )
        canvas.paste(enlarged, (x, y))
        bbox = draw.textbbox((0, 0), name, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + max(0, (cell_w - text_w) // 2)
        text_y = y + sample.height * CELL_SCALE + 4
        draw.text((text_x, text_y), name, fill=(20, 20, 20), font=font)
    return canvas


def save_sample_grid(dataset: str, data_dir: Path, out_path: Path) -> Path:
    names = dataset_class_names(dataset)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    if dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=tfm)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    samples = first_image_per_class(ds, len(names))
    grid = make_labeled_grid(samples, names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a labeled 2x5 class sample grid")
    parser.add_argument("--dataset", choices=sorted(CLASS_NAMES), default="fashion_mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out if args.out is not None else Path("results") / args.dataset / "samples.png"
    path = save_sample_grid(args.dataset, args.data_dir, out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
