from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CELL_SCALE = 4
LABEL_HEIGHT = 22
PAD = 8
COLS = 5
ROWS = 2


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def first_image_per_class(dataset) -> dict[int, Image.Image]:
    found: dict[int, Image.Image] = {}
    for img, label in dataset:
        if label not in found:
            found[label] = _tensor_to_pil(img)
        if len(found) == len(CIFAR10_CLASSES):
            return found

    remaining = [i for i in range(len(CIFAR10_CLASSES)) if i not in found]
    if remaining:
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(dataset), generator=g).tolist()
        for idx in perm:
            img, label = dataset[idx]
            if label not in found:
                found[label] = _tensor_to_pil(img)
            if len(found) == len(CIFAR10_CLASSES):
                break
    return found


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def make_labeled_grid(samples: dict[int, Image.Image]) -> Image.Image:
    missing = [name for i, name in enumerate(CIFAR10_CLASSES) if i not in samples]
    if missing:
        raise RuntimeError(f"Missing CIFAR-10 classes: {', '.join(missing)}")

    font = load_font()
    sample = samples[0]
    cell_w = sample.width * CELL_SCALE
    cell_h = sample.height * CELL_SCALE + LABEL_HEIGHT
    canvas_w = COLS * cell_w + (COLS + 1) * PAD
    canvas_h = ROWS * cell_h + (ROWS + 1) * PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for class_id, name in enumerate(CIFAR10_CLASSES):
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
        text_x = x + (cell_w - text_w) // 2
        text_y = y + sample.height * CELL_SCALE + 4
        draw.text((text_x, text_y), name, fill=(20, 20, 20), font=font)
    return canvas


def save_sample_grid(data_dir: Path, out_path: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    samples = first_image_per_class(dataset)
    grid = make_labeled_grid(samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a labeled 2x5 CIFAR-10 sample grid")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/cifar10_samples.png"))
    args = parser.parse_args()
    path = save_sample_grid(args.data_dir, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
