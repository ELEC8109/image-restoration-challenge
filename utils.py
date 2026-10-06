"""Paired images, RGB metrics, and figures for the reference baseline."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def pairs(root):
    root = Path(root)
    inputs = sorted((root / "input").glob("*.png"))
    targets = sorted((root / "target").glob("*.png"))
    if not inputs or [p.name for p in inputs] != [p.name for p in targets]:
        raise ValueError(f"Expected matching PNGs in {root}/input and target")
    return list(zip(inputs, targets))


class PairedImages(Dataset):
    def __init__(self, root, patch=0, augment=False, cache=False):
        self.paths = pairs(root)
        self.patch, self.augment = patch, augment
        self.images = [(load_image(a), load_image(b)) for a, b in self.paths] if cache else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x, y = self.images[index] if self.images is not None else tuple(map(load_image, self.paths[index]))
        if x.shape != y.shape:
            raise ValueError(f"Pair has different dimensions: {self.paths[index]}")
        if self.patch:
            h, w = x.shape[:2]
            if self.patch > min(h, w):
                raise ValueError("patch_size must not exceed the image dimensions")
            top = int(torch.randint(h - self.patch + 1, ()))
            left = int(torch.randint(w - self.patch + 1, ()))
            x = x[top:top+self.patch, left:left+self.patch]
            y = y[top:top+self.patch, left:left+self.patch]
        if self.augment:
            turns = int(torch.randint(4, ()))
            x, y = np.rot90(x, turns), np.rot90(y, turns)
            if bool(torch.randint(2, ())):
                x, y = np.fliplr(x), np.fliplr(y)
        return tuple(torch.from_numpy(a.transpose(2, 0, 1).copy()).float() / 255 for a in (x, y))


def psnr(prediction, target):
    """Per-image RGB PSNR on [0,1], no border crop; identical images capped at 120 dB."""
    # Compute this small validation reduction on CPU; MPS has no float64 tensors.
    error = (prediction.detach().cpu().double().clamp(0, 1) - target.detach().cpu().double()).square().flatten(1).mean(1)
    return -10 * error.clamp_min(1e-12).log10()


def curve(history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Path(history).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(r["train_loss"]) for r in rows], label="Train L1 (random crops)", color="#2563eb")
    ax.plot(epochs, [float(r["val_loss"]) for r in rows], label="Val L1 (full images)", color="#ea580c")
    ax.set(xlabel="Epoch", ylabel="Mean absolute error (RGB in [0, 1])", title="Training and validation loss")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def comparison(x, prediction, target, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arrays = [v.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) for v in (x, prediction, target)]
    h, w = arrays[-1].shape[:2]
    zoom = min(96, h, w)
    gray = arrays[-1].mean(2)
    gy, gx = np.gradient(gray)
    detail = np.abs(gx) + np.abs(gy)
    candidates = [(float(detail[y:y+zoom, z:z+zoom].mean()), y, z)
                  for y in range(0, h-zoom+1, max(1, zoom//2))
                  for z in range(0, w-zoom+1, max(1, zoom//2))]
    _, top, left = max(candidates)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for i, (array, label) in enumerate(zip(arrays, ("Input", "Restored", "Target"))):
        axes[0, i].imshow(array)
        axes[0, i].add_patch(plt.Rectangle((left, top), zoom, zoom, fill=False, edgecolor="#ef4444", lw=1))
        axes[0, i].set_title(label)
        axes[1, i].imshow(array[top:top+zoom, left:left+zoom], interpolation="nearest")
        axes[1, i].set_title(f"{zoom} x {zoom} detail")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
