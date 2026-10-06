"""Optional starter: python baseline.py train | export | evaluate."""
import argparse
import csv
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import build_model
from utils import PairedImages, comparison, curve, psnr, read_json, write_json

ROOT = Path(__file__).resolve().parent


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def load_model(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(state["config"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def export_model(model, path, spec):
    """Use the same export path before training and for the final checkpoint."""
    from onnx_io import check
    wrapper = torch.nn.Sequential(model.cpu().eval(), torch.nn.Hardtanh(0, 1)).eval()
    size = spec["image_size"]
    torch.onnx.export(wrapper, torch.zeros(1, 3, size, size), str(path), dynamo=False,
                      input_names=["input"], output_names=["output"], opset_version=spec["opset"])
    elements, session = check(path, spec)
    return wrapper, elements, session


@torch.inference_mode()
def validate(model, loader, device):
    loss, scores, raw, count = 0., [], [], 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss += float(torch.nn.functional.l1_loss(prediction, y)) * len(x)
        scores.extend(psnr(prediction, y).tolist())
        raw.extend(psnr(x, y).tolist())
        count += len(x)
    return {"loss": loss / count, "psnr": float(np.mean(scores)), "raw_psnr": float(np.mean(raw)),
            "image_psnr": scores, "raw_image_psnr": raw}


@torch.inference_mode()
def save_examples(model, dataset, device, out):
    out.mkdir(parents=True, exist_ok=True)
    # Deterministic, evenly spaced images; no selection by model performance.
    for index in sorted(set(np.linspace(0, len(dataset)-1, min(3, len(dataset)), dtype=int).tolist())):
        x, target = dataset[index]
        prediction = model(x[None].to(device))[0].clamp(0, 1).cpu()
        comparison(x, prediction, target, out / f"{dataset.paths[index][0].stem}.png")


def train(args):
    total_start = time.perf_counter()
    config = read_json(args.config)
    for key in ("epochs", "batch_size", "device", "width", "patch_size", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if min(config["epochs"], config["batch_size"], config["patch_size"]) < 1:
        raise ValueError("epochs, batch_size and patch_size must be positive")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = build_model(config)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model: {parameters:,} PyTorch parameters | checking export before training...", flush=True)
    with TemporaryDirectory(prefix="restoration-check-") as tmp:
        export_model(model, Path(tmp) / "model.onnx", read_json(ROOT / "challenge.json"))
    if args.check_only:
        print("Design check passed (untrained). Check the trained ONNX again before submission.", flush=True)
        return
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(config["device"])
    print(f"Preparing images | device: {device} | run: {out}", flush=True)
    data = Path(args.data)
    train_data = PairedImages(data / "train", config["patch_size"], True, config["cache_images"])
    val_data = PairedImages(data / "val", cache=config["cache_images"])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0)
    model = model.to(device)
    print(f"{len(train_data)} training pairs | {len(val_data)} validation pairs", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=1e-5)
    best, start = -float("inf"), time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    write_json(out / "config.json", config)
    print("Epoch | train L1 | val L1 | val PSNR | elapsed", flush=True)
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "seconds"])
        writer.writeheader()
        for epoch in range(1, config["epochs"]+1):
            model.train()
            total = 0.
            for batch, (x, y) in enumerate(train_loader, 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.l1_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach()) * len(x)
                if batch % 50 == 0:
                    print(f"  epoch {epoch}: {batch}/{len(train_loader)} batches", flush=True)
            result = validate(model, val_loader, device)
            scheduler.step()
            row = {"epoch": epoch, "train_loss": total / len(train_data), "val_loss": result["loss"],
                   "val_psnr": result["psnr"], "seconds": time.perf_counter()-start}
            writer.writerow(row)
            stream.flush()
            if result["psnr"] > best:
                best = result["psnr"]
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                            "val_psnr": best}, out / "best.pt")
            print(f"{epoch:3d}/{config['epochs']} | {row['train_loss']:.5f} | {row['val_loss']:.5f} | "
                  f"{result['psnr']:.3f} dB | {row['seconds']:.0f}s", flush=True)
    training_seconds = time.perf_counter()-start
    model, checkpoint = load_model(out / "best.pt", device)
    result = validate(model, val_loader, device)
    result.update(parameters=parameters, best_epoch=checkpoint["epoch"], training_seconds=training_seconds,
                  device=str(device), torch_version=str(torch.__version__),
                  peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None)
    curve(out / "history.csv", out / "loss.png")
    save_examples(model, val_data, device, out / "examples")
    result["total_seconds"] = time.perf_counter() - total_start
    write_json(out / "validation.json", result)
    print(f"Saved best.pt, loss.png, examples/ | raw {result['raw_psnr']:.3f} -> restored {result['psnr']:.3f} dB", flush=True)


def export(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.run) / "best.pt"
    model, _ = load_model(checkpoint)
    spec = read_json(ROOT / "challenge.json")
    path = Path(args.output or ROOT / "submission/model.onnx")
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper, stored_elements, session = export_model(model, path, spec)
    dataset = PairedImages(Path(args.data) / "val")
    errors = []
    with torch.inference_mode():
        for index in range(min(3, len(dataset))):
            x, _ = dataset[index]
            actual = session.run(None, {"input": x[None].numpy()})[0]
            expected = wrapper(x[None]).numpy()
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual-expected))))
    print(f"Exported and checked {path} | {stored_elements:,} stored model tensor elements | "
          f"max parity error {max(errors):.2g}")
    print("Submit model.onnx and your short report.pdf. Nothing else is required.")


def evaluate(args):
    from onnx_io import check, score_pair
    path = Path(args.model) if args.model else ROOT / "submission/model.onnx"
    _, session = check(path, read_json(ROOT / "challenge.json"))
    dataset = PairedImages(Path(args.data) / "val")
    scores, raw = [], []
    for i, (input_path, target_path) in enumerate(dataset.paths, 1):
        raw_score, score = score_pair(session, input_path, target_path)
        raw.append(raw_score)
        scores.append(score)
        if i % 25 == 0 or i == len(dataset):
            print(f"Evaluated {i}/{len(dataset)}", flush=True)
    print(f"Raw {np.mean(raw):.3f} -> restored {np.mean(scores):.3f} dB | {len(scores)} images")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "export", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=str(ROOT / "data"))
        if name != "evaluate":
            p.add_argument("--run", default=str(ROOT / "runs/baseline"))
        if name == "train":
            p.add_argument("--config", default=str(ROOT / "config.json"))
            p.add_argument("--check-only", action="store_true", help="check the untrained model without loading data or training")
            for option in ("epochs", "batch-size", "width", "patch-size", "seed"):
                p.add_argument("--" + option, type=int)
            p.add_argument("--device")
        elif name == "export":
            p.add_argument("--checkpoint")
            p.add_argument("--output")
        else:
            p.add_argument("model", nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        {"train": train, "export": export, "evaluate": evaluate}[args.command](args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
