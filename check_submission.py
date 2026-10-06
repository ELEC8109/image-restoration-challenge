"""Official checker for the final ONNX submission."""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_submission(model):
    start = time.perf_counter()
    try:
        from onnx_io import check, score_pair

        spec = json.loads((ROOT / "challenge.json").read_text(encoding="utf-8"))
        _, session = check(model, spec)
        data = ROOT / "data/val"
        images = sorted((data / "input").glob("*.png"))
        if not images:
            raise ValueError(f"Validation images not found: {data / 'input'}")
        print("Validation sample PSNR:", flush=True)
        scores = []
        for index in sorted({0, (len(images)-1)//2, len(images)-1}):
            path = images[index]
            raw, restored = score_pair(session, path, data / "target" / path.name)
            scores.append(restored)
            print(f"  {path.name}: input {raw:.3f} -> restored {restored:.3f} dB", flush=True)
        print(f"  Sample mean: {sum(scores)/len(scores):.3f} dB", flush=True)
    except Exception as error:
        print("SUBMISSION CHECK FAILED", file=sys.stderr)
        print(f"Reason: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"SUBMISSION CHECK PASSED ({time.perf_counter()-start:.2f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to the submitted model.onnx")
    args = parser.parse_args()
    print(f"Checking {args.model}...", flush=True)
    # Keep Ctrl+C responsive even if native ONNX inference stalls.
    worker = multiprocessing.get_context("spawn").Process(target=check_submission, args=(args.model,))
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.5)
    except KeyboardInterrupt:
        worker.terminate()
        worker.join()
        return 130
    return worker.exitcode


if __name__ == "__main__":
    raise SystemExit(main())
