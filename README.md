# Image Restoration Challenge

Restore clean RGB images from blurred, noisy inputs. You have **1,200 training pairs and 120 validation pairs**, all 384×384, with matching degraded `input/*.png` and clean `target/*.png`. At inference, the model receives only the degraded image, without blur or noise parameters.

- **Train from scratch** using the supplied training pairs. No pretrained weights or external images. Keep the supplied train/val split; use validation for model selection, not training.
- **Capacity: at most 300,000 stored model tensor elements.**
- **Submit `model.onnx` and a short `report.pdf`.** Final ranking uses mean per-image full-image RGB PSNR on hidden scenes, without border cropping.

## Run the starter

```bash
pip install -r requirements.txt
python baseline.py train
python baseline.py export
python check_submission.py submission/model.onnx
```

Training saves `best.pt`, `loss.png` and comparison images with zooms in `runs/baseline/`. Export writes `submission/model.onnx`.

Training uses **128×128 crops to save memory and computation**; validation uses full images after each epoch. In **`loss.png`**, the samples and measurement times differ, so train loss may stay above val loss. Keep the default, or increase `patch_size` in `config.json` if resources allow (`384` for whole images).

Reference PSNR on all 120 validation images: **29.79 dB input → 31.74 dB restored** (Windows, RTX 5080, seed 42). The 40-epoch run took **about 1–2 minutes**, including validation. Runtime and retraining results can vary across hardware and software versions, even with the same seed.

To evaluate all 120 validation images: `python baseline.py evaluate submission/model.onnx`.

## Design your model

The optional starter uses a three-layer CNN (1,027 PyTorch parameters), L1, batch 4 and 40 epochs.

You may change the architecture, loss, optimizer, augmentation, patch size and training schedule.

| Files | Purpose and editing rules |
| --- | --- |
| `model.py`, `config.json` | Model and training settings; edit freely. |
| `baseline.py`, `utils.py` | Optional training, export and evaluation code; edit or replace as needed. |
| `challenge.json`, `onnx_io.py`, `check_submission.py` | Official submission requirements and checker; keep unchanged. |

Check your model before training:

```bash
python baseline.py train --check-only
```

This exports the untrained model and checks capacity, interface and CPU inference. Normal `train` also checks first.

To reuse the exporter, implement `build_model(config)` and keep the starter's checkpoint format. Custom export is allowed. For separate experiments, pass `--run runs/my_experiment` to both `train` and `export`.

## ONNX submission requirements

`challenge.json` and the official checker define the fixed requirements:

- One self-contained ONNX file, **opset 18**, without external tensor data.
- One float32 `input` and one float32 `output`, both **[1, 3, 384, 384]**, RGB in **[0, 1]**. Output must be finite and CPU inference deterministic.
- Capacity counts stored tensors and all `Constant` values, including subgraphs/functions and sparse values/indices. It can differ from PyTorch's parameter count.

**The submitted model must pass `check_submission.py`; the teaching team uses the same checker.** It also runs three validation images and prints their PSNR, followed by `SUBMISSION CHECK PASSED` or a failure reason.

## Report and submission

In **1–2 pages**, show your network and training settings, checker element count, train/val curves using the same loss, validation input/output PSNR and two examples with zooms.

Clean images: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), Agustsson & Timofte, NTIRE 2017. Academic research use; original owners retain copyright. Degraded pairs are generated for this course.
