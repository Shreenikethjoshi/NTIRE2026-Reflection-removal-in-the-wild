# TransReflect: Gated Attention U-Net for Single Image Reflection Removal

> **NTIRE 2026 Challenge on Single Image Reflection Removal (SIRR) in the Wild**  
> Team **refineX** · Shreeniketh Joshi · KLE Technological University  
> CVPR 2026 Workshop

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Challenge](https://img.shields.io/badge/NTIRE-2026-red?style=flat-square)

---

## Overview

This repository contains our submission for the **NTIRE 2026 Single Image Reflection Removal (SIRR) in the Wild** challenge, hosted at CVPR 2026.

We propose **TransReflect**, a Restormer-style hierarchical U-Net trained with a multi-component loss and 8-fold test-time augmentation (TTA), designed to handle diverse, real-world reflection artifacts.

| Property | Value |
|---|---|
| Architecture | MDTA + GDFN + SE U-Net (4 levels) |
| Parameters | ~8.2 M |
| Training Strategy | 3-stage progressive (128→256→384 px) |
| Loss | L1 + VGG-19 + MS-SSIM + FFT + Sobel Edge |
| Inference TTA | 8-fold (flip + rotation) |
| Training Hardware | 3× NVIDIA RTX 6000 Ada (48 GB), DDP |

---

## Architecture

```
Input (blended)
    │
    ▼
[3×3 Conv] → 64ch
    │
    ├──[Encoder E1: 4 TBlocks + SE]──────────────────────────────┐
    │   ↓ PixelUnshuffle                                         │ skip
    ├──[Encoder E2: 6 TBlocks + SE]──────────────────────────┐  │
    │   ↓ PixelUnshuffle                                      │  │ skip
    ├──[Encoder E3: 6 TBlocks + SE]──────────────────────┐   │  │
    │   ↓ PixelUnshuffle                                  │   │  │ skip
    └──[Bottleneck: 8 TBlocks + SE @ 512ch]              │   │  │
            │                                            │   │  │
            │   PixelShuffle                             │   │  │
            ├──[Decoder D3: 6 TBlocks + SE] ←──cat──────┘   │  │
            │   PixelShuffle                                  │  │
            ├──[Decoder D2: 6 TBlocks + SE] ←──cat───────────┘  │
            │   PixelShuffle                                      │
            └──[Decoder D1: 4 TBlocks + SE] ←──cat──────────────┘
                    │
                [3×3 Conv]
                    │
                + Input (global residual)
                    │
                    ▼
            Output (transmission)
```

Each **TBlock** = LayerNorm → **MDTA** (channel-wise transposed attention) + LayerNorm → **GDFN** (gated depthwise FFN)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a full technical breakdown.

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Shreenikethjoshi/NTIRE2026-Reflection-removal-in-the-wild.git
cd NTIRE2026-Reflection-removal-in-the-wild
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your data
Organize your dataset as:
```
data/
├── train/
│   ├── blended/          # input images with reflections
│   └── transmission_layer/  # clean ground truth
└── val/
    └── blended/          # validation inputs
```

### 4. Edit config in `train.py`
Open `train.py` and update the `Cfg` class paths:
```python
class Cfg:
    TRAIN_BLEND = "/path/to/train/blended/"
    TRAIN_TRANS = "/path/to/train/transmission_layer/"
    VAL_BLEND   = "/path/to/val/blended/"
    S1_CKPT     = "/path/to/pretrained/Stage1_checkpoint.pth"
    ...
```

### 5. Train (Stage 2 → Stage 3)
```bash
python train.py
```
This automatically launches:
- **Stage 2** (256 px, 60 epochs) via `torchrun` on all available GPUs
- **Stage 3** (384 px, 30 epochs) with perceptual-boosted loss
- **Inference + 8-fold TTA** on validation set
- **Submission ZIP** generation

> Requires a pre-trained Stage 1 checkpoint. See [Training Details](#-training-details).

### 6. Inference only
To run inference on a folder of images with a trained checkpoint:
```python
from train import run_inference
run_inference(
    ckpt_path  = "path/to/checkpoint.pth",
    input_dir  = "path/to/blended/",
    output_dir = "path/to/output/",
    use_tta    = True
)
```

---

## Training Details

### Progressive Training Schedule

| Stage | Patch Size | Epochs | LR | Batch/GPU | Effective BS |
|-------|-----------|--------|----|-----------|-------------|
| 1 (coarse) | 128 px | external ckpt | — | — | — |
| 2 (mid) | 256 px | 60 | 5e-5 | 6 | 18 |
| 3 (perceptual) | 384 px | 30 | 1e-5 | 3 | 9 |

### Loss Function

$$\mathcal{L} = \lambda_1 \mathcal{L}_{L1} + \lambda_p \mathcal{L}_{perc} + \lambda_s \mathcal{L}_{MS\text{-}SSIM} + \lambda_f \mathcal{L}_{FFT} + \lambda_e \mathcal{L}_{Edge}$$

| Term | Stage 2 weight | Stage 3 weight |
|------|---------------|----------------|
| L1 | 1.0 | 0.7 |
| VGG-19 Perceptual | 0.1 | **0.25** |
| MS-SSIM | 0.3 | 0.3 |
| FFT Magnitude | 0.05 | 0.05 |
| Sobel Edge | 0.05 | **0.1** |

### Data Augmentation
- Random crop, horizontal/vertical flip, 90° rotations
- Color jitter (brightness, contrast, saturation, hue)
- JPEG compression simulation (quality 65–95)
- Gaussian noise injection (σ ∈ [0.005, 0.02])

---

## Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:

```
torch >= 2.0
torchvision
Pillow
numpy
```

---

## Project Structure

```
NTIRE2026-Reflection-removal-in-the-wild/
├── train.py             # Main orchestrator: stages, inference, zip creation
├── sirr_worker.py       # DDP worker: model definition, dataset, loss, training loop
├── requirements.txt     # Python dependencies
├── ARCHITECTURE.md      # Detailed technical description
├── factsheet/           # LaTeX factsheet for challenge report
└── README.md            # This file
```

---

## Results

Results on the NTIRE 2026 SIRR in the Wild validation set:

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| TransReflect (Ours) | — | — |

*Official leaderboard: [CodaBench](https://www.codabench.org)*

---

## Citation

If you find this work useful, please cite the NTIRE 2026 challenge report:

```bibtex
@inproceedings{ntire2026sirr,
  title     = {NTIRE 2026 Challenge on Single Image Reflection Removal},
  author    = {Cai, Jie and others},
  booktitle = {CVPR Workshops},
  year      = {2026}
}
```

---

## Acknowledgements

- Architecture inspired by [Restormer](https://github.com/swz30/Restormer) (Zamir et al., CVPR 2022)
- Training data: [OpenRR-5k](https://huggingface.co/datasets/qiuzhangTiTi/OpenRR-5k) provided by challenge organizers
- Challenge organized by Jie Cai, Kangning Yang, Radu Timofte et al.

---

## Contact

**Shreeniketh Joshi**  
KLE Technological University  
[shreenikethjoshi0605@gmail.com](mailto:shreenikethjoshi0605@gmail.com)
