# Technical Architecture — TransReflect

> Detailed breakdown of every component in `sirr_worker.py` and `train.py`

---

## 1. Model: `SIRR_Net`

A symmetric encoder-decoder U-Net with 4 resolution levels, built entirely from Transformer Blocks.

### 1.1 Overall Structure

| Module | Channels | Blocks | Heads |
|--------|----------|--------|-------|
| Input conv | 3 → 64 | — | — |
| Encoder E1 | 64 | 4 | 1 |
| Down 1 | 64 → 128 | PixelUnshuffle(2) | — |
| Encoder E2 | 128 | 6 | 2 |
| Down 2 | 128 → 256 | PixelUnshuffle(2) | — |
| Encoder E3 | 256 | 6 | 4 |
| Down 3 | 256 → 512 | PixelUnshuffle(2) | — |
| Bottleneck | 512 | 8 | 8 |
| Up 3 | 512 → 256 | PixelShuffle(2) | — |
| Fuse 3 (cat+1×1) | 512 → 256 | — | — |
| Decoder D3 | 256 | 6 | 4 |
| Up 2 | 256 → 128 | PixelShuffle(2) | — |
| Fuse 2 (cat+1×1) | 256 → 128 | — | — |
| Decoder D2 | 128 | 6 | 2 |
| Up 1 | 128 → 64 | PixelShuffle(2) | — |
| Fuse 1 (cat+1×1) | 128 → 64 | — | — |
| Decoder D1 | 64 | 4 | 1 |
| Output conv + residual | 64 → 3 | — | — |

**Total parameters:** ~8.2 M

---

### 1.2 Downsampling (`Down`)
```
Conv2d(d, d//2, 3×3) → PixelUnshuffle(2)
```
Halves spatial resolution (H,W → H/2, W/2), doubles channels (d → d).  
Space-to-depth rearrangement — no information loss.

### 1.3 Upsampling (`Up`)
```
Conv2d(d, d*2, 3×3) → PixelShuffle(2)
```
Doubles spatial resolution, halves channels. Inverse of Down.

### 1.4 Skip Connections
```
cat([Up(deeper), encoder_feat], dim=1) → Conv2d(2d, d, 1×1)
```
Standard U-Net skip connection with 1×1 fusion conv to halve channels before the decoder stage.

---

### 1.5 Transformer Block (`TBlock`)

```
x → LN2d → MDTA → + x  →  LN2d → GDFN → + x
```

Residual connections around both attention and FFN sub-layers.  
`LN2d` = LayerNorm operating on the channel dimension of (B, C, H, W) feature maps.

> **AMP note:** LayerNorm always outputs fp32. The block explicitly casts back to input dtype (`x.dtype`) before convolutions to avoid dtype mismatch under mixed-precision training.

---

### 1.6 MDTA — Multi-Dconv Head Transposed Attention

**Key idea:** Compute attention in *channel space* (C×C) rather than spatial space (HW×HW).  
Complexity: **O(C²)** vs O((HW)²) — enables high-resolution training without OOM.

```python
qkv  = DW-Conv(QKV-Conv(x))          # (B, 3C, H, W)
q, k, v = split into (B, h, D, HW)   # D = C//h
q, k = L2-normalize(q), L2-normalize(k)
attn = softmax(q @ k.T * temp)        # (B, h, D, D)
out  = Proj-Conv(attn @ v)
```

- Learnable temperature parameter `temp` per head
- Depthwise conv before QKV projection for local context

---

### 1.7 GDFN — Gated Depthwise Feed-Forward Network

```python
x → Conv1×1(d → 2*expand*d)
  → DW-Conv3×3
  → SimpleGate (split → element-wise product)
  → Conv1×1(expand*d → d)
```

**SimpleGate:** splits tensor along channel dim into (a, b) → returns `a * b`  
Expand ratio = **2.66×** (from Restormer).

---

### 1.8 SE — Squeeze-and-Excite

Channel recalibration applied once after each encoder/decoder stage.

```python
scale = sigmoid(fc(avg_pool(x)) + fc(max_pool(x)))
out   = x * scale
```

Uses both average and max pooling for richer global context. Reduction ratio = 4.

---

### 1.9 Global Residual + Output

```python
out = clamp(Conv3×3(dec1_features) + input_image, 0, 1)
```

Network learns the *residual* (what to remove / restore) rather than the full image.

---

## 2. Loss Function: `SIRRLoss`

### 2.1 L1 Loss
Standard pixel-wise absolute error. Weight = 1.0 (Stage 2), 0.7 (Stage 3).

### 2.2 VGG-19 Perceptual Loss
Features extracted from 3 slices of frozen VGG-19:
- `relu1_2` (layers 0–3)
- `relu2_2` (layers 4–8)
- `relu3_4` (layers 9–17)

L1 distance in feature space. Weight = 0.1 (Stage 2), 0.25 (Stage 3).

### 2.3 MS-SSIM Loss
3-scale SSIM averaged over pyramid (original + 2× downsampled).  
`loss = 1 - SSIM(pred, gt)` at each scale. Weight = 0.3.

### 2.4 FFT Magnitude Loss
```python
F.l1_loss(rfft2(pred).abs(), rfft2(gt).abs())
```
Penalizes differences in global frequency distribution. Captures low-frequency color shifts and high-frequency texture differences. Weight = 0.05.

### 2.5 Sobel Edge Loss
```python
edge = sqrt(conv(x, Kx)^2 + conv(x, Ky)^2 + eps)
F.l1_loss(edge(pred), edge(gt))
```
Horizontal + vertical Sobel kernels applied to grayscale image. Encourages sharp, well-preserved edges. Weight = 0.05 (Stage 2), 0.1 (Stage 3).

---

## 3. Training Pipeline (`train.py`)

### 3.1 Stage Launcher
Each stage is launched as a **separate `torchrun` subprocess**:
```bash
python -m torch.distributed.run \
    --nproc_per_node=<N_GPUS> \
    --master_port=<PORT> \
    --standalone \
    sirr_worker.py --cfg cfg_<stage>.json
```
Config is serialized to a JSON file and passed to the worker.

### 3.2 Optimizer & Scheduler
- **AdamW**: lr (scaled), weight_decay=1e-4, β=(0.9, 0.999)
- **LR scaling**: `lr_eff = base_lr × sqrt(eff_batch / 8)` — linear scaling rule adapted for square root
- **Scheduler**: CosineAnnealingWarmRestarts, T_0 = epochs//3

### 3.3 AMP + Gradient Clipping
- `torch.amp.GradScaler("cuda")` for loss scaling
- `clip_grad_norm_(params, 0.01)` — aggressive clipping for training stability

### 3.4 Checkpoint Key Remapping
The worker includes a `remap_key()` function to translate old checkpoint attribute names (`a.t`, `a.p`, `f.*`) to new names (`att.temp`, `att.proj`, `ffn.*`), ensuring backward compatibility across checkpoint versions.

---

## 4. Inference (`train.py → run_inference`)

### 4.1 Padding Strategy
Images padded to multiples of 8 (required by 3× PixelUnshuffle):
```python
pw = (8 - W % 8) % 8
ph = (8 - H % 8) % 8
padded = TF.pad(img, (0, 0, pw, ph), padding_mode="reflect")
```
Padding is cropped from the output before saving.

### 4.2 8-Fold TTA
All 8 D4 symmetry group transforms (identity + 3 flips + 4 rotations via transpose):

| k | Transform |
|---|-----------|
| 0 | Identity |
| 1 | Horizontal flip |
| 2 | Vertical flip |
| 3 | Both flips |
| 4 | Transpose (90°) |
| 5 | Transpose + H-flip (270°) |
| 6 | Transpose + V-flip (90° + V) |
| 7 | Transpose + both flips (180° rot) |

Each is applied, processed, de-augmented, then averaged.

### 4.3 Output
Saved as **lossless PNG** (`compress_level=0`) to preserve exact pixel values for evaluation.
