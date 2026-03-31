# -*- coding: ascii -*-
import os, sys, io, json, time, zipfile, subprocess, warnings, functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.checkpoint import checkpoint as grad_ckpt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")


# =============================================================================
# CONFIG  -- edit paths here if needed
# =============================================================================
class Cfg:
    BASE        = "/home/user1/Pneumonia_ML_Project/NTIRE"
    TRAIN_BLEND = "/home/user1/Pneumonia_ML_Project/NTIRE/trian_5k/trian_5k/blended/"
    TRAIN_TRANS = "/home/user1/Pneumonia_ML_Project/NTIRE/trian_5k/trian_5k/transmission_layer/"
    VAL_BLEND   = "/home/user1/Pneumonia_ML_Project/NTIRE/val_300_blended/val_300/blended/"
    OUTPUT_DIR  = BASE + "/submission"
    CKPT_DIR    = BASE + "/checkpoints"
    WORKER      = BASE + "/sirr_worker.py"   # <-- the second file you downloaded
    S1_CKPT     = BASE + "/best_Stage1_Coarse.pth"

    DIM    = 64
    HEADS  = [1, 2, 4, 8]
    EXPAND = 2.66
    BLOCKS = [4, 6, 6, 8]

    # Stage 2: 256px, 3 GPUs x bs=6 = eff_bs 18 (safe for 48GB)
    S2_PATCH, S2_BS, S2_EP, S2_LR = 256,  6, 60, 5e-5
    # Stage 3: 384px, 3 GPUs x bs=3 = eff_bs 9  (safe for 48GB)
    S3_PATCH, S3_BS, S3_EP, S3_LR = 384,  3, 30, 1e-5

    W_L1, W_PERC, W_SSIM, W_FREQ, W_EDGE = 1.0, 0.1, 0.3, 0.05, 0.05

    WORKERS   = 6
    PREFETCH  = 3
    SEED      = 42
    GRAD_CLIP = 0.01


cfg = Cfg()
os.makedirs(cfg.CKPT_DIR,   exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Sanity checks
if not os.path.exists(cfg.S1_CKPT):
    raise FileNotFoundError(
        "Stage 1 checkpoint not found: %s\n"
        "Files in %s:\n%s" % (
            cfg.S1_CKPT, cfg.BASE,
            "\n".join(str(p) for p in Path(cfg.BASE).glob("*.pth"))))

if not os.path.exists(cfg.WORKER):
    raise FileNotFoundError(
        "Worker script not found: %s\n"
        "Make sure sirr_worker.py is in %s" % (cfg.WORKER, cfg.BASE))

print("[OK] Stage 1 checkpoint : %s  (%.0f MB)" % (
    cfg.S1_CKPT, os.path.getsize(cfg.S1_CKPT) / 1e6))
print("[OK] Worker script      : %s" % cfg.WORKER)

n_gpus = torch.cuda.device_count()
print("")
print("-" * 55)
print("  GPUs available: %d" % n_gpus)
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print("  GPU %d: %s  %.0f GB" % (i, p.name, p.total_memory / 1e9))
print("-" * 55)
print("")
assert n_gpus >= 1, "No GPUs found!"


# =============================================================================
# STAGE LAUNCHER
# =============================================================================
_PORT = [29500]

def launch_stage(stage, patch, bs, epochs, lr, prev_ckpt, loss_w=None):
    ckpt_path = os.path.join(cfg.CKPT_DIR, "best_%s.pth" % stage)

    cfg_dict = dict(
        stage     = stage,
        blend_dir = cfg.TRAIN_BLEND,
        trans_dir = cfg.TRAIN_TRANS,
        patch     = patch,
        bs        = bs,
        epochs    = epochs,
        lr        = float(lr),
        prev_ckpt = str(prev_ckpt),
        ckpt_path = ckpt_path,
        dim       = cfg.DIM,
        blocks    = cfg.BLOCKS,
        heads     = cfg.HEADS,
        expand    = cfg.EXPAND,
        workers   = cfg.WORKERS,
        prefetch  = cfg.PREFETCH,
        seed      = cfg.SEED,
        grad_clip = cfg.GRAD_CLIP,
        loss_w    = loss_w or dict(
            w_l1=cfg.W_L1, w_p=cfg.W_PERC,
            w_s=cfg.W_SSIM, w_f=cfg.W_FREQ, w_e=cfg.W_EDGE),
    )

    cfg_path = os.path.join(cfg.CKPT_DIR, "cfg_%s.json" % stage)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    port = _PORT[0]
    _PORT[0] += 1

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=%d" % n_gpus,
        "--master_port=%d"    % port,
        "--standalone",
        cfg.WORKER,
        "--cfg", cfg_path,
    ]

    print("")
    print("-" * 60)
    print("  Launching: %s  (%d GPUs)" % (stage, n_gpus))
    print("  CMD: %s" % " ".join(cmd))
    print("-" * 60)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"]        = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="")
    proc.stdout.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            "torchrun failed (exit %d) for [%s]\n"
            "See traceback above." % (proc.returncode, stage))

    print("")
    print("  [DONE] %s  ->  %s" % (stage, ckpt_path))
    return ckpt_path


# =============================================================================
# INFERENCE MODEL  (identical arch to worker, defined here for inference only)
# =============================================================================
class LN2d_I(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d)
    def forward(self, x):
        B, C, H, W = x.shape
        return self.n(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

class SG_I(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b

class MDTA_I(nn.Module):
    def __init__(self, d, h, bias=False):
        super().__init__()
        self.h    = h
        self.temp = nn.Parameter(torch.ones(h, 1, 1))
        self.qkv  = nn.Conv2d(d, d * 3, 1, bias=bias)
        self.dw   = nn.Conv2d(d * 3, d * 3, 3, 1, 1, groups=d * 3, bias=bias)
        self.proj = nn.Conv2d(d, d, 1, bias=bias)
    def forward(self, x):
        B, C, H, W = x.shape
        D = C // self.h
        q, k, v = self.dw(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(B, self.h, D, H * W)
        k = k.reshape(B, self.h, D, H * W)
        v = v.reshape(B, self.h, D, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp
        return self.proj((attn.softmax(-1) @ v).reshape(B, C, H, W))

class GDFN_I(nn.Module):
    def __init__(self, d, ex=2.66, bias=False):
        super().__init__()
        h = int(d * ex)
        self.p1 = nn.Conv2d(d, h * 2, 1, bias=bias)
        self.dw = nn.Conv2d(h * 2, h * 2, 3, 1, 1, groups=h * 2, bias=bias)
        self.sg = SG_I()
        self.p2 = nn.Conv2d(h, d, 1, bias=bias)
    def forward(self, x):
        return self.p2(self.sg(self.dw(self.p1(x))))

class SE_I(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        m = max(d // r, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(d, m, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(m, d, 1, bias=False))
    def forward(self, x):
        return x * (self.fc(F.adaptive_avg_pool2d(x, 1)) +
                    self.fc(F.adaptive_max_pool2d(x, 1))).sigmoid()

class TBlock_I(nn.Module):
    def __init__(self, d, h, ex=2.66):
        super().__init__()
        self.n1  = LN2d_I(d)
        self.n2  = LN2d_I(d)
        self.att = MDTA_I(d, h)
        self.ffn = GDFN_I(d, ex)
    def forward(self, x):
        dt = x.dtype
        x = x + self.att(self.n1(x).to(dt))
        x = x + self.ffn(self.n2(x).to(dt))
        return x

def _stage_I(d, h, n, ex):
    return nn.Sequential(*[TBlock_I(d, h, ex) for _ in range(n)], SE_I(d))

class Down_I(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(d, d // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.b(x)

class Up_I(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(d, d * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))
    def forward(self, x):
        return self.b(x)

class SIRR_Net_I(nn.Module):
    def __init__(self, dim=64, blocks=[4,6,6,8], heads=[1,2,4,8], expand=2.66):
        super().__init__()
        D = dim
        self.inp  = nn.Conv2d(3, D, 3, 1, 1, bias=False)
        self.e1   = _stage_I(D,     heads[0], blocks[0], expand)
        self.d1   = Down_I(D)
        self.e2   = _stage_I(D * 2, heads[1], blocks[1], expand)
        self.d2   = Down_I(D * 2)
        self.e3   = _stage_I(D * 4, heads[2], blocks[2], expand)
        self.d3   = Down_I(D * 4)
        self.bot  = _stage_I(D * 8, heads[3], blocks[3], expand)
        self.u3   = Up_I(D * 8)
        self.f3   = nn.Conv2d(D * 8, D * 4, 1, bias=False)
        self.dec3 = _stage_I(D * 4, heads[2], blocks[2], expand)
        self.u2   = Up_I(D * 4)
        self.f2   = nn.Conv2d(D * 4, D * 2, 1, bias=False)
        self.dec2 = _stage_I(D * 2, heads[1], blocks[1], expand)
        self.u1   = Up_I(D * 2)
        self.f1   = nn.Conv2d(D * 2, D, 1, bias=False)
        self.dec1 = _stage_I(D,     heads[0], blocks[0], expand)
        self.out  = nn.Conv2d(D, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        f  = self.inp(x)
        e1 = self.e1(f)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        b  = self.bot(self.d3(e3))
        d3 = self.dec3(self.f3(torch.cat([self.u3(b),  e3], 1)))
        d2 = self.dec2(self.f2(torch.cat([self.u2(d3), e2], 1)))
        d1 = self.dec1(self.f1(torch.cat([self.u1(d2), e1], 1)))
        return (self.out(d1) + x).clamp(0, 1)


# =============================================================================
# INFERENCE + 8-FOLD TTA
# =============================================================================
@torch.no_grad()
def tta_infer(model, tensor, device):
    x = tensor.unsqueeze(0).to(device)

    def aug(t, k):
        if k == 0: return t
        if k == 1: return t.flip(3)
        if k == 2: return t.flip(2)
        if k == 3: return t.flip([2, 3])
        if k == 4: return t.transpose(2, 3).contiguous()
        if k == 5: return t.transpose(2, 3).flip(3).contiguous()
        if k == 6: return t.transpose(2, 3).flip(2).contiguous()
        if k == 7: return t.transpose(2, 3).flip([2, 3]).contiguous()

    def deaug(t, k):
        if k == 0: return t
        if k == 1: return t.flip(3)
        if k == 2: return t.flip(2)
        if k == 3: return t.flip([2, 3])
        if k == 4: return t.transpose(2, 3).contiguous()
        if k == 5: return t.flip(3).transpose(2, 3).contiguous()
        if k == 6: return t.flip(2).transpose(2, 3).contiguous()
        if k == 7: return t.flip([2, 3]).transpose(2, 3).contiguous()

    preds = []
    with torch.amp.autocast("cuda"):
        for k in range(8):
            preds.append(deaug(model(aug(x, k)), k).cpu())
    return torch.stack(preds).mean(0).squeeze(0).clamp(0, 1)


def run_inference(ckpt_path, input_dir, output_dir, use_tta=True):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0")
    model  = SIRR_Net_I(
        dim=cfg.DIM, blocks=cfg.BLOCKS,
        heads=cfg.HEADS, expand=cfg.EXPAND).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print("")
    print("  Model loaded: %s" % ckpt_path)

    paths = sorted(Path(input_dir).glob("*.[jp][pn][ge]*"))
    print("  Inference: %d images   TTA=%s" % (len(paths), use_tta))
    t0 = time.time()

    for i, p in enumerate(paths, 1):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        pw = (8 - W % 8) % 8
        ph = (8 - H % 8) % 8
        pad = TF.pad(img, (0, 0, pw, ph), padding_mode="reflect")
        t   = T.ToTensor()(pad)
        if use_tta:
            out = tta_infer(model, t, device)
        else:
            out = model(t.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
        out = out[:, :H, :W]
        T.ToPILImage()(out).save(
            os.path.join(output_dir, p.stem + ".png"),
            format="PNG", compress_level=0)
        if i % 50 == 0 or i == len(paths):
            print("  [%3d/%d]  %.0fs" % (i, len(paths), time.time() - t0))

    print("  [OK] %d images saved -> %s" % (len(paths), output_dir))


# =============================================================================
# SUBMISSION ZIP
# =============================================================================
def make_zip():
    zp = os.path.join(cfg.BASE, "submission.zip")
    readme = (
        "runtime per image [s] : 1.8\n"
        "CPU[1] / GPU[0] : 0\n"
        "Extra Data [1] / No Extra Data [0] : 0\n"
        "Other description : Restormer U-Net (MDTA + SimpleGate GDFN + SE). "
        "3x RTX 6000 Ada DDP torchrun. "
        "Loss=L1+VGG19+MS-SSIM+FFT+Edge. "
        "Progressive: 128px(80ep) -> 256px(60ep) -> 384px(30ep). "
        "8-fold TTA. AMP fp16. PyTorch 2.x.\n"
    )
    with open(os.path.join(cfg.OUTPUT_DIR, "readme.txt"), "w") as f:
        f.write(readme)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in Path(cfg.OUTPUT_DIR).iterdir():
            z.write(fp, fp.name)
    print("  [OK] submission.zip  %.0f MB  ->  %s" % (os.path.getsize(zp) / 1e6, zp))
    return zp


# =============================================================================
# GPU STATUS
# =============================================================================
def gpu_status():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        print("")
        print("-- GPU Status --------------------------------------------------")
        for line in r.stdout.strip().split("\n"):
            parts = [s.strip() for s in line.split(",")]
            if len(parts) == 5:
                idx, name, util, mu, mt = parts
                used  = int(mu) / 1024
                total = int(mt) / 1024
                pct   = int(util)
                bar   = "#" * (pct // 10) + "." * (10 - pct // 10)
                print("  GPU%s  %-35s  [%s] %3d%%  VRAM %5.1f/%.0f GB" % (
                    idx, name, bar, pct, used, total))
        print("----------------------------------------------------------------")
        print("")
    except Exception as e:
        print("  nvidia-smi: %s" % e)


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    gpu_status()
    print("")
    print("-" * 60)
    print("  NTIRE 2026 SIRR  |  %dx RTX 6000 Ada  |  Resume Stage 2" % n_gpus)
    print("-" * 60)
    print("")

    # Stage 2: load Stage 1, train at 256px
    ckpt_s2 = launch_stage(
        stage     = "Stage2_Mid",
        patch     = cfg.S2_PATCH,
        bs        = cfg.S2_BS,
        epochs    = cfg.S2_EP,
        lr        = cfg.S2_LR,
        prev_ckpt = cfg.S1_CKPT,
    )

    # Stage 3: boost perceptual + edge for Phase 2 judges
    ckpt_s3 = launch_stage(
        stage     = "Stage3_Perceptual",
        patch     = cfg.S3_PATCH,
        bs        = cfg.S3_BS,
        epochs    = cfg.S3_EP,
        lr        = cfg.S3_LR,
        prev_ckpt = ckpt_s2,
        loss_w    = dict(w_l1=0.7, w_p=0.25, w_s=0.3, w_f=0.05, w_e=0.1),
    )

    # Inference + TTA
    run_inference(
        ckpt_path  = ckpt_s3,
        input_dir  = cfg.VAL_BLEND,
        output_dir = cfg.OUTPUT_DIR,
        use_tta    = True,
    )

    make_zip()
    gpu_status()
    print("[DONE] Submit: %s/submission.zip" % cfg.BASE)