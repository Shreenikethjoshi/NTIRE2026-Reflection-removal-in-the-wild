# -*- coding: ascii -*-
import os, sys, io, math, random, time, warnings, argparse, json, functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.checkpoint import checkpoint as grad_ckpt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")


# =============================================================================
# DATASET
# =============================================================================
class SIRRDataset(Dataset):
    def __init__(self, blend_dir, trans_dir=None, patch=256, aug=True):
        self.bp    = sorted(Path(blend_dir).glob("*.[jp][pn][ge]*"))
        self.tp    = sorted(Path(trans_dir).glob("*.[jp][pn][ge]*")) if trans_dir else None
        self.patch = patch
        self.aug   = aug

    def __len__(self):
        return len(self.bp)

    def __getitem__(self, idx):
        b = Image.open(self.bp[idx]).convert("RGB")
        t = Image.open(self.tp[idx]).convert("RGB") if self.tp else None
        fname = self.bp[idx].name
        if self.aug and t is not None:
            b, t = self._augment(b, t)
        bt = T.ToTensor()(b)
        tt = T.ToTensor()(t) if t else torch.zeros_like(bt)
        return bt, tt, fname

    def _pad(self, img):
        W, H = img.size
        if W < self.patch or H < self.patch:
            s = max(self.patch / W, self.patch / H) + 0.01
            img = img.resize((int(W * s), int(H * s)), Image.BICUBIC)
        return img

    def _augment(self, b, t):
        b, t = self._pad(b), self._pad(t)
        i, j, h, w = T.RandomCrop.get_params(b, (self.patch, self.patch))
        b, t = TF.crop(b, i, j, h, w), TF.crop(t, i, j, h, w)
        if random.random() > 0.5: b, t = TF.hflip(b), TF.hflip(t)
        if random.random() > 0.5: b, t = TF.vflip(b), TF.vflip(t)
        k = random.randint(0, 3)
        if k: b, t = TF.rotate(b, 90 * k), TF.rotate(t, 90 * k)
        if random.random() > 0.3:
            b = T.ColorJitter(0.2, 0.2, 0.15, 0.05)(b)
            t = T.ColorJitter(0.05, 0.05)(t)
        if random.random() > 0.6:
            buf = io.BytesIO()
            b.save(buf, "JPEG", quality=random.randint(65, 95))
            b = Image.open(buf).convert("RGB")
        if random.random() > 0.7:
            bt2 = T.ToTensor()(b)
            bt2 = (bt2 + torch.randn_like(bt2) * random.uniform(0.005, 0.02)).clamp(0, 1)
            b = T.ToPILImage()(bt2)
        return b, t


# =============================================================================
# MODEL
# =============================================================================
class LN2d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d)
    def forward(self, x):
        B, C, H, W = x.shape
        return self.n(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class MDTA(nn.Module):
    # Channel-transposed attention: attention matrix is (B,h,D,D) not (B,h,HW,HW)
    # Memory is O(C^2), never OOM regardless of patch size
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
        attn = (q @ k.transpose(-2, -1)) * self.temp  # (B, h, D, D)
        return self.proj((attn.softmax(-1) @ v).reshape(B, C, H, W))


class GDFN(nn.Module):
    def __init__(self, d, ex=2.66, bias=False):
        super().__init__()
        h = int(d * ex)
        self.p1 = nn.Conv2d(d, h * 2, 1, bias=bias)
        self.dw = nn.Conv2d(h * 2, h * 2, 3, 1, 1, groups=h * 2, bias=bias)
        self.sg = SimpleGate()
        self.p2 = nn.Conv2d(h, d, 1, bias=bias)

    def forward(self, x):
        return self.p2(self.sg(self.dw(self.p1(x))))


class SE(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        m = max(d // r, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(d, m, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(m, d, 1, bias=False))

    def forward(self, x):
        return x * (self.fc(F.adaptive_avg_pool2d(x, 1)) +
                    self.fc(F.adaptive_max_pool2d(x, 1))).sigmoid()


class TBlock(nn.Module):
    def __init__(self, d, h, ex=2.66):
        super().__init__()
        self.n1  = LN2d(d)
        self.n2  = LN2d(d)
        self.att = MDTA(d, h)
        self.ffn = GDFN(d, ex)

    def forward(self, x):
        # FIX: LayerNorm always outputs fp32 under AMP.
        # Cast back to input dtype so Conv2d sees matching dtype.
        dt = x.dtype
        x = x + self.att(self.n1(x).to(dt))
        x = x + self.ffn(self.n2(x).to(dt))
        return x


def make_stage(d, h, n, ex):
    return nn.Sequential(*[TBlock(d, h, ex) for _ in range(n)], SE(d))


class Down(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(d, d // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.b(x)


class Up(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(d, d * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.b(x)


class SIRR_Net(nn.Module):
    def __init__(self, dim=64, blocks=[4,6,6,8], heads=[1,2,4,8],
                 expand=2.66, use_ckpt=True):
        super().__init__()
        D = dim
        self.uc   = use_ckpt
        self.inp  = nn.Conv2d(3, D, 3, 1, 1, bias=False)
        self.e1   = make_stage(D,     heads[0], blocks[0], expand)
        self.d1   = Down(D)
        self.e2   = make_stage(D * 2, heads[1], blocks[1], expand)
        self.d2   = Down(D * 2)
        self.e3   = make_stage(D * 4, heads[2], blocks[2], expand)
        self.d3   = Down(D * 4)
        self.bot  = make_stage(D * 8, heads[3], blocks[3], expand)
        self.u3   = Up(D * 8)
        self.f3   = nn.Conv2d(D * 8, D * 4, 1, bias=False)
        self.dec3 = make_stage(D * 4, heads[2], blocks[2], expand)
        self.u2   = Up(D * 4)
        self.f2   = nn.Conv2d(D * 4, D * 2, 1, bias=False)
        self.dec2 = make_stage(D * 2, heads[1], blocks[1], expand)
        self.u1   = Up(D * 2)
        self.f1   = nn.Conv2d(D * 2, D, 1, bias=False)
        self.dec1 = make_stage(D,     heads[0], blocks[0], expand)
        self.out  = nn.Conv2d(D, 3, 3, 1, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _bot(self, x):
        return self.bot(x)

    def forward(self, x):
        f  = self.inp(x)
        e1 = self.e1(f)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        bi = self.d3(e3)
        if self.uc and self.training:
            b = grad_ckpt(self._bot, bi, use_reentrant=False)
        else:
            b = self.bot(bi)
        d3 = self.dec3(self.f3(torch.cat([self.u3(b),  e3], 1)))
        d2 = self.dec2(self.f2(torch.cat([self.u2(d3), e2], 1)))
        d1 = self.dec1(self.f1(torch.cat([self.u1(d2), e1], 1)))
        return (self.out(d1) + x).clamp(0, 1)


# =============================================================================
# LOSSES: L1 + VGG Perceptual + MS-SSIM + FFT + Sobel Edge
# =============================================================================
class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        v = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.slices = nn.ModuleList([v[:4], v[4:9], v[9:18]])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, p, g):
        loss = 0.0
        for s in self.slices:
            p = s(p)
            g = s(g)
            loss = loss + F.l1_loss(p, g)
        return loss


def ssim_loss(p, g, k=11):
    C1, C2 = 1e-4, 9e-4
    mp = F.avg_pool2d(p, k, 1, k // 2)
    mg = F.avg_pool2d(g, k, 1, k // 2)
    mp2, mg2, mpg = mp ** 2, mg ** 2, mp * mg
    sp2 = F.avg_pool2d(p ** 2, k, 1, k // 2) - mp2
    sg2 = F.avg_pool2d(g ** 2, k, 1, k // 2) - mg2
    spg = F.avg_pool2d(p * g,  k, 1, k // 2) - mpg
    return 1 - ((2 * mpg + C1) * (2 * spg + C2) /
                ((mp2 + mg2 + C1) * (sp2 + sg2 + C2))).mean()


def ms_ssim_loss(p, g):
    loss = 0.0
    for _ in range(3):
        loss = loss + ssim_loss(p, g)
        p = F.avg_pool2d(p, 2)
        g = F.avg_pool2d(g, 2)
    return loss / 3.0


def freq_loss(p, g):
    return F.l1_loss(
        torch.fft.rfft2(p, norm="ortho").abs(),
        torch.fft.rfft2(g, norm="ortho").abs())


def edge_loss(p, g):
    def sobel(x):
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        x  = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(x, kx, padding=1)
        ey = F.conv2d(x, ky, padding=1)
        return (ex ** 2 + ey ** 2 + 1e-6).sqrt()
    return F.l1_loss(sobel(p), sobel(g))


class SIRRLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_p=0.1, w_s=0.3, w_f=0.05, w_e=0.05):
        super().__init__()
        self.perc = Perceptual()
        self.w    = (w_l1, w_p, w_s, w_f, w_e)

    def forward(self, pred, gt):
        l1   = F.l1_loss(pred, gt)
        perc = self.perc(pred, gt)
        ssim = ms_ssim_loss(pred, gt)
        freq = freq_loss(pred, gt)
        edge = edge_loss(pred, gt)
        tot  = (self.w[0] * l1   + self.w[1] * perc +
                self.w[2] * ssim + self.w[3] * freq  + self.w[4] * edge)
        with torch.no_grad():
            mse  = F.mse_loss(pred.float(), gt.float())
            psnr = 10.0 * math.log10(1.0 / (mse.item() + 1e-8))
        return tot, dict(
            l1=l1.item(), perc=perc.item(),
            ssim=round(1.0 - ssim.item(), 4),
            freq=freq.item(), edge=edge.item(), psnr=psnr)


# =============================================================================
# MAIN  (one process per GPU, launched by torchrun)
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    C = json.load(open(parser.parse_args().cfg))

    dist.init_process_group("nccl")
    rank    = dist.get_rank()
    ws      = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev     = torch.device("cuda:%d" % rank)
    is_main = (rank == 0)

    s = C["seed"] + rank
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

    # Model
    model = SIRR_Net(
        dim=C["dim"], blocks=C["blocks"],
        heads=C["heads"], expand=C["expand"],
        use_ckpt=True).to(dev)

    if C.get("prev_ckpt") and os.path.exists(C["prev_ckpt"]):
        sd = torch.load(C["prev_ckpt"], map_location=dev, weights_only=True)
        # Strip DDP prefix
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # Remap old short attr names to new full names so Stage1 ckpt loads cleanly.
        # Old TBlock: self.a (MDTA), self.f (GDFN)
        #   a.t   -> att.temp,  a.p   -> att.proj
        #   a.qkv -> att.qkv,   a.dw  -> att.dw
        #   f.*   -> ffn.*
        def remap_key(k):
            parts = k.split(".")
            out = []
            i = 0
            while i < len(parts):
                p = parts[i]
                if p == "a" and i + 1 < len(parts):
                    nxt = parts[i + 1]
                    out.append("att")
                    out.append("temp" if nxt == "t" else ("proj" if nxt == "p" else nxt))
                    i += 2
                elif p == "f" and i + 1 < len(parts):
                    out.append("ffn")
                    out.append(parts[i + 1])
                    i += 2
                else:
                    out.append(p)
                    i += 1
            return ".".join(out)
        sd_new = {remap_key(k): v for k, v in sd.items()}
        if sd_new != sd and is_main:
            print("  [INFO] Remapped old-format checkpoint keys to new names")
        sd = sd_new
        model.load_state_dict(sd, strict=True)
        if is_main:
            print("  [OK] Loaded: %s" % C["prev_ckpt"])
    elif is_main:
        print("  [WARN] No prev_ckpt - training from scratch")

    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False)

    # Data
    ds   = SIRRDataset(C["blend_dir"], C["trans_dir"],
                       patch=C["patch"], aug=True)
    samp = DistributedSampler(ds, ws, rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        ds, batch_size=C["bs"], sampler=samp,
        num_workers=C["workers"], pin_memory=True,
        prefetch_factor=C["prefetch"],
        persistent_workers=True, drop_last=True)

    # Optimizer
    eff   = C["bs"] * ws
    lr    = C["lr"] * math.sqrt(eff / 8.0)
    opt   = AdamW(model.parameters(), lr=lr,
                  weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    sched = CosineAnnealingWarmRestarts(
        opt, T_0=max(C["epochs"] // 3, 1), T_mult=1, eta_min=lr / 200)
    scaler    = torch.amp.GradScaler("cuda")
    crit      = SIRRLoss(**C["loss_w"]).to(dev)
    best      = float("inf")
    best_psnr = 0.0
    ckpt_path = C["ckpt_path"]
    t_total   = time.time()

    if is_main:
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        print("=" * 72)
        print("  STAGE      : %s" % C["stage"])
        print("  GPUs       : %d    Params : %.1fM" % (ws, n_p))
        print("  Patch      : %dpx  Per-GPU bs : %d   Eff bs : %d" % (C["patch"], C["bs"], eff))
        print("  Base LR    : %.2e  Scaled LR : %.2e" % (C["lr"], lr))
        print("  Epochs     : %d" % C["epochs"])
        print("  Loss       : L1 + Perceptual + MS-SSIM + FFT + Edge")
        print("  Checkpoint : %s" % ckpt_path)
        print("=" * 72)
        print("")
        print("  %5s/%-5s  %8s  %7s  %7s  %6s  %7s  %7s  %7s  %8s  %9s  %5s  %5s  %5s  %5s  Note" % (
            "Ep", "Tot", "Loss", "L1", "Perc", "SSIM", "FFT", "Edge", "PSNR", "Best", "LR", "G0", "G1", "G2", "Sec"))
        print("  " + "-" * 136)

    for epoch in range(1, C["epochs"] + 1):
        samp.set_epoch(epoch)
        model.train()
        ep = dict(loss=0, l1=0, perc=0, ssim=0, freq=0, edge=0, psnr=0)
        nb = 0
        t0 = time.time()

        for blend, trans, _ in loader:
            blend = blend.to(dev, non_blocking=True)
            trans = trans.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred = model(blend)
                loss, comp = crit(pred, trans)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), C["grad_clip"])
            scaler.step(opt)
            scaler.update()
            ep["loss"] += loss.item()
            for k in ("l1", "perc", "ssim", "freq", "edge", "psnr"):
                ep[k] += comp[k]
            nb += 1

        sched.step()
        nb = max(nb, 1)

        keys = list(ep.keys())
        t_s  = torch.tensor([ep[k] / nb for k in keys], device=dev)
        dist.all_reduce(t_s, op=dist.ReduceOp.AVG)
        avg  = dict(zip(keys, t_s.tolist()))

        if is_main:
            elapsed = time.time() - t0
            is_best = avg["loss"] < best
            note    = ""
            if is_best:
                best      = avg["loss"]
                best_psnr = avg["psnr"]
                torch.save(model.module.state_dict(), ckpt_path)
                mb   = os.path.getsize(ckpt_path) / 1e6
                note = "[BEST %.0fMB]" % mb
            cur_lr = sched.get_last_lr()[0]
            m0 = torch.cuda.memory_allocated(0) / 1e9
            m1 = torch.cuda.memory_allocated(1) / 1e9 if ws > 1 else 0.0
            m2 = torch.cuda.memory_allocated(2) / 1e9 if ws > 2 else 0.0
            print("  %5d/%-5d  %8.4f  %7.4f  %7.4f  %6.4f  %7.4f  %7.4f  %6.2fdB  %8.4f  %9.2e  %4.1fG  %4.1fG  %4.1fG  %4.0fs  %s" % (
                epoch, C["epochs"],
                avg["loss"], avg["l1"], avg["perc"], avg["ssim"],
                avg["freq"], avg["edge"], avg["psnr"],
                best, cur_lr, m0, m1, m2, elapsed, note))

    if is_main:
        total = (time.time() - t_total) / 60.0
        print("=" * 72)
        print("  [DONE] %s" % C["stage"])
        print("  Best loss  : %.6f" % best)
        print("  Best PSNR  : %.2f dB" % best_psnr)
        print("  Checkpoint : %s  (%.1f MB)" % (ckpt_path, os.path.getsize(ckpt_path) / 1e6))
        print("  Total time : %.1f min" % total)
        print("=" * 72)
        print("")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()