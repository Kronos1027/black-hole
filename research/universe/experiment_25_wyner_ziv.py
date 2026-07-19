#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 25: Wyner-Ziv — Incremental Compression with Pre-trained Backbone

Based on: Wyner-Ziv rate-distortion with side information (1976)
Claude priority: YELLOW

HYPOTHESIS: Compressing a NEW image is cheaper when a backbone is already
trained on similar images. The backbone is "side information" — the decoder
already has it, so we only need to transmit the small per-image modulation.

METHOD:
  1. Train shared backbone on N=50 images (one-time cost)
  2. For each NEW image: freeze backbone, fit ONLY a small head (65 params)
  3. Compare to COIN (train full SIREN from scratch for each new image)

EXPECTED:
  - Wyner-Ziv per-image cost: ~75 bytes (just the head)
  - COIN per-image cost: ~856 bytes (full SIREN)
  - Ratio: ~11x smaller per new image
  - Quality: lower (backbone not adapted to new image) but testable

Author: Darlan Pereira da Silva (Kronos1027)
"""
import torch; torch.set_num_threads(8)
import numpy as np, time, json, zlib
from skimage.data import (astronaut, camera, cell, coins, moon,
                           page, text, clock, coffee, chelsea)
import torch.nn as nn

# Load images
sources = [astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea]
imgs = []
for src_fn in sources:
    arr = src_fn()
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 4: arr = arr[:,:,:3]
    H, W = arr.shape[:2]
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    count = 0
    for cy in range(0, H-64+1, 64):
        for cx in range(0, W-64+1, 64):
            if count >= 10: break
            imgs.append(gray[cy:cy+64, cx:cx+64].astype(np.uint8))
            count += 1
            if len(imgs) >= 60: break
        if count >= 10: break
    if len(imgs) >= 60: break

n_train = 50; n_new = 10
coords = np.stack(np.meshgrid(np.linspace(-1,1,64),np.linspace(-1,1,64)),axis=-1).reshape(-1,2)
xt = torch.tensor(coords, dtype=torch.float32)

# ========== Step 1: Train backbone on 50 images ==========
print(f"=== Step 1: Train backbone on {n_train} images ===")
train_imgs = imgs[:n_train]
targets_train = [torch.tensor((img.astype(np.float32)/255.0).flatten(), dtype=torch.float32) for img in train_imgs]

torch.manual_seed(0)
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(); d = 2
        for k in range(2):
            lin = nn.Linear(d, 64)
            bound = 1.0/d if k == 0 else np.sqrt(6.0/64)/15.0
            lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
            self.layers.append(lin); d = 64
        self.omega = 15.0
    def forward(self, x):
        h = x
        for layer in self.layers: h = torch.sin(self.omega*layer(h))
        return h

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_train)])
        for h in self.heads:
            bound = np.sqrt(6.0/64)/15.0
            h.weight.data.uniform_(-bound,bound); h.bias.data.uniform_(-bound,bound)
    def forward(self, x, idx):
        return self.heads[idx](self.backbone(x))

model = FullModel()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(10):
    opt.zero_grad(); loss = 0
    for i,yt in enumerate(targets_train):
        pred = model(xt,i).squeeze(-1); loss = loss + ((pred-yt)**2).mean()
    loss = loss/n_train; loss.backward(); opt.step()
t_train = time.time()-t0
print(f"  Backbone trained: {t_train:.1f}s")

# Backbone size (one-time cost, amortized)
bb_params = np.concatenate([p.detach().numpy().flatten() for p in model.backbone.parameters()])
max_val = np.abs(bb_params).max(); scale = max_val/127.5
q = np.round(bb_params/scale).astype(np.int8)
bb_compressed = zlib.compress(q.tobytes(), 9)
bb_size = len(bb_compressed) + 10
print(f"  Backbone size: {bb_size}B (amortized over all images)")

# ========== Step 2: Wyner-Ziv — fit ONLY head for new images ==========
print(f"\n=== Step 2: Wyner-Ziv — fit head for {n_new} new images (backbone frozen) ===")
new_imgs = imgs[n_train:n_train+n_new]
targets_new = [torch.tensor((img.astype(np.float32)/255.0).flatten(), dtype=torch.float32) for img in new_imgs]

# Freeze backbone
for p in model.backbone.parameters():
    p.requires_grad = False

wz_psnrs = []; wz_total_bytes = 0
for i, img in enumerate(new_imgs):
    # Fit ONLY a new head (65 params)
    head = nn.Linear(64, 1)
    bound = np.sqrt(6.0/64)/15.0
    head.weight.data.uniform_(-bound,bound); head.bias.data.uniform_(-bound,bound)
    opt2 = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(50):
        opt2.zero_grad()
        with torch.no_grad():
            h = model.backbone(xt)
        pred = head(h).squeeze(-1)
        loss = ((pred - targets_new[i])**2).mean()
        loss.backward(); opt2.step()

    # Evaluate
    model.backbone.eval(); head.eval()
    with torch.no_grad():
        h = model.backbone(xt)
        recon = head(h).squeeze(-1).numpy()
    r = (recon*255).clip(0,255).astype(np.uint8).reshape(64,64)
    err = img.astype(float)-r.astype(float); mse = np.mean(err**2)
    psnr = 100.0 if mse==0 else 10*np.log10(255**2/mse)
    wz_psnrs.append(psnr)

    # Size: just the head
    head_params = np.concatenate([p.detach().numpy().flatten() for p in head.parameters()])
    max_val = np.abs(head_params).max(); scale = max_val/127.5
    q = np.round(head_params/scale).astype(np.int8)
    compressed = zlib.compress(q.tobytes(), 9)
    wz_total_bytes += len(compressed) + 10

wz_psnr = np.mean(wz_psnrs)
print(f"  Wyner-Ziv: {wz_psnr:.2f}dB, {wz_total_bytes}B for {n_new} images ({wz_total_bytes/n_new:.0f}B/img)")

# ========== Step 3: COIN baseline (from scratch) ==========
print(f"\n=== Step 3: COIN baseline (train from scratch, 50 epochs) ===")
coin_psnrs = []; coin_total_bytes = 0
for i, img in enumerate(new_imgs):
    torch.manual_seed(0)
    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(); d = 2
            for k in range(2):
                lin = nn.Linear(d, 32)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/32)/15.0
                lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
                self.layers.append(lin); d = 32
            self.head = nn.Linear(32, 1)
            bound = np.sqrt(6.0/32)/15.0
            self.head.weight.data.uniform_(-bound,bound); self.head.bias.data.uniform_(-bound,bound)
            self.omega = 15.0
        def forward(self, x):
            h = x
            for layer in self.layers: h = torch.sin(self.omega*layer(h))
            return self.head(h)
    m = Siren(); o = torch.optim.Adam(m.parameters(), lr=1e-3)
    yt = torch.tensor((img.astype(np.float32)/255.0).flatten(), dtype=torch.float32)
    for ep in range(50):
        o.zero_grad(); pred = m(xt).squeeze(-1)
        loss = ((pred-yt)**2).mean(); loss.backward(); o.step()
    m.eval()
    with torch.no_grad(): recon = m(xt).squeeze(-1).numpy()
    r = (recon*255).clip(0,255).astype(np.uint8).reshape(64,64)
    err = img.astype(float)-r.astype(float); mse = np.mean(err**2)
    coin_psnrs.append(100.0 if mse==0 else 10*np.log10(255**2/mse))
    params = np.concatenate([p.detach().numpy().flatten() for p in m.parameters()])
    max_val = np.abs(params).max(); scale = max_val/127.5
    q = np.round(params/scale).astype(np.int8)
    compressed = zlib.compress(q.tobytes(), 9)
    coin_total_bytes += len(compressed) + 10

coin_psnr = np.mean(coin_psnrs)
print(f"  COIN: {coin_psnr:.2f}dB, {coin_total_bytes}B for {n_new} images ({coin_total_bytes/n_new:.0f}B/img)")

# ========== COMPARISON ==========
print(f"\n=== COMPARISON ({n_new} new images, backbone pre-trained on {n_train}) ===")
ratio = coin_total_bytes / wz_total_bytes
print(f"  COIN:        {coin_psnr:.2f}dB, {coin_total_bytes}B ({coin_total_bytes/n_new:.0f}B/img)")
print(f"  Wyner-Ziv:   {wz_psnr:.2f}dB, {wz_total_bytes}B ({wz_total_bytes/n_new:.0f}B/img)")
print(f"  Ratio:       {ratio:.2f}x (WZ smaller per new image)")
print(f"  PSNR diff:   {wz_psnr-coin_psnr:+.2f}dB")

# Including backbone amortized over different N_new
print(f"\n  Amortized analysis (backbone={bb_size}B):")
for n_amort in [10, 50, 100, 1000]:
    wz_amort = (bb_size + wz_total_bytes * n_amort / n_new) / n_amort
    ratio_amort = coin_total_bytes / n_new / wz_amort
    print(f"    N_new={n_amort:>4}: WZ={wz_amort:.0f}B/img, COIN={coin_total_bytes/n_new:.0f}B/img, ratio={ratio_amort:.2f}x")

print('\n--- JSON ---')
print(json.dumps({'experiment':25,'approach':'wyner_ziv',
    'n_train':n_train,'n_new':n_new,
    'backbone_size':int(bb_size),
    'coin':{'psnr':float(coin_psnr),'bytes':int(coin_total_bytes),'per_img':float(coin_total_bytes/n_new)},
    'wyner_ziv':{'psnr':float(wz_psnr),'bytes':int(wz_total_bytes),'per_img':float(wz_total_bytes/n_new)},
    'ratio':float(ratio),'psnr_diff':float(wz_psnr-coin_psnr)}, indent=2))
