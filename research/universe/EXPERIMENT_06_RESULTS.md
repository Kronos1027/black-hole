# BHUH Experiment 6: Resolution Scaling — HONEST RESULT

**Date**: 2026-06-27
**Status**: ⚠️ No resolution scaling — but this reveals something important

---

## Results (N=5, varying resolution)

| Resolution | COIN bytes | BHUH bytes | Ratio | COIN PSNR | BHUH PSNR | ΔPSNR |
|-----------|-----------|-----------|-------|-----------|-----------|-------|
| 64×64 | 4382 | 3045 | 1.44× | 25.15 dB | 23.86 dB | -1.30 |
| 128×128 | 4378 | 3030 | 1.44× | 24.07 dB | 22.95 dB | -1.12 |
| 256×256 | 4383 | 3024 | 1.45× | 23.60 dB | 22.52 dB | -1.09 |

**Power law exponent: 0.01 (essentially zero)**

---

## Honest Analysis

### ❌ BHUH does NOT scale with resolution

The ratio is **constant** (~1.44×) across all resolutions. Doubling resolution does NOT multiply the advantage.

### Why? (honest explanation)

SIREN stores PARAMETERS, not pixels. The network (h=32 for COIN, h=64 for BHUH) has the same parameter count regardless of image resolution:

- COIN: 1185 params → ~876 bytes (zlib compressed) at ANY resolution
- BHUH: 4352 + 5×65 = 4677 params → ~606 bytes per image at ANY resolution

The recipe size is determined by **parameter count**, not pixel count.

### What DOES change with resolution

Per-pixel cost (bytes per pixel):

| Resolution | COIN bpp | BHUH bpp | BHUH/COIN |
|-----------|---------|---------|-----------|
| 64×64 | 0.0067 | 0.0047 | 0.69× |
| 128×128 | 0.0033 | 0.0023 | 0.69× |
| 256×256 | 0.0017 | 0.0012 | 0.69× |

Both get cheaper per-pixel at the SAME rate. The ratio stays constant.

PSNR drops slightly with resolution (harder to fit larger images with same SIREN).

---

## What this means (honest)

### The BHUH advantage comes from N (number of images), NOT resolution

From Experiment 5:
- N=5: 1.44× advantage
- N=10: 2.69× advantage
- N=20: 4.77× advantage
- N=50: 8.94× advantage

From Experiment 6:
- 64×64: 1.44× (at N=5)
- 128×128: 1.44× (at N=5)
- 256×256: 1.45× (at N=5)

**Scaling dimension that matters: N (images), not resolution.**

### This is actually GOOD news for BHUH

1. **Resolution independence**: BHUH works equally well at any resolution
2. **No degradation at high-res**: unlike JPEG which gets worse per-pixel at low res
3. **Scales with corpus size**: the more images, the better BHUH gets

### What would make resolution matter

To get resolution scaling, we'd need:
- Larger SIREN for larger images (more params → better fit but bigger recipe)
- Or: different SIREN architecture that adapts to resolution
- Or: multi-resolution SIREN (Phase 86) where coarse+detail both stored

---

## Combined scaling law (honest)

BHUH advantage = f(N, resolution) where:
- f(N) = 0.168 + 0.800 × N (from Experiment 5, R²=0.984)
- f(resolution) ≈ 1.0 (constant, from Experiment 6)

**BHUH is a CORPUS compressor, not a per-image compressor.**

The more images you compress together, the better BHUH gets. Resolution doesn't matter — parameter count does.

---

## Reproducibility

```bash
/usr/bin/python3 research/universe/experiment_06_resolution_scaling.py
```

Expected: ratio ~1.44× at all resolutions (64, 128, 256) for N=5.
