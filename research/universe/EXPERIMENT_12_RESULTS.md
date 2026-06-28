# BHUH Experiment 12: Hierarchical K Sweep — HONEST RESULT

**Date**: 2026-06-27
**Status**: ⚠️ PARTIAL — 3.4× smaller than COIN, but 3.15 dB quality gap

---

## Results (N=100, 64×64, real crops)

| Method | Bytes | PSNR | vs COIN size | vs COIN dB |
|--------|-------|------|-------------|-----------|
| COIN (separate) | 85601 | 28.10 dB | 1.00× | 0.00 |
| Hier K=5 | 13999 | 22.43 dB | 6.11× | -5.67 |
| Hier K=10 | 16848 | 23.24 dB | 5.08× | -4.86 |
| Hier K=15 | 22407 | 23.78 dB | 3.82× | -4.32 |
| Hier K=20 | 25199 | 24.43 dB | 3.40× | -3.67 |
| **Hier K=25** | **25146** | **24.95 dB** | **3.40×** | **-3.15** |

---

## Honest Analysis

### ⚠️ Projection was OPTIMISTIC

Predicted K=20: ~27 dB at ~28KB → 3× smaller at matching quality.
Actual K=20: 24.43 dB at 25199B → 3.4× smaller but 3.67 dB worse.

### Why quality didn't reach prediction

1. **KMeans clustering imperfect** — some groups still have diverse images
2. **Fewer epochs** (60 vs 80 in Exp 10) for time budget
3. **Small groups (2-3 images)** drag down average PSNR (20-21 dB)
4. **One-hot conditioning overhead** grows with K

### Positive trend (honest)

Quality IMPROVES with K:
- K=5: 22.43 dB
- K=10: 23.24 dB (+0.81)
- K=15: 23.78 dB (+0.54)
- K=20: 24.43 dB (+0.65)
- K=25: 24.95 dB (+0.52)

Each +5 K gives ~0.5-0.8 dB improvement. At this rate:
- K=35: ~26 dB (projected)
- K=50: ~27.5 dB (projected)
- K=100 (= separate COIN): ~28 dB

### Size plateau (interesting)

Size barely changes from K=20 to K=25 (25199 → 25146):
- Smaller groups = smaller one-hot vectors
- But more backbones = more overhead
- These effects CANCEL OUT around K=20-25

This means K=25 is FREE quality improvement over K=20 — same size, +0.5 dB.

---

## Rate-Distortion Position

```
COIN:     85601B @ 28.10 dB
Hier K=25: 25146B @ 24.95 dB  ← 3.4× smaller, 3.15 dB worse
Hier K=20: 25199B @ 24.43 dB
Hier K=10: 16848B @ 23.24 dB  ← 5.1× smaller, 4.86 dB worse
Flat:      2884B @ 17.11 dB   ← 29.7× smaller, 10.99 dB worse
```

BHUH Hierarchical occupies a DIFFERENT point on the R-D curve than COIN:
- Not matching COIN quality (yet)
- But significantly smaller at acceptable quality for many applications
- 25 dB is usable for previews, thumbnails, satellite quick-look

---

## What's needed to MATCH COIN quality

1. **K=50+** — more groups, more specialization
2. **More epochs** (100+) per group
3. **Better clustering** (spectral, not just KMeans)
4. **Larger backbone** per group (h=128, but Exp 9 showed limited benefit)
5. **Quality-aware clustering** — group by SIREN-fittability, not pixel similarity

---

## Reproducibility

```bash
/usr/bin/python3 research/universe/experiment_12_k20.py
```

Expected: K=25 gives ~25 dB at ~25KB, 3.4× smaller than COIN.
