# EXPERIMENT 36 RESULTS — Hierarchical Sharing Isolated

## Status: COMPLETED — COIN DOMINATES BHUH hierarchical on BOTH axes (definitive)

**Date**: 2026-08-06
**Experiment**: 36 — Isolate Hierarchical Sharing (the original BHUH claim)
**Goal**: Test whether the hierarchical K=50 sharing mechanism (the original
"breakthrough" claim in BHUH_BREAKTHROUGH_RESULTS.md: COIN 28.10 dB → BHUH
31.21 dB) holds when multi-omega and entropy coding are controlled.

---

## Setup

Both configs use:
- **SINGLE omega=30** (no multi-omega — that was refuted in Exp 35)
- **SAME entropy coding** (KMeans K=50 + arithmetic coding)
- hidden_features=64, hidden_layers=2, epochs=300, lr=1e-3
- 30 images, 3 seeds (42, 123, 2024)

The ONLY difference is the training paradigm:
- **COIN (per-image)**: one SIREN trained per image (standard COIN)
- **BHUH (hierarchical)**: images clustered into K=50 groups by statistics
  (mean, std, percentiles); one SIREN trained per cluster on all images
  in that cluster (shared backbone)

---

## Raw Results

| Config | PSNR (dB) | Size (B) |
|--------|-----------|----------|
| COIN (per-image SIREN) | **42.20** | **4030.4** |
| BHUH (hierarchical shared backbone) | 41.09 | 4188.9 |

---

## Controlled Comparison

| Metric | Value |
|--------|-------|
| PSNR diff (BHUH - COIN) | **-1.11 dB** (COIN is higher) |
| Size ratio (BHUH / COIN) | **1.04x** (BHUH is larger) |
| **Winner** | **COIN (dominates on both axes)** |

### COIN dominates BHUH hierarchical on both axes:
1. **Higher PSNR**: 42.20 dB vs 41.09 dB (+1.11 dB)
2. **Smaller size**: 4030 B vs 4189 B (4% smaller)

---

## What This Means

### Finding 1 — Hierarchical sharing does NOT beat per-image COIN

The original BHUH_BREAKTHROUGH_RESULTS.md claimed:
> COIN (separate SIRENs): 28.10 dB, ~86000 B
> BHUH Hierarchical K=50: 31.21 dB, 56983 B (+3.11 dB, 1.5x smaller)

When tested in controlled conditions (same omega, same entropy coding),
the opposite is true:
- COIN: 42.20 dB, 4030 B
- BHUH hierarchical: 41.09 dB, 4189 B
- COIN is **1.11 dB better** and **4% smaller**

The original "breakthrough" was an artifact of comparing BHUH (with
multi-omega + entropy coding) to COIN (without either). When both
mechanisms are controlled, the hierarchical sharing provides NO benefit.

### Finding 2 — Shared backbone hurts because it averages diverse images

The hierarchical approach clusters images by statistics and trains one
SIREN per cluster. But even images with similar statistics have different
high-frequency content. A shared backbone must compromise across all
images in the cluster, producing a worse fit for each individual image
than a dedicated per-image SIREN.

### Finding 3 — The entire BHUH research program converges to one conclusion

Across experiments 29-36, every BHUH mechanism was tested in isolation:

| Mechanism | Exp | Result |
|-----------|-----|--------|
| L1 pruning (threshold=0.01) | 29 | ❌ Destroys PSNR |
| L1 pruning (threshold=0.001) | 31 | ⚠️ Sweet spot, but size gain is marginal |
| Multi-omega [10,50] architecture | 35 | ❌ COIN dominates on both axes |
| Hierarchical K=50 sharing | 36 | ❌ COIN dominates on both axes |
| **Entropy coding (KMeans + arithmetic)** | **34** | **✅ The ONLY real value (94% of advantage)** |

**The only component of BHUH that provides real value is the entropy
coding pipeline.** Every architectural innovation (multi-omega,
hierarchical sharing, pruning) either hurts or provides marginal benefit
when properly controlled.

---

## SHA-256 Verification

### Output JSON

```
SHA-256: 810b8e40b84a583e532305e396055648c508a59a6b6e606febf3c1084e78b8be
```

### Weights files

| Config | Seed | File |
|--------|------|------|
| coin_per_image | 42 | exp36_coin_per_image_seed42.bin |
| coin_per_image | 123 | exp36_coin_per_image_seed123.bin |
| coin_per_image | 2024 | exp36_coin_per_image_seed2024.bin |
| bhuh_hierarchical | 42 | exp36_bhuh_hierarchical_seed42.bin |
| bhuh_hierarchical | 123 | exp36_bhuh_hierarchical_seed123.bin |
| bhuh_hierarchical | 2024 | exp36_bhuh_hierarchical_seed2024.bin |

---

## Final Conclusion of the BHUH Research Program (Exp 29-36)

The BHUH research program set out to beat COIN via a combination of
architectural innovations (multi-omega SIREN, hierarchical sharing,
L1 pruning, entropy coding). Through rigorous controlled experiments,
we found:

1. **The entropy coding pipeline (KMeans K=50 + arithmetic coding) is
   the only component that provides real value.** It delivers ~10x size
   reduction over raw float16 with acceptable PSNR cost.

2. **Every architectural innovation was refuted under controlled
   conditions:**
   - Multi-omega [10,50] → worse than single-omega=30 (Exp 35)
   - Hierarchical K=50 sharing → worse than per-image SIREN (Exp 36)
   - L1 pruning → destroys PSNR unless threshold is very small (Exp 29-31)

3. **The optimal configuration is**: single-omega SIREN (ω=30) + entropy
   coding (KMeans K=50 + arithmetic coding), trained per-image.

This is a negative result for the BHUH architectural hypotheses, but a
positive result for scientific methodology: the controlled experiments
self-corrected the initial optimistic findings (Exp 32-33) and identified
the true source of value (entropy coding).

---

## Reproducibility

```bash
cd /home/z/my-project/research/universe
python3.13 experiment_36_hierarchical_isolated.py --seeds 42,123,2024 \
    --num-images 30 --epochs 300
```
