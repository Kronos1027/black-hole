# EXPERIMENT 31 RESULTS — L1 Pruning Threshold Sweep

## Status: COMPLETED — Sweet spot identified at threshold=0.001

**Date**: 2026-08-05
**Experiment**: 31 — L1 Pruning Threshold Sweep
**Goal**: Map the size-PSNR tradeoff curve across pruning thresholds
[0.001, 0.002, 0.005, 0.01] and identify the sweet spot that balances
size reduction and PSNR preservation.

---

## Raw Results — COIN Baseline (reused from Exp 29)

| Metric | Value |
|--------|-------|
| Mean PSNR | 64.0930 dB |
| Mean weights size (float16) | 42114.0 B |

---

## Aggregated Results (3 seeds × 2 hidden_layers × 4 thresholds = 24 runs)

### hidden_layers = 2

| Threshold | PSNR (dB) | Size (B) | Reduction vs COIN (x) | Sparsity |
|-----------|-----------|----------|-----------------------|----------|
| 0.001 | **36.04 ± 0.32** | 9515 ± 169 | 4.43 ± 0.08 | 4.43% |
| 0.002 | 32.42 ± 0.38 | 9724 ± 70 | 4.33 ± 0.03 | 8.86% |
| 0.005 | 22.54 ± 0.56 | 9270 ± 89 | 4.54 ± 0.04 | 21.90% |
| 0.01 | 14.95 ± 0.27 | 7995 ± 35 | 5.27 ± 0.02 | 40.77% |

### hidden_layers = 5

| Threshold | PSNR (dB) | Size (B) | Reduction vs COIN (x) | Sparsity |
|-----------|-----------|----------|-----------------------|----------|
| 0.001 | **33.41 ± 0.25** | 15767 ± 233 | 2.67 ± 0.04 | 6.79% |
| 0.002 | 24.95 ± 0.27 | 15538 ± 110 | 2.71 ± 0.02 | 13.46% |
| 0.005 | 15.70 ± 0.23 | 14177 ± 89 | 2.97 ± 0.02 | 32.05% |
| 0.01 | 11.88 ± 0.15 | 10969 ± 63 | 3.84 ± 0.02 | 55.91% |

---

## Size-PSNR Tradeoff Curve

The data reveals a clear non-linear tradeoff. PSNR drops steeply as threshold
increases, while size reduction improves only modestly:

```
hl=2:
  thr=0.001 → 36.04 dB, 4.43x   ← SWEET SPOT (best PSNR, good size)
  thr=0.002 → 32.42 dB, 4.33x   ← -3.6 dB, -0.10x (bad trade)
  thr=0.005 → 22.54 dB, 4.54x   ← -13.5 dB, +0.11x (terrible trade)
  thr=0.01  → 14.95 dB, 5.27x   ← -21.1 dB, +0.84x (catastrophic trade)

hl=5:
  thr=0.001 → 33.41 dB, 2.67x   ← SWEET SPOT (best PSNR, smallest size)
  thr=0.002 → 24.95 dB, 2.71x   ← -8.5 dB, +0.04x (bad trade)
  thr=0.005 → 15.70 dB, 2.97x   ← -17.7 dB, +0.30x (terrible trade)
  thr=0.01  → 11.88 dB, 3.84x   ← -21.5 dB, +1.17x (catastrophic trade)
```

### Key Observation

Going from threshold=0.001 to 0.01 (10x more aggressive pruning):
- hl=2: loses 21.1 dB PSNR to gain only 0.84x size reduction
- hl=5: loses 21.5 dB PSNR to gain only 1.17x size reduction

**The size gains from aggressive pruning are marginal; the PSNR costs are
catastrophic.** Threshold=0.001 is the clear sweet spot.

---

## Byte Parity Analysis: BHUH vs COIN at Same PSNR

**IMPORTANT CAVEAT**: The COIN size at same PSNR is ESTIMATED via the
6 dB/bit quantization heuristic (each bit removed from weight quantization
reduces PSNR by ~6 dB and halves size). This is NOT a measured comparison.
A rigorous comparison would require running COIN at multiple quantization
levels (float16, float8, float4, etc.).

| Config | BHUH PSNR (dB) | BHUH Size (B) | Est. COIN Size at Same PSNR (B) | BHUH/COIN Ratio |
|--------|----------------|---------------|---------------------------------|-----------------|
| hl=2 thr=0.001 | 36.04 | 9515 | 1649 | **5.77x larger** |
| hl=2 thr=0.002 | 32.42 | 9724 | 1085 | 8.96x larger |
| hl=2 thr=0.005 | 22.54 | 9270 | 346 | 26.77x larger |
| hl=2 thr=0.01 | 14.95 | 7995 | 144 | 55.43x larger |
| hl=5 thr=0.001 | 33.41 | 15767 | 1216 | 12.96x larger |
| hl=5 thr=0.002 | 24.95 | 15538 | 458 | 33.95x larger |
| hl=5 thr=0.005 | 15.70 | 14177 | 157 | 90.21x larger |
| hl=5 thr=0.01 | 11.88 | 10969 | 101 | 108.50x larger |

### Interpretation

At every threshold, BHUH is **significantly larger** than the estimated COIN
size at the same PSNR. The best case (hl=2, thr=0.001) is 5.77x larger than
the estimated COIN equivalent.

**This means BHUH loses to COIN on byte parity at every threshold tested.**
The size advantage BHUH appears to have (4.43x reduction vs COIN's float16)
disappears when you account for the fact that COIN at 36 dB would be much
smaller than COIN at 64 dB.

### Honest Limitation

The 6 dB/bit heuristic is an approximation. Real quantization behavior
depends on the weight distribution. For SIREN weights (which have a specific
distribution due to the sin activation), the actual PSNR-vs-quantization
curve may differ. A rigorous comparison requires:
1. Running COIN with float8, float4, float2 quantization
2. Measuring actual PSNR at each level
3. Comparing byte sizes at matched PSNR points

This is left as future work (Exp 32).

---

## SHA-256 Verification (Anti-Fabrication)

### Output JSON

File: `/home/z/my-project/research/universe/_exp31_out/experiment_31_results.json`

```
SHA-256: ea93885b5c4d51d03536b6e40d836a217b2702035af6f615c072fbe8fe6ff88d
```

### Weights files (24 files, one per run)

| hl | threshold | seed | SHA-256 (first 16 chars) |
|----|-----------|------|--------------------------|
| 2 | 0.001 | 42 | 8f1321d65b1b02e9... |
| 2 | 0.001 | 123 | e4f64337cbc75eca... |
| 2 | 0.001 | 2024 | 543887504d400102... |
| 2 | 0.002 | 42 | b6beac36531fd3a8... |
| 2 | 0.002 | 123 | 99cb7d275d87383f... |
| 2 | 0.002 | 2024 | ade361c2384aa95f... |
| 2 | 0.005 | 42 | ea04ca380774ca25... |
| 2 | 0.005 | 123 | ddce71b84f688607... |
| 2 | 0.005 | 2024 | f2f0b59b01ab1e86... |
| 2 | 0.01 | 42 | d5465775e5991c9e... |
| 2 | 0.01 | 123 | 452da3644733b8f3... |
| 2 | 0.01 | 2024 | 29d50edd20d039b4... |
| 5 | 0.001 | 42 | a332c148e4457070... |
| 5 | 0.001 | 123 | c1941a93be923015... |
| 5 | 0.001 | 2024 | f6a40bcb23224228... |
| 5 | 0.002 | 42 | d31fc286d8ba79a0... |
| 5 | 0.002 | 123 | 3cdf1575dae65db8... |
| 5 | 0.002 | 2024 | a4b3ad4d73c21eda... |
| 5 | 0.005 | 42 | ee9c8096084dea33... |
| 5 | 0.005 | 123 | 4fa4b50bd3a3d1b0... |
| 5 | 0.005 | 2024 | 9562b997fb2d9f05... |
| 5 | 0.01 | 42 | d3211cc0509198d7... |
| 5 | 0.01 | 123 | 7317bd47fddc9e1d... |
| 5 | 0.01 | 2024 | c3e06a70ee3ab4ec... |

---

## What This Means

### Finding 1 — Sweet spot is threshold=0.001

At threshold=0.001:
- hl=2: PSNR 36.04 dB (close to the 37.05 dB no-prune baseline from Exp 30)
- hl=5: PSNR 33.41 dB (only 4% below the no-prune baseline of 49.54 dB... wait, that's 32% below)

Actually, hl=5 at thr=0.001 loses 16.13 dB vs no-prune (49.54 → 33.41). That's
still a significant loss. The hl=2 configuration is more robust to pruning:
only loses 1.01 dB (37.05 → 36.04).

### Finding 2 — hl=2 is more pruning-robust than hl=5

At every threshold, hl=2 retains more PSNR than hl=5:
- thr=0.001: hl=2 36.04 dB vs hl=5 33.41 dB (hl=2 wins by 2.63 dB)
- thr=0.01: hl=2 14.95 dB vs hl=5 11.88 dB (hl=2 wins by 3.07 dB)

This is counterintuitive — deeper networks have more capacity, so they should
be more robust to pruning. But the data shows the opposite. The likely
explanation: deeper SIREN networks have more weights in the "small but
critical" regime, making them more vulnerable to magnitude-based pruning.

### Finding 3 — The size-PSNR tradeoff is highly non-linear

The marginal size gain from increasing threshold diminishes rapidly:
- 0.001 → 0.002: +0% size gain (actually slightly worse for hl=2), -3.6 dB PSNR
- 0.002 → 0.005: -5% size gain, -9.9 dB PSNR
- 0.005 → 0.01: -14% size gain, -7.6 dB PSNR

Each step trades more PSNR for less size reduction. The first step (0.001 → 0.002)
is particularly bad: you lose 3.6 dB and gain NOTHING in size.

### Finding 4 — BHUH loses to COIN on byte parity at every threshold

Even at the sweet spot (thr=0.001), BHUH is 5.77x larger than the estimated
COIN size at the same PSNR. This means:
- BHUH's "4.43x reduction vs COIN" is misleading — it compares BHUH at 36 dB
  to COIN at 64 dB.
- When you match PSNR, COIN (even with aggressive quantization) would be smaller.

**This is a significant negative finding.** The BHUH pipeline as currently
configured does not beat COIN on the rate-distortion frontier.

### Finding 5 — The byte parity analysis has a critical limitation

The 6 dB/bit heuristic assumes uniform quantization noise, which doesn't hold
for SIREN weights. The actual COIN-at-same-PSNR size could be different:
- If SIREN weights quantize worse than uniform, COIN would be even smaller
  (BHUH loses by more)
- If SIREN weights quantize better than uniform, COIN would be larger
  (BHUH might win in some configs)

A rigorous answer requires Exp 32: run COIN at multiple quantization levels
and measure actual PSNR-vs-size curves.

---

## Recommended Next Steps

- **Exp 32**: Run COIN with float8, float4, float2 quantization to get the
  actual rate-distortion curve, then compare BHUH to COIN at matched PSNR
  points. This will definitively answer whether BHUH ever beats COIN.

- **Exp 33**: Test structured pruning (prune entire neurons/channels) instead
  of unstructured L1. Structured pruning may preserve more PSNR because it
  removes redundant features rather than individual weights.

- **Exp 34**: Test magnitude-based pruning at fixed sparsity ratios (10%,
  20%, 30%) rather than fixed thresholds. This gives a cleaner comparison
  across architectures.

---

## Reproducibility

```bash
cd /home/z/my-project/research/universe
python3.13 experiment_31_pruning_sweep.py --seeds 42,123,2024 \
    --num-images 30 --epochs 300 --hidden-layers 2,5 \
    --thresholds 0.001,0.002,0.005,0.01
```

Expected runtime: ~90 minutes (24 runs of ~3.5 minutes each).

Checkpoints are cached in `_exp31_out/ckpt_hl{N}_thr{T}_seed{S}.json`.
