# BHUH Experiment 28: Multi-Omega at N=100 — CONFIRMED ✅

**Date**: 2026-07-10

---

## Results (N=100, 10 epochs)

| Config | Layers | PSNR | Bytes | vs Default |
|--------|--------|------|-------|------------|
| [15,15] (default) | 2 | 15.08 dB | 2,751 B | baseline |
| **[10,50] (best)** | **2** | **16.36 dB** | **1,969 B** | **+1.29 dB, -28%** ✅ |
| [10,30,50] | 3 | 16.04 dB | 3,210 B | +0.96 dB, +17% |
| [10,50,10] | 3 | 15.47 dB | 3,867 B | +0.39 dB, +41% |
| [10,50,10,50] | 4 | 16.40 dB | 4,926 B | +1.32 dB, +79% |

---

## Key Findings

1. **2-layer [10,50] is optimal** — confirmed at N=100 (same as N=50)
   - +1.29 dB better than default
   - 28% smaller bytes
   - Adding more layers doesn't improve quality, only adds bytes

2. **More layers = more bytes, same quality**
   - 4-layer [10,50,10,50]: 16.40 dB (same as 2-layer 16.36) but 2.5× larger
   - Extra layers learn nothing new — just parameter overhead

3. **Pattern is clear**: 
   - **Low omega first (10)**: smooth foundation
   - **High omega second (50)**: fine detail
   - **2 layers is enough** — multi-scale captured in 2 bands

---

## Combined Multi-Omega Results (Exp 27 + 28)

| N | Default [15,15] | Multi [10,50] | Gain |
|---|----------------|---------------|------|
| 50 | 14.86 dB, 2725B | 16.43 dB, 1964B | +1.57 dB, -28% |
| 100 | 15.08 dB, 2751B | 16.36 dB, 1969B | +1.29 dB, -28% |

**Multi-omega [10,50] consistently beats default by ~1.3-1.6 dB AND 28% smaller.**

This is the FIRST improvement found after 8 failed hypotheses.
It's real, reproducible, and consistent across N=50 and N=100.
