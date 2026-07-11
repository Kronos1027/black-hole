# BHUH Experiment 11: KAN vs SIREN — HONEST NEGATIVE

**Date**: 2026-06-27
**Status**: ❌ KAN (simplified) LOSES to SIREN on all signals
**Based on**: Claude's suggestion from Liu et al., KAN (ICLR 2025)

---

## Results

| Image | SIREN PSNR | KAN PSNR | Diff | SIREN params | KAN params |
|-------|-----------|---------|------|-------------|-----------|
| Mandelbrot | 47.80 dB | 40.94 dB | -6.86 dB | 1185 | 1195 |
| Gaussian | 47.22 dB | 28.52 dB | -18.71 dB | 1185 | 1195 |
| Natural | 26.74 dB | 13.03 dB | -13.71 dB | 1185 | 1195 |

**KAN is WORSE than SIREN on ALL three signal types.**

---

## Honest Analysis

### ❌ KAN did NOT solve the Mandelbrot problem

Claude predicted KAN would beat SIREN on Mandelbrot (fractal). 
Result: KAN is 6.86 dB WORSE than SIREN on Mandelbrot.

### Why KAN failed (honest assessment)

1. **Our KAN is simplified** — we used piecewise-linear splines, not full B-splines
   - Full KAN library (effortlessKAN) uses proper B-spline basis functions
   - Our simplified version may not capture KAN's full power
   - This is NOT a fair test of KAN — it's a test of simplified KAN

2. **SIREN's sin activations are actually GOOD for high-frequency content**
   - SIREN achieved 47.80 dB on Mandelbrot (surprisingly good!)
   - Sin functions naturally represent periodic/fractal structure
   - The Mandelbrot "problem" from Phase 84 may have been about
     LARGER Mandelbrot images, not the 32×32 version tested here

3. **KAN needs more training epochs**
   - SIREN converges in 300 epochs
   - KAN's learnable splines need more epochs to find optimal knot positions
   - 300 epochs may be insufficient for KAN

4. **KAN is 3.5× slower** (3.5s vs 1.0s) — spline interpolation is expensive

### Important discovery: SIREN is GOOD on Mandelbrot at 32×32

SIREN achieved **47.80 dB** on 32×32 Mandelbrot — this is EXCELLENT quality.
The Phase 84 "failure" was on LARGER Mandelbrot (128×128+), where SIREN
needed 17025 bytes. At 32×32, SIREN handles Mandelbrot fine.

**This means the Mandelbrot "problem" is about SCALE, not architecture.**
SIREN can represent Mandelbrot — it just needs enough parameters for
larger versions.

---

## What Claude got WRONG

Claude's suggestion was theoretically sound but empirically wrong:
1. KAN does NOT automatically beat SIREN on fractals
2. SIREN's sin activations are well-suited for periodic/fractal content
3. The Kolmogorov-Arnold theorem guarantees REPRESENTATION, not EFFICIENCY
4. KAN's advantage may be in interpretability, not compression

---

## What we should do instead

1. **Test full KAN library** (effortlessKAN) — our simplified version may be unfair
2. **Test KAN with more epochs** (1000+) — splines need more training
3. **Test KAN on LARGER Mandelbrot** (128×128) — where SIREN actually struggles
4. **Don't abandon SIREN** — user's instinct was correct

---

## Verdict

**KAN (simplified) does NOT beat SIREN.** Claude's suggestion, while
theoretically reasonable, did not validate empirically with our
simplified implementation. 

**SIREN remains the best generator for BHUH.**

The user was RIGHT to resist "discarding SIREN" — SIREN is genuinely
good, even on fractals at small scale.

---

## Reproducibility

```bash
/usr/bin/python3 research/universe/experiment_11_kan.py
```

Expected: SIREN beats simplified KAN by 7-19 dB on all signals.
