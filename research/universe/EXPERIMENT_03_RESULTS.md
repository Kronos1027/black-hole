# BHUH Experiment 3: Meta-Learning — HONEST NEGATIVE RESULT

**Date**: 2026-06-27
**Status**: ❌ INVALID — Meta-learning did not help
**Based on**: COIN++ (Dupont et al., 2022)

---

## Result

| Method | Total Bytes | Avg PSNR | Avg Time |
|--------|------------|----------|----------|
| COIN (separate) | 4388 | 25.96 dB | 0.24s |
| Meta (COIN++ style) | 9275 | 17.13 dB | 0.43s |

- Size ratio: 0.47× (Meta is LARGER)
- Time ratio: 0.24× (Meta is SLOWER)
- PSNR diff: -8.83 dB (Meta is MUCH WORSE quality)

**Meta-learning FAILED on all three metrics.**

---

## Why it failed (honest analysis)

### 1. Insufficient meta-training data
- Only 5 images for meta-training (COIN++ uses 50,000+)
- 300 epochs insufficient (COIN++ uses 10,000+)
- Base model didn't learn general image structure

### 2. Base model overhead dominates at N=5
- Base model compressed: 8882 bytes
- Per-image modulation: 79 bytes
- Total for 5 images: 8882 + 5×79 = 9277 bytes
- COIN: 5 × 878 = 4390 bytes
- **Base model overhead > modulation savings at small N**

### 3. Modulation too weak
- FiLM with dim=64 and 0.1× scale insufficient
- Cannot represent diverse image content
- PSNR dropped 8.83 dB (unacceptable)

### 4. FiLM forward pass is expensive
- Each layer: extra Linear(64→2×hidden) computation
- Slower than plain SIREN forward
- Negated the "fewer params to optimize" advantage

---

## What we LEARNED (positive takeaways)

### 1. COIN is faster than expected
- COIN at 64×64 with 200 epochs: only 0.24s/image
- At this speed, N=50 would take ~12s total
- **No need for meta-learning — COIN is already fast enough at 64×64**

### 2. Meta-learning needs scale
- COIN++ works because of massive meta-training (50K images)
- With 5 images, meta-learning can't generalize
- This is consistent with literature: meta-learning is data-hungry

### 3. FiLM modulation insufficient
- 64-dim modulation cannot represent 4096 pixels
- Need larger modulation (256? 512?) — but then no compression
- COIN++ uses more sophisticated modulation (hypernetwork)

---

## Honest implication for BHUH scaling experiments

**Good news**: COIN at 64×64 is fast (0.24s/image). We CAN run scaling experiments.

| N | COIN time (projected) | Feasible? |
|---|----------------------|-----------|
| 5 | 1.2s | ✅ Yes |
| 10 | 2.4s | ✅ Yes |
| 20 | 4.8s | ✅ Yes |
| 50 | 12s | ✅ Yes |
| 100 | 24s | ✅ Yes |

**We don't need meta-learning.** We can run the full scaling experiment with COIN at 64×64.

---

## What to do next

1. **Run Experiment 2 properly** at 64×64 with N=3, 5, 10, 20
   - COIN is fast enough (0.24s/image)
   - BHUH shared should also be feasible
2. **If BHUH shared is also fast at 64×64**: full scaling validation
3. **Don't pursue meta-learning** without large dataset (50K+ images)

---

## Verdict

**Meta-learning (COIN++ style) FAILED with small dataset.** This is honest negative result. But it revealed that COIN at 64×64 is fast enough for scaling experiments.

**Next step**: Run scaling experiment at 64×64 with N=3, 5, 10, 20 — should now complete in reasonable time.
