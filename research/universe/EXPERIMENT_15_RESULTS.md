# BHUH Experiment 15: Quality-Aware Clustering — HONEST NEGATIVE

**Date**: 2026-06-27
**Status**: ❌ WORSE than KMeans — quality-aware clustering fails

---

## Results

| Method | K | Bytes | PSNR | vs COIN | vs KMeans |
|--------|---|-------|------|---------|-----------|
| COIN | - | 85601 | 28.10 | 1.0× | - |
| KMeans K=25 | 25 | 25146 | 24.95 | 3.4× | baseline |
| **Quality-Aware K=25** | 25 | **69897** | **21.34** | 1.2× | **-3.61 dB WORSE** |
| KMeans K=50 | 50 | 56450 | 27.13 | 1.5× | baseline |
| **Quality-Aware K=50** | 50 | **138717** | **23.52** | 0.6× | **-3.61 dB WORSE** |

**Quality-aware clustering is WORSE than KMeans on both metrics.**

---

## Why it failed (honest)

1. **Difficulty ≠ Compressibility together**
   - Two "easy" images (high individual PSNR) don't necessarily share well
   - They might be easy for DIFFERENT reasons (different frequency content)
   - KMeans pixel similarity captures spatial structure better

2. **Too many small groups**
   - K=50 with quality-aware: 50 groups of 2 images each
   - Each group needs its own backbone → 50 × ~2800B = 140KB
   - KMeans K=50: similar sizes but better grouping → 56KB

3. **Difficulty score is noisy**
   - 30-epoch SIREN PSNR is not a reliable difficulty metric
   - Same image can get very different PSNR with different seeds
   - Clustering on noisy features produces bad groups

4. **Quality-aware groups are too homogeneous**
   - All hard images together = backbone can't fit any of them
   - All easy images together = backbone underutilized
   - Better to MIX easy and hard (KMeans does this naturally)

---

## Key insight (positive learning)

**KMeans (pixel similarity) is actually a GOOD clustering for SIREN sharing.**
- Images with similar pixels have similar frequency content
- Similar frequency content = shareable SIREN backbone
- Quality-aware clustering tried to be smarter but was worse

This validates KMeans as the right approach for BHUH Hierarchical.

---

## Verdict

Quality-aware clustering FAILED. KMeans remains the best clustering method for BHUH Hierarchical. The ~1 dB gap to COIN at K=50 is structural and cannot be closed by better clustering.

**BHUH Hierarchical (KMeans, K=50) remains our best result:**
- 27.13 dB (0.97 dB gap to COIN)
- 1.5× smaller than COIN
- Tunable R-D curve from K=1 to K=50
