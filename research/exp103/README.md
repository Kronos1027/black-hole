# Exp103 — Adaptive predictor prototype (research only)

This is a **new hypothesis implementation**, not a competitive compression claim. Four reversible predictors (none, left, up, Paeth) compete on the **entire** zlib-compressed residual stream. The complete 13-byte header is counted. The decoder reconstructs grayscale pixels without the original file. Format: `B103` + width uint32 BE + height uint32 BE + predictor uint8 + zlib payload. Max decode pixels: 100 million.

## Local exploratory results (2026-11-16)
Python stdlib, 64×64 8-bit **synthetic** grayscale data, seed 103:
- Smooth gradient: 40 bytes total prototype versus 243 bytes zlib(raw), lossless SHA-256-verified roundtrip.
- Uniform random noise: 4,120 bytes prototype versus 4,107 bytes zlib(raw), a loss.
- Unit tests: 2 passed, including 18 varied roundtrips and corruption rejection, in the local prototype. The committed implementation was simplified and must be independently rerun in CI.

**These synthetic numbers are not evidence of generalization or superiority.** No external images, PNG/WebP/AVIF/JPEG XL comparisons, fair rate–distortion curves, RAM peak measurements, or Actions runs were performed for this experiment.

## Reproduce
```sh
cd research/exp103
python -m unittest -v test_exp103.py
python exp103.py image.gray --width 512 --height 512
```
Use raw grayscale pixels, not a PNG container. Report actual encoded byte length including header. Future work: use disjoint independent image corpora, test lossless baselines including PNG, lossless WebP and JPEG XL, and only compare lossy codecs at matched distortion with a separately designed lossy variant. Record timing, peak RSS, software versions and decode correctness; add CI matrix and preserve failures. Do not claim a discovery from these exploratory tests.
