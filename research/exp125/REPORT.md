# Exp125 — modular color lifting + spatial residuals (negative)

Date: 2026-11-17. Parent: Exp124. Branch is isolated from prior research.

Hypotheses: (H1) modular reversible YCoCg-like lifting decorrelates RGB better than green subtraction; (H2) applying left/up differences to transformed channels improves LZMA/zlib coding. Five modes (raw, green, ycocg, ycocg_left, ycocg_up) × two backends (zlib-9, LZMA-6). Full 14-byte BLKH container header included. Modular lifting uses co=(r-b)%256, t=(b+(co>>1))%256, cg=(g-t)%256, y=(t+(cg>>1))%256, and exact inverse; this is NOT conventional signed YCoCg-R.

Development: astronaut, coffee, chelsea, rocket. Held-out test: camera, hubble_deep_field, immunohistochemistry, colorwheel. Source: skimage.data; all resized to 128×128 RGB, so results are lossless only relative to resized inputs, not original files. Development selects a single configuration before evaluating test: ycocg_up + LZMA-6. 80 encode/decode pixel-exact checks across 8 images and 10 configs, plus 8 WebP lossless and 8 PNG independent decode checks. Three invalid headers rejected.

| Test image | BLKH bytes | WebP lossless bytes |
|---|---:|---:|
| camera | 10950 | 8596 |
| hubble_deep_field | 32182 | 22604 |
| immunohistochemistry | 36390 | 29998 |
| colorwheel | 12358 | 7826 |
| **Total** | **91880** | **69024** |

BLKH is **33.1% larger** than WebP on the held-out set; loses all four. Negative result. Earlier Exp124 held-out result: 84284 vs 69024; modular lifting worsened the baseline by 7596 bytes. No claims of competitive advantage.

Local artifacts: /mnt/data/exp125/experiment.py and results.json. Network from container to GitHub unavailable (DNS resolution failure), so the executable and raw JSON were not pushed in this run. GitHub Actions reproducibility **not verified**. Baselines AVIF/JPEG XL unavailable; no rate-distortion curve. Peak RSS in results includes entire Python process and is not isolated per codec. Future: publish executable and pinned dependency workflow, test independent full-resolution photographs and AVIF/JXL, isolate encode/decode memory and timing. No demo or LinkedIn publication authorized.
