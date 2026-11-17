# Exp109 — entropy backend ablation (2026-11-17)

Hypothesis: replacing zlib with Brotli or LZMA after reversible RGB color decorrelation and Paeth prediction may improve compression on photographs. This is a **negative result against WebP lossless**. This experiment was executed locally using Python 3.13.5, Pillow 12.3.0, scikit-image 0.26.0 and Brotli 1.2.0. All 7 decode roundtrips matched original RGB bytes; PNG and WebP lossless decodes also matched. Width/height and backend/color IDs are included in the 14-byte prototype header. The encoder selects minimum payload length over two color modes and three entropy backends, using no external learned model.

| image | Exp109 bytes | PNG bytes | WebP lossless bytes | chosen backend |
|---|---:|---:|---:|---|
| astronaut | 15918 | 18886 | 14210 | LZMA |
| chelsea | 10222 | 12367 | 9094 | LZMA |
| coffee | 11538 | 12716 | 9742 | LZMA |
| rocket | 7682 | 8360 | 6988 | LZMA |
| hubble_deep_field | 12742 | 15308 | 10702 | LZMA |
| immunohistochemistry | 18374 | 20859 | 17046 | LZMA |
| colorwheel | 6438 | 7862 | 4578 | LZMA |

Note: local benchmark actually chose LZMA for all seven images, including colorwheel (identity color transform). Encode wall time 0.089–0.136 s, decode 0.0091–0.0155 s (single samples, not robust). Process peak RSS 177584 KiB (includes image libraries and datasets). This was an exploratory same-library sample, not an external independent holdout; no AVIF/JPEG XL measurements or matched-distortion lossy RD analysis. The local benchmark implementation and JSON results were created in a separate working environment. **Do not treat this report as CI-reproducible until executable code and data are committed and GitHub Actions succeeds.** The next step is to commit the exact local source and workflow, independently validate results, then move to a genuinely external holdout. Do not publish a demo or LinkedIn superiority claim.

No prior branch or automation was changed.