# Exp200 — Pretrained zlib dictionary holdout (2026-11-20)

Local experiment; six lossless configurations: RGB, green modular decorrelation, green+vertical residual; each with/without a 32768-byte dictionary trained exclusively on four development images. Standalone files include the entire dictionary, not an uncharged external dependency.

Source: eight bundled scikit-image examples resized to 128×128 RGB. Development: astronaut, coffee, chelsea, rocket. Held-out: camera, coins, horse, hubble_deep_field. All 48 encode/decode roundtrips exactly reconstructed RGB pixels. The selection was locked using development totals: **green_up without dictionary**.

Held-out four-image total: **56,177 bytes BLKH prototype vs 40,054 bytes WebP lossless (Pillow method=6)**; **40.25% larger**. Negative result. The dictionary variants pay 32,768 bytes per file and were not selected. Python 3.13.5; Pillow 12.3.0. This is an exploratory ablation, not an external photographic dataset or fair lossy rate-distortion study. AVIF/JXL, isolated memory, independent external corpus and GitHub Actions validation remain missing. No scientific breakthrough, no demo/LinkedIn recommendation.

Executable experiment and full per-image JSON are in the conversation artifact BLKH_exp200_bundle.zip; this report does not imply that executable code is committed or CI ran.
