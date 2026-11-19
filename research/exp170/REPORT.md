# BLKH Exp170 — green XOR versus green subtraction (2026-11-19)

## Protocol
Eight scikit-image images resized to RGB 128×128 (Lanczos). Four development images: astronaut, coffee, chelsea, rocket. Four held-out: camera, coins, grass, gravel. 24 configurations: raw, green modular subtraction, green XOR; predictors none/up/left/average; zlib-9 or LZMA-6. All file sizes include 20-byte header, compressed payload, 4-byte CRC. Every configuration underwent exact RGB encode/decode and deliberate payload corruption rejection (192 each). Selection by aggregate development bytes only. Pillow WebP lossless quality=100 method=6 and PNG level=9, with exact RGB decode checks.

## Findings
Selected gsub/up/lzma on development (105128 bytes versus WebP 89402). Held-out: BLKH 59012 bytes, WebP lossless 46982 bytes, PNG 99084 bytes. BLKH **25.60% larger than WebP**. PNG comparison does not imply codec superiority. Result negative.

## Limitations
Images are tiny resized scikit-image samples, not external independent photographs; no AVIF/JXL encoder comparison or lossy RD curves; tracemalloc is Python allocation tracking, not isolated peak RSS; GitHub Actions not run. Code and full JSON results currently exist in local Exp170 artifacts, not yet committed to GitHub. Do not claim reproducibility in CI. No demo or LinkedIn claims.

## Follow-up
Commit executable experiment.py and results.json, add pinned dependencies and Actions job, run external image benchmarks with SHA256 and separate splits, isolate per-codec memory/time and add JPEG XL/AVIF lossless and lossy RD baselines.