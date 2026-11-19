# Exp176 — Row-average and green-relative predictor ablation (2026-11-19)

## Protocol
Local Python 3.13.5 / Linux, Pillow WebP lossless method=6 and scikit-image sample images resized to 128×128 RGB. Four development images (astronaut, coffee, chelsea, rocket) and four untouched holdout images (camera, coins, moon, page). Eight variants: RGB/green-relative × fixed left/up/average/adaptive per-row predictor. All use LZMA preset 6. Adaptive selector minimizes modular signed absolute residual per row, with selector bytes included in compressed payload. Container includes 14-byte header and compressed payload. Reconstructed RGB bytes checked for exact equality for all 64 variant-image pairs; WebP independently decoded and checked for all eight images.

## Result
Chosen solely on development: green_adaptive. Development: BLKH 104,180 bytes vs WebP lossless 89,746 bytes. Holdout: BLKH **45,584 bytes** vs WebP lossless **36,014 bytes** (**26.57% larger**). No competitive gain. Shared-process ru_maxrss=147,628 KiB; not codec-isolated. Per-image sizes, encode/decode timings and SHA256 were recorded locally in results.json.

## Limitations
No external independent dataset, AVIF/JXL comparison, lossy rate-distortion curve, isolated memory or GitHub Actions execution. Source samples are built into scikit-image and include grayscale images converted to RGB. Do not claim superiority or publish demo/LinkedIn. Local executable and JSON are not committed here; workflow reproduction remains pending. Previous Exp175 already explored row selection; this ablation specifically adds an arithmetic-average predictor and explicit green-relative transform under a single container.

## Reproduction dependency
Run local experiment.py with numpy, scikit-image and Pillow WebP support, Python 3.13; holdout split is fixed as above. Record all bytes, reconstruct hashes, times and separate peak RSS by process before considering competitive claims.
