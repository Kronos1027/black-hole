# Exp128 — per-channel predictor ablation (negative)

Date: 2026-11-17 (America/Sao_Paulo). Continuation of Exp127; no changes to earlier branches.

## Hypothesis
Allow each RGB channel to select its own reversible spatial predictor (raw, left, up, integer-average), instead of a shared global predictor. Test all 4^3=64 combinations, planar channel order, LZMA preset 6. Complete B128 header: 4 magic + 2 height + 2 width + 3 predictor IDs + 4 payload length = 15 bytes. Each candidate was decoded and compared bit-for-bit to its original RGB array.

## Protocol
Eight scikit-image source images resized to 128x128 RGB with Pillow LANCZOS. Development: astronaut, coffee, chelsea, rocket. Held-out: camera, hubble_deep_field, immunohistochemistry, colorwheel. Select one configuration minimizing total development bytes, freeze before held-out scoring. WebP lossless via Pillow method=6, decode equality asserted. 512 exact candidate reconstructions; 8 WebP decode checks; malformed header rejection. Dataset is *not* independent external photographs at original resolution; results cannot establish generalization. Reported bytes include full B128 file overhead. Memory is process-wide peak only, not isolated by codec. No AVIF/JXL, lossy rate-distortion curves, or GitHub Actions reproduction.

## Results
Development-selected configuration: up/up/up. Held-out B128 bytes vs WebP: camera 9,251 vs 8,596; hubble_deep_field 31,911 vs 22,604; immunohistochemistry 37,051 vs 29,998; colorwheel 8,603 vs 7,826. **Total 86,816 vs 69,024 bytes, B128 +25.78%**. Lost all four images. Prior Exp127 selected green+up planar LZMA, yielding 78,596 bytes on this holdout: Exp128 is worse. This does not show useful competitive compression.

## Reproducibility blocker
Executed script and JSON were produced in local runtime at /mnt/data/exp128/experiment.py and /mnt/data/exp128/results.json. Local git network access failed: 'Could not resolve host: github.com'. GitHub connector can commit this report, but script/JSON are **not yet in repository**. Therefore no CI reproducibility is claimed. Next run: transfer full source and JSON, add GitHub Actions job with pinned dependencies, test original-resolution independent images, WebP/JXL/AVIF when installed, log per-codec time/RSS, and evaluate fair RD curves.

Decision: negative result, no demo or LinkedIn announcement.
