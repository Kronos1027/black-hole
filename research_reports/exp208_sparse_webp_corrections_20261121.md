# Exp208: WebP base + sparse RGB corrections (2026-11-21)

Local prototype uses a WebP lossy base image plus a zlib-compressed sparse stream of signed RGB pixel differences for pixels whose maximum channel error exceeds a threshold. Sidecar includes 32-bit indices and three signed 16-bit deltas. Decoder applies the deltas after decoding WebP. Complete byte count includes 8-byte envelope, WebP payload, and zlib sidecar.

Development: astronaut, coffee, chelsea (scikit-image), resized to 256x256 RGB. Holdout: rocket, hubble_deep_field, immunohistochemistry. Selection on development: WebP quality 80 and correction threshold 40.

Holdout results: rocket 6024 B vs WebP quality 82 at 6090 B (BLKH 34.72 dB; WebP 34.77 dB); hubble 11081 B vs WebP quality 82 at 11324 B (34.70 vs 34.81 dB); immuno 16256 B vs WebP quality 80 at 16236 B (both ~33.77 dB). Aggregate 33361 B vs 33650 B, 0.86% smaller. WebP quality 1..100 sweep, only comparing baselines with equal-or-higher PSNR. An earlier sparse quality sweep misleadingly suggested 22.8% advantage; denser sweep removed almost all of it. **Not a validated competitive breakthrough**: image set small/non-independent, tiny gain, and no AVIF/JXL RD curves, memory isolation or Actions validation.

Local runnable source and JSON are in the Exp208 run artifact; the repository currently contains only this report. Do not claim GitHub reproducibility. Next: upload exact code + pinned environment, external datasets and Actions workflow, then replicate with AVIF/JXL and a broader PSNR range. No demo or LinkedIn.
