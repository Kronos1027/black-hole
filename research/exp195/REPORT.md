# Exp195 — palette-index vs planar lossless (2026-11-20)

Independent local execution, Python 3.13.5, Pillow 12.3, numpy 2.3.5, skimage samples. Six configurations: RGB/planar/palette indexed × zlib9/LZMA6. Four development and four held-out images, 256×256 RGB. All successful encodes decoded exactly; truncation rejected. Palette only applies when <=256 unique colors and was not eligible for universal selection on photographs.

Development-only winner: planar/LZMA6, 161032 bytes over four images. Held-out: **175332 BLKH bytes vs 129722 WebP lossless bytes (35.16% larger)**. Eight images, 6 variants with palette variants omitted where inapplicable. Complete 13-byte BL95 header included. Memory peak 153508 KiB process-wide, not codec-isolated.

**Negative result.** No AVIF/JXL, no external independent photo dataset, no RD curves, no Actions validation. Do not claim a competitive discovery or publish demo/LinkedIn. Local executable: BLKH_exp195_bundle.zip from the automation's runtime (not committed here). Next step: incorporate executable and Actions, independent images and fair multi-codec benchmarks. Previous Exp185–194 already explored adaptive predictors, YCoCg, bitplanes, lifting, etc.; this tests palette-indexing and planar layout.
