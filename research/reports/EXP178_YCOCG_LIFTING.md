# BLKH Exp178 — reversible YCoCg lifting versus green-difference and RGB

2026-11-19. New hypothesis: reversible YCoCg-R channel lifting (modulo 256), paired with Up/Paeth and zlib-9/LZMA-6. Twelve combinations, eight 128x128 RGB images (96 exact roundtrips, 96 CRC corruption checks). Images: scikit-image built-ins; development: astronaut, coffee, chelsea, rocket; held-out: camera, coins, moon, page. Fixed split, no test-set tuning.

**Dev-selected:** gsub/paeth/lzma. **Held-out full-file bytes:** BLKH 45,400; WebP lossless (Pillow method=6) 34,732. **BLKH is 30.72% larger**. YCoCg did not win development selection. Each WebP decoded to exact RGB. Per-run encoding/decoding times and SHA-256 in local results.json. Peak process RSS 147,836 KiB (shared, not isolated). Python 3.13.5, Pillow 12.3.0.

**Limitations:** scikit-image built-in images only, no independent external corpus, no AVIF/JPEG XL or lossy R-D curves, no GitHub Actions reproduction. Executable script and full JSON are in the run artifact BLKH_exp178_bundle.zip; these are not yet in this branch. This report does NOT constitute a competitive compression claim. Next run should import script/results, execute Actions, test external photographic corpus and additional codecs. No demo/LinkedIn warranted.
