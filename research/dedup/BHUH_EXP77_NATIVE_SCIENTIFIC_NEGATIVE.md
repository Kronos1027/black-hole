# BLKH Exp77 — native uint16 exact temporal compression (NEGATIVE vs real video standards)

**Status: GitHub Actions PASSED with independent persisted-byte audit.** The new scientific experiment resolved an important limitation in Exp75–76: source native unsigned integer values, especially the **uint16 nickel X-ray acquisition values**, survive complete serialized codec decoding **without any 8-bit normalization, RGB conversion, rounding or loss**. Nevertheless, conventional codecs won in complete file bytes on **both** actual time-series sources.

- **Preregistered fixed methods:** [BHUH_EXP77_NATIVE_16BIT_PROTOCOL.md](BHUH_EXP77_NATIVE_16BIT_PROTOCOL.md), committed before this experiment's measurements.
- **Official successful workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40699846404
- **Code:** `research/dedup/experiment_77_native_uint16_temporal.py`; CI `.github/workflows/exp77.yml`.
- **Full official evidence:** `exp77-native-uint16-scientific-video-lossless` artifact with actual `.bl77` compressed files, source full-movie/crop/native-frame SHA256 and all recorded results.

## Frozen original scientific sources

1. `skimage.data.protein_transport()` — first 8 actual timepoints, channel 0, center 128×128 crop, **original uint8**; original sample values preserved exactly.
2. `skimage.data.nickel_solidification()` — first 8 actual timepoints, center 128×128 crop, **original uint16**; original sample values preserved exactly.

These sources were already seen in Exp75/76, so are **NOT independently held out new datasets**. Frame timing 100ms is an assigned benchmark playback convention, **not** the original scientific recording interval. Full original multi-frame source SHA256, original crop SHA256, native input data type, temporal indices, exact decoded sample hashes and unmodified native range are stored in each counted manifest.

## Exact codec competition

Four predeclared temporal reversible predictors: consecutive-frame **XOR** or **modular subtraction** on **native 8/16-bit values**, compressed with **ZSTD level 9** or **Brotli level 9**. All pixel values of frame0 are encoded relative to zeros. No adaptive predictor tuning, lossy preprocessing or per-frame parameter fitting.

Strong references:
- full 4×2 original-grayscale 8-frame atlas encoded as PNG (8bit or uint16 native precision),
- reversible JPEG2000 lossless 8/16bit atlas,
- tiled GeoTIFF ZSTD predictor=2 native bit depth,
- **FFV1** native gray8/gray16le, stored in a Matroska container with real FFmpeg decode,
- concatenated exact native-bit-depth source values compressed with ordinary ZSTD level 9.

All **18 saved complete files**, nine per actual source, include matching original-source metadata (full header size counted). Both the experiment and a **separate GitHub CI audit** decoded all stored files independently and verified every original unsigned integer sample and SHA256. Random uint8/uint16 complete-codec corruption tests also passed.

## Official complete, metadata-counted byte sizes

| Codec | protein_transport native uint8 (B) | nickel_solidification native uint16 (B) |
|---|---:|---:|
| mod_ZSTD | 90,069 | 132,037 |
| XOR_ZSTD | 85,884 | 141,666 |
| mod_Brotli | 87,599 | **129,456 (best custom)** |
| XOR_Brotli | **84,960 (best custom)** | 139,508 |
| PNG lossless atlas | 80,899 | 147,887 |
| JPEG2000 lossless atlas | 80,978 | 168,452 |
| TIFF-ZSTD predictor=2 atlas | 91,779 | 151,108 |
| FFV1 lossless MKV | 78,888 | **118,214 (best reference)** |
| Raw original native source + ZSTD | **76,913 (best reference)** | 146,593 |

**Precommitted winner criterion:** custom method strictly smaller than all five actual native-depth lossless competitors **on both** genuine scientific time series. Result: **0/2**. Protein: custom best 84,960 B vs standard 76,913 B, **10.46% extra bytes**. Nickel: custom best 129,456 B vs FFV1 118,214 B, **9.51% extra bytes**.

## Why this corrects earlier interpretations

Exp76 offered a limited 3.0% win on nickel against a WebP atlas, but it **first converted original uint16 measurements into globally normalized uint8 RGB**. It did not show a native-scientific lossless compression advantage. Exp77 addresses this directly, retains the original 16-bit samples, and compares against stronger **FFV1 and other genuine high-bit-depth lossless** methods. Neither native source demonstrates a custom win. Both Exp76's genuine positive *derived-RGB benchmark* and Exp77's negative *native-uint16 benchmark* remain accurate in their stated settings; they must never be combined into a misleading native-data claim.

Limitations: only two previously studied independent time-series sources; no extra scientific recordings, 256² atlas representation derived from eight 128² frames, one shared runner, relative library/encoding speeds not normalized, FFV1 settings are fixed rather than exhaustively optimized, source timing and geospatial/medical metadata not directly embedded in the native competitor containers (though complete shared manifest was counted for each). This is not a broad all-codecs theoretical result.

## Release decision / future meaningful research

- **Reject the hypothesis** that the four simple temporal predictors create a competitive native-scientific video codec on these two studied sources.
- The verified Exp74 positive result remains a **narrow *simulated exact-tile-update* engineering advantage**; Exp75 showed it does not transfer to two actual natural sensor movies with zero exact tile duplicates.
- The original BLKH core may still have promise for **real-world source-asset versioning** such as sprites/UI textures that genuinely change through discrete authored revisions rather than sensor noise. A scientifically sound next experiment needs *actual independent open-licensed asset revisions*, tracked source hashes and version provenance, and a strong native lossless screen/animation/atlas or content-dedup standard.
- Native-uint16 compression would require a new independently sampled scientific dataset and improved predictive modeling, not repeated retuning on the same two acquisitions.

**Demo, main and LinkedIn breakthrough: HOLD.** All work stays in draft research PR #1.
