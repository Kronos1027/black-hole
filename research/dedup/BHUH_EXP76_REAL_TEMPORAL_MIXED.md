# BLKH Experiment 76 — reversible temporal prediction on genuine microscopy sequences (MIXED)

**Date:** 2026-10-28 UTC. **Status:** ✅ complete and independently reproduced by GitHub Actions; a **small positive niche result in one of two real time series**, and a clear negative result in the other. No universal compression novelty, independent-video population proof or native uint16 lossless claim.

**Preregistered before compression measurements:** [BHUH_EXP76_PROTOCOL.md](BHUH_EXP76_PROTOCOL.md).  
**Successful reproducible CI:** https://github.com/Kronos1027/black-hole/actions/runs/40698575897  
**Implementation:** `research/dedup/experiment_76_temporal_predictor.py`; workflow `.github/workflows/exp76.yml`.  
**Evidence archive:** `exp76-real-temporal-prediction-lossless` includes `results.json`, all **22 actual saved compressed files**, original raw scientific source SHA256, per-frame decoded RGB checksums and exact standalone decoder verification.

## Why this experiment was needed

Exp74's exact 16×16 cross-frame block dictionary yielded 15/15 savings against native lossless animation comparators in **synthetically edited** photographic sequences. Exp75 then analyzed genuine real time series and found **512/512 unique exact tiles** in both; the dictionary was larger than the strongest valid WebP atlas in 2/2 cases. Thus, exact equality cannot exploit small nonidentical pixel changes induced by sensor noise, real temporal motion or lighting.

Exp76 tests four **precommitted established reversible** temporal residual variants:
- `xor_webp`: XOR each RGB byte of frame_t with frame_{t-1} (all zero previous frame for frame0), pack the eight residual frames into one 4×2 lossless WebP atlas;
- `xor_zlib`: same XOR residuals then zlib9;
- `mod_webp`: modular uint8 per-channel frame difference (subtract previous frame modulo 256), packed in one lossless WebP atlas;
- `mod_zlib`: same modular differences then zlib9.

All weights/model bytes? **There is no neural model**, only an exact temporal predictor and native compression. The complete source metadata, schema, frame hashes and all compressed bytes count. The saved decoder inverts predictors on **each previously decoded frame** and validates all eight decoded source hashes. Every candidate and every comparator was saved and independently decoded from actual bytes.

## True natural temporal source inputs, but *derived* color benchmark

- `skimage.data.protein_transport()`: original real uint8 fluorescence microscopy time series, first eight timepoints, channel 0, central 128² crop.
- `skimage.data.nickel_solidification()`: real original uint16 X-ray scientific temporal sequence, first eight timepoints, central 128² crop.

**Critical:** per precommitted Exp75–76 protocol, each cropped video was normalized **once across all eight frames** to 8-bit grayscale and copied into RGB channels. All codecs reconstruct those **derived uint8 RGB** pixels exactly, but **do not preserve nickel's original uint16 measurements**. These are benchmark RGB files, *not interchangeable replacements for the original scientific datasets*. The actual sample provenance and normalized source hashes are saved.

All baselines from Exp75 were re-encoded with the same metadata: WebP lossless animation, native APNG, WebP/PNG whole 4×2 atlas, zlib raw source, and both exact-dictionary variants.

## Official complete byte-counted result

| Genuine source | Best temporal predictor | Complete predictor bytes | Best conventional lossless RGB method | Complete standard bytes | Result |
|---|---|---:|---|---:|---|
| protein_transport | XOR + WebP | 88,502 | **whole WebP atlas** | **72,616** | Temporal **21.88% larger** |
| nickel_solidification | **modular delta + WebP** | **71,033** | whole WebP atlas | 73,231 | Temporal **3.00% smaller** |

**Outcome:** `1/2` independent real time series beat the strongest tested conventional lossless RGB image/animation atlas. It **fails** the precommitted `2/2` limited-success criterion. It cannot establish superiority of a general temporal codec, but documents a genuine within-source positive rate difference (2,198 complete bytes on nickel, with full input-equivalent reconstruction *after the fixed 8-bit conversion*).

The first source's XOR predictor is significantly larger than the WebP atlas. Crucially, these two sources are extremely different physically, and there are only two unrelated time series. One success must not be generalized into a broad marketing claim.

## Interpretation and limitations

- No exact repeated 16² tiles existed in these two real time-series datasets; temporal *predictability* can still be high enough to provide a 3% size difference when using modular deltas and WebP for the nickel sequence.
- Stronger real video codecs (FFV1, true lossless AV1, HEVC lossless, lossy video at matched distortion, optimized intra/inter codecs, JXL animation), scientific uint16 sequence formats, TIFF/PNG16/JPEG2000, motion estimation and hardware encoding are **not included**.
- No independent large video test set, dedicated CPU calibration, decoder speed/memory comparison or original scientific temporal frame-rate preservation (the 100ms playback interval is a benchmark convention only).
- Protocol calls all candidates **lossless with respect to derived RGB8**, not with respect to native raw sensor data. No claim is made that bit-depth normalization is lossless.

## Research decision

A substantial **restricted application capability** is emerging for user interfaces and exact repeated assets (Exp74) and a potentially useful modular-temporal path on low-motion scientific sequences (Exp76). However, the current body of evidence does **not** justify a BLKH "world-leading general image/video codec" claim.

**Demo/main/LinkedIn breakthrough announcement: HOLD.** Retain the current draft PR. An appropriate next major scientific milestone is a benchmark against actual user-generated sprite/UI states and real editable asset sequences, with source rights/provenance, file-size accounting, and a motion/patch-aware lossless baseline. For full scientific fidelity, introduce a **native uint16 temporal predictor with matched 16-bit lossless standards**, rather than evaluating uint16 input via 8-bit conversion.
