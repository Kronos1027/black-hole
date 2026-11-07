# BLKH Experiment 77 — native-bit-depth temporal coding (precommitted protocol)

**Registered before any Experiment 77 compression measurements.** This is a benchmark of established reversible differences plus standard entropy codecs; **not** a new compression theory or neural codec.

## Scientific input and independence

Use exactly two previously studied, real, independent temporal acquisitions:
- `skimage.data.protein_transport()`, first 8 real frames, channel 0, central 128×128 integer crop, original `uint8`;
- `skimage.data.nickel_solidification()`, first 8 real frames, central 128×128 integer crop, original `uint16`.

**Do not normalize, rescale, stretch, clone grayscale into RGB, clamp, or drop bit depth.** The exact source crop's native unsigned integer pixels and original frame order must survive independently decoded from each archive with byte-for-byte equality. Save original full-source SHA-256, full dimensions, selected crop SHA-256, dtype, 8 frame SHA-256 values and the original numeric min/max in the byte-counted manifest. Fixed benchmark playback time 100ms per frame is artificial; it is *not* original acquisition timing.

These are the **same** two acquisitions previously seen in Exp75/76; not an independent new external holdout. Source frames are correlated observations.

## Prespecified temporal methods

For each source's actual native width (8 or 16 bits), current frame0 has zero-valued predecessor and frame1...7 predict from the immediately previous original frame. All differences are **exactly** modulo `2^bits`.

- `mod_zstd`: modular integer subtraction then zstandard compression level 9, one complete contiguous temporal stream;
- `xor_zstd`: bitwise XOR then zstandard level 9;
- `mod_brotli`: same modular differences then Brotli level 9;
- `xor_brotli`: same XOR differences then Brotli level 9.

No per-sequence or per-frame tuning, training, block-size selection, motion compensation, quantization or learned parameters. The codec inputs and output stream are native little-endian `uint8/uint16` pixels in chronological frame-major order. Decoding proceeds sequentially and verifies all original frame hashes.

## Strong native-depth lossless baselines

Use each source's eight frames as one chronological **4×2 atlas**, height 256 and width 512, with the exact same integer pixel values and **identical full metadata header**. Compare these actual decoded formats:
- one **PNG** atlas, mode L for 8-bit and I;16 for 16-bit;
- one **reversible JPEG2000** atlas, exactly preserving 8/16-bit pixels;
- one **GeoTIFF** ZSTD atlas (one grayscale band, tiled blocks, predictor 2, dtype maintained);
- one **FFV1** lossless video in a Matroska container, explicitly native gray8/gray16le, decoded by the same ffmpeg executable;
- the unmodified native-intensity frame buffer with ordinary Zstandard 9.

The FFV1 decode must be from the actual saved Matroska byte stream and verify the native original bit depth and pixel hashes. If a required reference codec is missing/unavailable, CI fails visibly; do not replace it with PNG only or claim unsupported comparisons.

All files include the same exact source manifest: time-point indices, original dtype, original movie/crop SHA256, frame checksums, 128×128×8 geometry, 100ms benchmark duration and standard source names. Count full file bytes, metadata, headers and any codebook. All output compressed archives are actually decoded, not compared on API-estimated byte counts.

## Decision rule and limits

Per source, a temporal encoder wins only if its **complete** file is strictly smaller than the **smallest** valid native-depth PNG/JPEG2000/TIFF/FFV1/ZSTD baseline. A meaningful (still narrow) engineering result requires a win on **both** temporal sources. Mixed 1/2 results or loss on both must be explicitly disclosed, along with the strongest reference.

The key target is **exact native uint16 scientific fidelity** on the nickel sequence, correcting Exp75–76's derived uint8-only benchmark. Source is too small for claims about broader microscopy, science archiving or video codecs. Full speed/memory characterization and external uint16 datasets are required before practical product claims. No change to `main`, public demo or LinkedIn breakthrough announcement based solely on this experiment.
