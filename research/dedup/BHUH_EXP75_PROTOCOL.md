# BLKH Experiment 75 — native temporal scientific video, strict domain transfer

**Pre-registration before loading or inspecting Experiment 75 time-series pixel data.**

Exp74's 15/20 positive result used real photographic patches but synthetic exact-duplicate edits from a fixed donor pool. This independent *domain transfer* now tests **real temporal acquisitions** with no artificial tile replacement, interpolation, translations, edge smoothing, thresholding, background removal or motion alignment. The dictionary algorithm, exact 16×16 tile size, all header/ref storage and lossless WebP/APNG/atlas comparators are **unchanged** from verified Exp74.

## Independent real temporal acquisitions

- `protein_transport`: scikit-image `data.protein_transport()` (15 time points, 2 channels, 180×183; real fluorescence-microscopy transport), **first 8 real time points**, **channel 0**, spatial **central 128×128 integer crop**. Cite source sample provenance and scikit-image version.
- `nickel_solidification`: scikit-image `data.nickel_solidification()` (11 time points, 384×512, uint16 X-ray radiographs), **first 8 real time points**, spatial **central 128×128 integer crop**. Cite source provenance and skimage version.

These are exactly **two independent original time-series sources**, not 16 independent movies. They are NOT held out examples of sprite editing or human videos.

## Fixed 8-bit RGB benchmark representation

BLKH Exp74 and animated WebP/APNG target uint8 RGB, whereas scientific microscopy sources may be uint16. **Do not call this pipeline lossless relative to the original uint16 scientific files.** Convert each source's first 8 cropped single-channel frames to uint8 by **one shared min/max scalar normalization over all 8×128×128 cropped pixel values**, using `round(255*(value-global_min)/(global_max-global_min))`, then repeat the SAME uint8 channel three times into RGB. Record original sequence data type, selected full array original SHA256, uncropped movie shape, original exact crop hash/dtype, normalization extrema, derived uint8 RGB eight-frame SHA256 and per-frame hashes in the manifest. This fixed preprocessing is lossy for uint16 sources; **codec evaluation after the defined conversion must be strictly lossless**.

The compression benchmark is therefore *an exact-RGB representation benchmark on true temporal source content*, not native scientific bit-depth preservation or scientifically archival lossless image compression.

## Fixed comparators and acceptance

Reuse the exact seven Exp74 algorithms, no retuning: 16×16 SHA256 dictionary + zlib9, 16×16 dictionary packed as WebP lossless, native lossless animated WebP, native APNG, whole 4×2 atlas WebP/PNG lossless, all-frame zlib9. Each gets the same source/crop/frame-count/100ms-duration manifest. Independently decode and SHA256-check all **14 serialized files**, preserving original temporal ordering and 8 sampled frames. Count all metadata. Report unique tile count and reuse rate per source.

**Success gate (narrow):** an Exp74 custom dictionary must be smaller than the smallest valid conventional comparator on **both** actual temporal sequences. If only one wins, report mixed. If neither wins, explicitly retract generalization of Exp74's synthetic-edit improvement to real temporal microscopy, without changing the valid synthetic Exp74 finding.

**No demo, main branch or LinkedIn breakthrough announcement** from Exp75 alone. GIF indexed/low-color conversion is intentionally excluded because the benchmark is lossless full 8-bit RGB. For truly lossless uint16 archival data, additional TIFF/JPEG2000 scientific-video baselines would be needed.
