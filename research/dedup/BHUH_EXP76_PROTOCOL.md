# BLKH Experiment 76 — frozen temporal lossless byte-delta predictive coding

**Fixed protocol registered prior to Exp76 compression measurements.**

Motivation: Exp75 on two actual scientific image sequences found **0/512** repeated 16×16 RGB tiles in each video. Rather than artificially inserting repeated blocks, test if **reversible temporal pixel differences** allow exploiting near-equality between consecutive *genuine* frames. This is established video compression engineering, not new neural coding or information theory.

## Inputs, exactly as Exp75

Use exactly the same two real temporal datasets (`skimage.data.protein_transport()`, `skimage.data.nickel_solidification()`), original first 8 timepoints and central 128×128 crop, one shared min/max quantization to 8-bit across the selected frames, and replicated 8-bit RGB representation. The original nickel uint16 values are **not preserved by this fixed preprocessing**; encoded frames must nevertheless reconstruct **every pixel of the derived RGB representation exactly**. No shifts, filtering, motion adjustment, color transforms, or source substitution. Reuse the Exp75 metadata with original scientific file SHA256, original crop dtype/extrema and all eight exact RGB frame hashes.

## Four prespecified temporal methods

For each 8-frame RGB sequence define previous(frame0) = an all-zero uint8 image, and previous(frame_t) = the immediately preceding **true decoded source frame**, to avoid error propagation from any lossy approximation:

1. `xor_webp`: bitwise XOR each uint8 RGB channel with previous frame, then encode the resulting **full 4×2 atlas** with WebP lossless.
2. `xor_zlib`: same XOR residual stream concatenated in time, zlib level 9.
3. `mod_webp`: per-channel uint8 modular subtraction (current minus previous, modulo 256), then full 4×2 WebP lossless atlas.
4. `mod_zlib`: same modular temporal residual stream compressed with zlib9.

Use one standalone archive per method: magic/version, complete source metadata/durations, method identifier and compressed bytes. Decode saved compressed bytes, invert predictors frame by frame, verify all eight original derived RGB SHA256, output frame order and timing. No decoder-side preloaded source data and no uncounted state.

## Fair previously validated baselines

Re-run the *same seven* Exp75 already validated lossless methods (exact dictionary+zlib, exact dictionary+WebP, native animated WebP, APNG, full WebP RGB 4×2 atlas, PNG atlas and full-frame zlib), all with the **same source metadata**. Compare every complete serialized file byte count, not raw image payload estimates. Declare best conventional competitor as the smallest of animated WebP, APNG, WebP atlas, PNG atlas or raw full-frame zlib; compare temporal models against that min, and also report the exact dictionary results separately.

**Narrow claim gate:** at least one of the four fixed temporal variants must make a *strictly smaller complete file than the best standard* on **both** genuine temporal sources, preserving all derived RGB bytes. A 1/2 or 0/2 win is mixed/negative. A successful lossless RGB finding would still require direct uint16 scientific compression and stronger FFV1/JPEG2000/video baselines before production/demo claims.

**Release / LinkedIn:** HOLD. Main and demo must remain untouched; these are only two independent time series and a known family of predictors.
