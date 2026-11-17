# Exp122 — row-adaptive palette residuals (2026-11-17)

Status: **negative, exploratory; local-only execution**. Based on Exp116–121. Branch isolated from Exp121.

## Hypotheses

Compare four reversible representations of 16-level grayscale palette indices: raw, horizontal subtraction modulo 256, vertical subtraction modulo 256, and per-row choice among raw/horizontal/vertical using count of nonzero residuals (one selector byte per row). Each with zlib level 9 or LZMA preset 6: 8 candidates. Include full custom container header and row-selector bytes.

## Protocol

96x96 grayscale quantized to 16 levels. Development: five scikit-image samples (astronaut, coffee, chelsea, camera, rocket) and 12 deterministic synthetic samples (random, blocks, stripes, diagonal; seeds 101,102,103). Holdout: five different scikit-image samples (brick, grass, gravel, page, text) and 12 synthetic samples (same families; seeds 9101,9102,9103). Freeze the **single global** configuration minimizing aggregate development bytes, without selecting per holdout image. WebP lossless method 6 on reconstructed 8-bit RGB grayscale images. Custom BLKH container represents indices and requires an externally specified fixed grayscale palette (0,17,...,255); an interoperable format would need to signal this palette or define it in its specification. The 16-level quantization is preprocessing, not lossless compression of original photos.

## Observed local run

- Locked configuration: **raw + LZMA-6**. Row-adaptive mode did not win development selection.
- Holdout 17 samples, 136 BLKH exact roundtrips (8 configurations per image); WebP exact RGB reconstruction also checked for every image. Three malformed inputs rejected.
- BLKH locked total **28,810 bytes** vs WebP **26,974 bytes**, **3/17 wins**. This is a negative aggregate result.
- Natural holdout examples: brick 1426 vs WebP 1402 bytes; grass 3178 vs 2726; gravel 3482 vs 3006; page 2086 vs 2000; text 2030 vs 1914.
- Local source and complete JSON were generated as `/mnt/data/exp122/experiment.py` and `/mnt/data/exp122/results.json`. **They have not been uploaded to GitHub** because this connector cannot directly read local container files. Thus GitHub Actions reproduction is **not established**. The report alone is insufficient for independent replication.

## Interpretation and next actions

Adaptive per-row residual selection with explicit one-byte/row selectors does not overcome conventional WebP on this corpus. Do not tune on the holdout. Transfer exact script and JSON to repository before adding a workflow; verify SHA-256 of each sample, container byte counts, timing and RAM; expand to independent photographic corpora and fair JPEG XL/AVIF lossless and rate-distortion comparisons. No demo/LinkedIn publication warranted. Prior branches and alert automation unchanged.
