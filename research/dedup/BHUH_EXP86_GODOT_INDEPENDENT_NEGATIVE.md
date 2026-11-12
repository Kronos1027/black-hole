# BLKH Experiment 86 — independently sourced Godot original PNG art edits: NEGATIVE

**Status: official GitHub Actions SUCCESS** on the preregistered Godot third-repository experiment, including its independent verification of **42 fully persisted archives** and all **14 original old/new PNG file pairs** (28 images). Compression hypothesis **FAILED** in the aggregate and both frozen groups.

- Frozen source/data selection before reading original image bytes or evaluating compression: [BHUH_EXP86_GODOT_SOURCE_HOLDOUT_PROTOCOL.md](BHUH_EXP86_GODOT_SOURCE_HOLDOUT_PROTOCOL.md), recorded in commit `fd84fc27fcc8002622ebc07cd5d2292408e189cf`.
- Source loader and experiment: [experiment_86_godot_png_holdout.py](experiment_86_godot_png_holdout.py).
- Successful official CI: https://github.com/Kronos1027/black-hole/actions/runs/40930445363 (head `48caff25e4c9aca6f4889b20c67e3f88bed6bbe2`).
- Source upstream: https://github.com/godotengine/godot-demo-projects, original project-root MIT license with original Git file paths, first-parent SHA, original PNG hashes and attribution in the CI archive. Root licensing does not necessarily override separate third-party art ownership. Source sprite bytes are retained in CI artifacts, not committed to this repository.
- Godot upstream includes a **merge commit** in the first frozen group; its Git *first parent*, rather than assuming one-parent history, is explicitly selected by the source loader and checked by the independent audit.

## Locked comparison

The **same two experimental sparse lossless patch formats** from Exp84/85 are compared to **all 12 unchanged, exact lossless reference formats**, including pixelwise XOR/modulo references, complete PNG/WebP/Zstandard atlases, originals, tile16 and PNG-byte bsdiff4. Every resulting file includes complete original old images and full source metadata. Actual persisted bytes were reopened, SHA256 verified and decoded to the exact original uint8 RGBA, *including RGB beneath alpha zero*. The binary PNG diff method additionally verifies the original source PNG file-byte SHA256. The decoder is not granted free copies of old images.

| Original Godot Git source group | Actual changed target RGBA pixels | Best sparse codec | Sparse bytes | Best of 12 references | Reference bytes | Experimental overhead |
|---|---:|---|---:|---|---:|---:|
| **Aggregate 14 original PNG edits** | **59.0694%** | `sparse_webpold_zstd` | **336,813** | `atlas_webp` | **291,571** | **+45,242 B (+15.517%)** |
| Platformer art redraw, 13 edits | 77.3810% | `sparse_webpold_zstd` | 81,677 | `atlas_webp` | **69,399** | +12,278 B (+17.692%) |
| Godotea sprite revision, 1 edit | 42.1459% | `sparse_webpold_zstd` | 222,884 | `atlas_webp` | **170,663** | +52,221 B (+30.599%) |

The 13 platformer originals are genuinely RGBA changed, and the one Godotea original is also genuinely changed; all **14/14 source pairs** pass source-qualification. The single-file Godotea group is **not** a statistically independent population or a separate upstream project.

**Prespecified transfer gate:** (i) aggregate strict win against *all 12* competing formats, (ii) wins in both fixed source groups, and (iii) at least two genuinely RGBA-changed source files. Observed source-qualified but **0/3 wins; gate FAILS**. The true overall loss is **45,242 bytes**, not a win. The percentages of changed pixels are **diagnostic**, not post hoc source selection controls.

## Engineering incident corrected without compression parameter tuning

The preliminary workflows failed inside `experiment_84_sparse_pixel_patches.py::old_atlas_decode` because its safety condition interpreted a `3_000_000` **pixel** limit as a `3_000_000` **RGBA-element** limit (four channels for every pixel). The common original atlas **encoder** and the other legacy decoder checked at most **3 million pixels**, so exact, valid larger originals were incorrectly rejected.

The guard was corrected on the research branch in commit `48caff25e4c9aca6f4889b20c67e3f88bed6bbe2`, before the successful CI run, to require positive dimensions, four channels and `height*width <= 3_000_000`. It **does not change** atlas encoding, compression algorithms, codec parameters, wire layout, reconstruction rules, frozen original source selection or competitive method set. All 42 files were successfully audited from disk after the fix. Past premature failed runs are setup errors, **not** successful compression comparisons. The original sparse source benchmark requires a complete 42-file audit; that criterion was met.

## Scientific interpretation and constraints

- On these real edits, **42–77%** of target pixels differ from their parent versions. Sparse per-changed-pixel replacement is not competitive with a conventional complete lossless WebP atlas when many pixels changed. This is plausible and consistent with the measured negative outcome; it is **not** a proof of a global threshold or mathematical scaling law.
- This test measures Godot authoring history for **one additional independent upstream repository**, not 14 independent authors, not random images, and not independent overall artistic populations. It intentionally stresses substantial source art redraws instead of post hoc targeting source patches with <1% edited pixels.
- Exp84 showed no win on 19 Pixelorama historical PNG changes. Exp85 on 30 Mindustry historical PNG changes showed one small 429-byte win in a narrowly defined 15-pair source group, but an aggregate 35-byte loss and failure of its preregistered gate. **Exp86 adds another independently sourced aggregate loss.** No replicated independent **whole-corpus** win.
- All 14 methods use equal *complete* shared provenance metadata. Some manifest fields are redundant for certain methods, so this is a **controlled within-run** fixed packaging comparison, not a guarantee that the deployed codecs' respective optimal custom formats would have the same absolute sizes.
- Only archive byte-size and exact reversible reconstruction are audited. Source-independent end-to-end encode/decode throughput, memory behavior, Git packfiles and other production baselines remain open. The experiment provides **no** general-purpose algorithmic or neural breakthrough evidence.

**Release decision:** Maintain PR #1 in draft; leave `main`, public demo and breakthrough-marketing claims unchanged. Continue only with an independently prespecified use case using small-area authentic edits or a stronger rigorous codec baseline. Do **not** inspect the Godot outcomes and then revise source paths or parameters to manufacture a result.
