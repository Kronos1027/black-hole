# BLKH Experiment 92 — PRE-REGISTERED temporal motion compensated exact RGBA patches on sixth independent Git upstream

**Immutable study design frozen in Git before downloading selected original PNG image bytes, observing RGBA values, or running any codec.** Experimental method (±8-pixel translation search) and exact source list fixed below. Source selection based only on public upstream Git commit messages/status=modified path names. Both empty/identical and substantial changed pairs must remain; source availability or software bounds failure ⇒ INCONCLUSIVE, not sample replacement or post-hoc selected sources.

## Sixth completely separate project: Endless Sky

- Upstream https://github.com/endless-sky/endless-sky ; `endless-sky/endless-sky` (not Pixelorama, Mindustry, Godot, Wesnoth, SuperTux).
- Root `license.txt` contains GPLv3 project license; upstream `copyright` identifies `images/*` as **CC BY-SA 4.0** subject to more specific exceptions; preserve `license.txt`, `copyright`, credits where available and original Git source path/commit/first-parent SHA and per-file PNG checksum in CI artifacts; don't upload upstream art to BLKH repository, and don't infer all individual image licenses universally.
- Exactly **11 original Git status=modified PNG image revision pairs (22 PNG files)** from 3 original commits, no selection by image decoded bytes, pixel fraction or achieved compression size:
### Group 1 — `astral_crop_first8` (commit `6985a40773ede0f175805fde399e8cc3fea1e5e7`, 8 pairs)
Original Git change topic: crop ship frames, removing empty pixels. Take precisely the first eight zero-indexed frame paths (00–07), deterministic capacity-bounded prefix; do not later extend/truncate based on outcomes.
- `images/ship/astral cetacean/astral-00.png`
- `images/ship/astral cetacean/astral-01.png`
- `images/ship/astral cetacean/astral-02.png`
- `images/ship/astral cetacean/astral-03.png`
- `images/ship/astral cetacean/astral-04.png`
- `images/ship/astral cetacean/astral-05.png`
- `images/ship/astral cetacean/astral-06.png`
- `images/ship/astral cetacean/astral-07.png`

### Group 2 — `horizon_effect_fix` (commit `57051117483c502f389240060b20c1a83167ae91`, 2 pairs)
Original Git change topic: small correction to Horizon Grappler effect artwork. 
- `images/effect/horizon grappler+0.png`
- `images/effect/horizon grappler+1.png`

### Group 3 — `vesvi_planet_tweak` (commit `b85cc0a066639ebe78a64d25f2b27b427ecb637f`, 1 pairs)
Original Git change topic: adjust Vesvi planet artwork. 
- `images/planet/vesvi.png`


These natural Git commit groups are **not independent projects/artist populations**; 8 related animation frames may be highly correlated. Real PNG file byte `status=modified` does not guarantee decoded RGBA pixels change; count/retain identical pixel pairs honestly. Old/new array shape may change due to cropping and this is expected.

## New algorithm frozen BEFORE source pixel read

**`temporal_shift8_webpold_zstd`:**
1. Same full shared historical source manifest and bounds as Exp84's `sparse_webpold_zstd`; same ordered old sources, 16-pixel atlas `pack`, exact WebP lossless old atlas with `exact=True`, same Zstandard level 9 for delta. No decoder-free old pixels or source PNG bytes. Same original source filenames/Git hashes and archive metadata overhead as all prior 14 methods (12 references + 2 sparse).
2. For each pair old RGBA and new RGBA, independently evaluate all integer translations **dx,dy in inclusive [−8,+8]**, including **(0,0)**. Candidate reference at target coordinate (x,y) is `old[y+dy,x+dx]` when in bounds, otherwise **exact (0,0,0,0)**. Use all four RGBA channels including invisible RGB to find exact pixel equality. Choose the displacement giving the **smallest number of unequal target pixels**, not entropy or compressed-size optimization; deterministic ties use ascending `(|dx|+|dy|, |dy|, |dx|, dy, dx)` so `(0,0)` wins ties.
3. For each pair, append `<bbI>` (signed int8 dx, signed int8 dy, uint32 unequal pixel count), then `ceil(target_pixel_count/8)` byte little-bit-order exact change mask and exact **4 RGBA bytes for each changed target pixel**, row major. Compress concatenated pair records via **Zstandard level 9**, no lossy residual quantization. Full archive = `<5sBI>` (`BL92M`, version 1, exact full metadata byte count) + entire canonical JSON `{**shared_source_manifest,codec:"temporal_shift8_webpold_zstd"}` + `<II>` baseline-old-WebP byte length and delta-Zstd byte length + both actual streams. This intentionally matches the old encoders' verbose provenance, sizes and two-stream framing to isolate shifted delta coding effect.
4. Decoder validates full structure, decompressed target count and bit padding, signed translations in range, old atlas checksum, per-image original RGBA SHA256, reconstructed total atlas SHA256 and exact original image pair ordering, consuming every byte. Decode to original **exact uint8 RGBA**, including hidden RGB beneath zero alpha. It does **not** promise to restore the original PNG byte encoding or ancillary metadata.
5. No hyperparameter tuning after holdout; any discovered bug can be corrected, while retaining same shifts, data, mode, stronger baselines and byte-counting. Archive corrupt/truncated tests, specifically shifts, mask, wrong checksum, and source size changes. Freeze bounds: old decoder 3-million-pixel atlas, each PNG max 1024×1024 and max 5MB original source bytes.

## Complete benchmark and independent source audit

4 cases: 11-pair aggregate and exact three named commit subgroups. **15** complete methods, including the **unchanged original 14** (2 sparse, 12 fixed full lossless references) + one novel temporal-shift candidate. **60 persisted actual complete archive streams**; independently re-open, SHA256 check, decode full archived original RGBA, verify original PNG Git source and status=modified and full canonical manifest. All methods retain their actual full image-containing old data and identical rich historical shared metadata. Report saved full file bytes, modes, codec time and pixel mismatch fraction, number of nonzero selected shifts, actual changed PNG pairs, per-group and aggregate reference winners, and outcome of original 14-method sparse comparator.

**Frozen scientific win criteria:** positive new hypothesis on sixth project only if (i) entire 11-pair temporal-shift archive strictly smaller than *all 12 original reference* complete archives AND strictly smaller than both prior sparse archives, (ii) temporal-shift strictly wins against smallest of 12 references in >=2 of three original commit groups, (iii) >=2 selected PNG pairs genuinely changed decoded RGBA, all 60 complete saved archives decode bit-perfect and independently refetched original source PNGs/status verified. A partial/isolated win = mixed, aggregate failure = negative, source or metadata/geometry qualification failure = **INCONCLUSIVE**. Even a passed test is one independent project, not a world record or proof of general superiority.

**Release gate: HOLD** no `main` merge, public demo update or public scientific announcement. First disclose all signed positive/negative results and performance tradeoffs, then replicate elsewhere if promising. This experiment compares RGBA-exact, **not original-PNG-byte-exact** tier. The latter has separate Exp91 Zstandard/Git PACK baselines.
