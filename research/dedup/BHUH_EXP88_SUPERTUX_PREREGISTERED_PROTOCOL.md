# BLKH Exp88 — PREREGISTERED fifth-source SuperTux authentic PNG Git history

**Immutable source protocol, saved to Git before reading any of these selected PNG image bytes or compressed sizes.** Selection uses Git's commit message, changed file path, `status=modified`, and a fixed path-family prefix only; no prior decoded pixels, similarity metric, output byte count or codec test. Never silently replace missing/invalid files, trim a bad group, retune an existing codec, or call a source-qualification failure an improvement.

## Upstream and source rights

- Independent upstream repository: `supertux/supertux` at https://github.com/SuperTux/supertux (fifth source distinct from Pixelorama, Mindustry, Godot and Wesnoth).
- At first pinned commit root `LICENSE.txt` is **GPLv3**; artwork may contain separately attributable contributors and licenses. Save upstream path/commit/first parent, original Git PNG SHA256, pixel SHA256, original GPL text and available source attribution in the short-lived Actions artifact. **Do not commit upstream art** into BLKH repository.

## Fixed original Git commit groups and exact path list

These are exactly **29 distinct real original Git `modified` PNG filename pairs** (58 source file versions) split into three fixed naturally related source groups from one SuperTux upstream organization, **not three independent author populations**.

### 1. `rock_fallblock_art`: commit `624d2b951320923eaae9a1b1e403d90c7218427e`, 14 PNG pairs

Git message: Add Rock Tileset, Update Rock-related Sprites, and Update Signs. Selection is limited to these original changed paths only, before inspecting pixel values:

- `data/images/objects/fallblock/branchleft.png`
- `data/images/objects/fallblock/branchright.png`
- `data/images/objects/fallblock/cave-2x3.png`
- `data/images/objects/fallblock/cave-3x2.png`
- `data/images/objects/fallblock/cave-3x4.png`
- `data/images/objects/fallblock/cave-4x3.png`
- `data/images/objects/fallblock/cave-4x4.png`
- `data/images/objects/fallblock/cave-4x6.png`
- `data/images/objects/fallblock/cave-5x3.png`
- `data/images/objects/fallblock/cave-5x7.png`
- `data/images/objects/fallblock/cave-6x4.png`
- `data/images/objects/rock/rock-b.png`
- `data/images/objects/rock/rock-c.png`
- `data/images/objects/rock/rock.png`

### 2. `autotile_relayout`: commit `2cdc40783befc7535a3d787a283876e5476d155b`, 7 PNG pairs

Git message: Remodel Many Autotilesets as Autotileset-Corners. Selection is limited to these original changed paths only, before inspecting pixel values:

- `data/images/tiles/forest/forestmerge.png`
- `data/images/tiles/forest/underground/groundmerge.png`
- `data/images/tiles/ghostwood/ghostwood-13.png`
- `data/images/tiles/ghostwood/underground/ghostwood-13_under.png`
- `data/images/tiles/pole/pole.png`
- `data/images/tiles/snow/snowmerge.png`
- `data/images/tiles/snow/underground/snowmerge.png`

### 3. `penny_png_profile_cleanup`: commit `0362b47a9b39a44d1efe7ef05c9f5ac8aa6bd256`, 8 PNG pairs

Git message: Fix a ton of libpng iCCP duplicate warnings. Selection is limited to these original changed paths only, before inspecting pixel values:

- `data/images/creatures/penny/credits-0.png`
- `data/images/creatures/penny/credits-1.png`
- `data/images/creatures/penny/credits-2.png`
- `data/images/creatures/penny/credits-3.png`
- `data/images/creatures/penny/credits-4.png`
- `data/images/creatures/penny/credits-5.png`
- `data/images/creatures/penny/credits-6.png`
- `data/images/creatures/penny/credits-7.png`

**Scope explanation registered ahead of time:** group 1 uses the entire `objects/fallblock/` and `objects/rock/` path families at the pinned edit, excluding unrelated signs, bonus blocks and platforms; group 2 uses the exact `images/tiles/` path family, excluding separate worldmap artwork; group 3 uses the eight consecutive `creatures/penny/credits-0..7.png` frames, excluding other unchanged path families from the same bulk color-profile cleanup commit. Fixed same-domain path-family grouping is a resource-bounded sampling design, **not random sampling**. All selected filenames must be verified as `modified` by Git commit-to-parent status before acceptance.

## Fixed 14-method baseline, no new parameter tuning

The code must reuse **unmodified** Exp83P/Exp84 exact reversible RGBA data paths (including RGB under fully transparent alpha); two experimental exact sparse mask+changed four-channel RGBA codec methods versus **all 12** established references, including bsdiff4 over original historical PNG bytes, PNG/WebP/Zstd exact atlases, XOR/modulo deltas and tile16. Keep the same byte-counting conventions, old image included in all archives (no free side data), full metadata hashes/provenance, tile size, Zstd9 and WebP exact lossless. All stored compressed archives must be independently reloaded and round-trip exact before any comparative verdict.

Run **four cases**: complete 29-pair bundle, `rock_fallblock_art`, `autotile_relayout`, `penny_png_profile_cleanup`. Thus **4×14 = 56 persisted actual archives**. Each must be SHA256 audited by independent workflow code, all original PNG bytes refetched and Git `status=modified` verified. Record bytes, codecs, time, changed-pixel fractions and count of actually changed decoded RGBA PNG pairs. Genuine PNG file re-encoding only (same RGBA) must be retained and disclosed as different *source PNG bytes* versus decoded *pixel identity*. The third group is deliberately a **negative control for metadata-only changes**, not a cherry-picked success case. Keep lossless RGBA vs original exact PNG bytes distinct.

## Frozen decision rule

Claim limited independent-source transfer only if **both** (1) the *full complete 29-pair* experimental sparse archive is strictly smaller than the **smallest of 12 fixed reference full archives**, and (2) at least **two of the three frozen natural groups** also strictly win against their smallest valid reference. Require **≥2** genuinely changed RGBA pairs, every original source accessible and within old fixed geometry/atlas bounds, **all 56 archives exact, and independent source-file audit successful**. Failed source/memory feasibility = **INCONCLUSIVE**, not a win; ordinary compression losses = negative, a single-group/aggregate win without criterion = mixed.

No claim of statistical independence across files from the same project, no universal compression claim, no `main` merge, public demo update, scientific breakthrough advertising, or source selection change following observation. Comparing shared historical provenance metadata is a controlled within-run experiment rather than proof of best production packaging. Independently reproduce on a new project if a meaningful gain is observed.
