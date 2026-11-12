# BLKH Exp95 — audited retrospective compressed-byte shift selection

**Status:** technically reproducible, scientifically exploratory, **release HOLD**.
**Source:** frozen historical 21 original Wesnoth Git PNG edit pairs from Exp87.
**Implementation:** [060904c](https://github.com/Kronos1027/black-hole/commit/060904c22a2cc93723a0e2294e4e2193afde982b).
**Verified run:** [43281916571](https://github.com/Kronos1027/black-hole/actions/runs/43281916571) (success, independent persisted archive audit).
**Artifact:** [13973506092](https://github.com/Kronos1027/black-hole/actions/runs/43281916571/artifacts/13973506092).

## Frozen methods and complete actual archive sizes

| Group | Exp93 pixel-mismatch selector | Exp95 per-pair zstd-compressed patch selector | Savings vs Exp93 | Best historical competing full archive | Exp95 minus competitor |
|---|---:|---:|---:|---:|---:|
| all_21_pairs | 68,051 B | 64,810 B | 3,241 B (4.76%) | 64,802 B | **+8 B** (loss) |
| thea_shadow_fix | 47,012 B | 43,795 B | 3,217 B (6.84%) | 43,783 B | **+12 B** (loss) |
| dune_firetrooper_cleanup | 13,305 B | 13,305 B | 0 B | 13,119 B | **+186 B** (loss) |
| ant_sprite_distinction | 9,949 B | 9,949 B | 0 B | 9,916 B | **+33 B** (loss) |

The three named subgroups overlap the all_21_pairs aggregate: **not four independent source families**.
The selector minimizes *per-pair* Zstandard compressed patch size; the actual final archive is compressed jointly, so local optimality is **not** a global compressed-stream optimum.

## Reproducibility and integrity

The GitHub Actions run installed Python 3.11, NumPy, Pillow, zstandard, and bsdiff4; fetched the frozen original source images, saved 8 complete motion archives, then **separately re-fetched originals and decoded each saved archive**. All persisted SHA-256 checks, full sizes, metadata checks, and exact RGBA comparisons passed. The report concerns RGBA pixels; it **does not preserve the original PNG file bytes** and is not a lossy rate-distortion benchmark.

## Limitations / decision

The new selector substantially improves the motion candidate compared with Exp93, but does **not** beat the strongest already measured competitor in any group. The test is retrospective and uses previously inspected images; there is no fresh blinded source family, no measured cross-project latency/RSS threshold, and no fair lossy quality-matched JPEG/WebP/AVIF RD evidence. No public breakthrough claim. Do not merge into main or announce compression superiority.

## Next controlled phase

Exp96 freezes the Exp95 algorithm without retuning and applies it to the pre-existing SuperTux Exp88 original Git PNG corpus (29 old/new pairs, three genuine revision groups, including a PNG metadata-only control), comparing **all 14** existing complete RGBA reference archives. This is cross-project transfer, **not a newly blinded independent holdout**, because SuperTux had already been used by prior research. Follow with new preregistered independent sources and measured encode/decode CPU/RSS if transfer is promising.
