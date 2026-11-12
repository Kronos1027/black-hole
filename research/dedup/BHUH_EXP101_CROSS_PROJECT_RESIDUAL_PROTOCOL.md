# BLKH Exp101 — frozen dual-project transfer of Exp100 residual representation

**Preregistered before Exp100 has any validated outcomes.** This stress test applies the six fixed Exp100 variants on **two previously studied** real Git PNG histories, without tuning on outcomes. It is not a newly blinded independent holdout. **Public release HOLD**.

## Fixed original source families

- SuperTux `supertux/supertux`, historical 29 PNG parent/child pairs frozen in Exp88, groups rock/fallblock (14), autotile (7), metadata-only profile cleanup control (8), plus the overlapping all-29 aggregate.
- Godot `godotengine/godot-demo-projects`, historical 14 PNG parent/child pairs frozen in Exp86, groups platformer redraw (13), Godotea revision (1), plus the overlapping all-14 aggregate.

Every original upstream path, child SHA and first-parent SHA remains fixed in its earlier Exp88/86 protocol. No pixel-driven source selection.

## Six frozen reversible methods and all baselines

Reuse Exp100 encoder/decoder and the same two existing ±8 integer-motion selection algorithms, each with three fixed residual representations: direct changed RGBA, XOR relative to predicted old, and modulo-256 subtraction relative to predicted old. No change to source WebP lossless atlas, source metadata, Zstandard compression settings or encoding methods between projects. Persist all six full candidate archives plus **all 14 same-fidelity codecs** from Exp84 for each of seven overlapping groups: 7 × 20 = **140 full archived outputs**.

## Integrity and limits

A separate workflow step refetches original historical Git source PNGs and verifies checksums, exact RGBA (including hidden RGB under transparent alpha), identical source manifest, SHA256 and sizes for all 140 persisted archives. Check upstream attribution and root project license; do not commit third-party PNGs. Report every win/loss, use the minimum whole-archive size among all 14 historical baselines.

No scientific replication claim from aggregate and subgroup overlap. Source projects are distinct from Wesnoth but were **already used in older experiments**; any success is exploratory cross-project transfer, requiring genuinely unseen preregistered sources. No lossy rate-distortion superiority is claimed, and CPU/RSS budgets must be evaluated before deployment. Do not merge to main or claim public breakthrough on green CI alone.
