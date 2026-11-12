# BLKH Exp96–97: frozen Exp95 cross-project transfer protocol

Status: **experiments queued/running, no verified outcome at protocol creation**. This is a retrospective transfer test, NOT a new blinded independent holdout. Never label a successful CI alone as compression superiority.

## Frozen source families

- Exp96: `supertux/supertux`, 29 real original Git PNG parent/child pairs from the existing Exp88 manifest; 14 rock/fallblock, 7 autotile relayout, 8 PNG profile cleanup (the latter is a metadata-only control). All source commits/paths were frozen in Exp88.
- Exp97: `godotengine/godot-demo-projects`, 14 real original Git PNG parent/child pairs from the existing Exp86 manifest; 13 platformer art and 1 Godotea sprite. All source commits/paths were frozen in Exp86.

## Algorithm and fairness

Exp95's selector is reused without retuning: evaluate 289 integer shifts from -8 to +8, select the minimum *individually Zstandard-compressed patch* with deterministic tie-breaking. The archive still uses exact old RGBA WebP plus Zstandard patches and full source metadata. Measure actual total self-contained archive size, not changed-pixel count or patch-only size.

For each group compare:
1. Exp92/93 mismatch-minimizing motion selection;
2. Exp95 compressed-byte-minimizing motion selection;
3. all 14 exact RGBA codecs from Exp84 (including sparse and binary-delta references).

Persist complete archive bytes. Independently re-fetch original historical source PNGs, decode every saved candidate and all 14 references, check exact RGBA including hidden RGB under alpha zero, compare manifest, archive length and SHA-256. Preserve source licensing/attribution; do not commit third-party images.

## Decision rule

Report every group, including failures and metadata-only controls. Aggregate groups overlap the named subsets; do not count them as independent replications. Record wins/losses versus the **best** existing full-archive method, not merely Exp93 or uncompressed PNG. A cross-project positive result would justify a **new preregistered blinded test** only; it would not meet the public demo threshold. A meaningful public result still needs three new independent source families, full source and exact decode audit, >=10% advantage against best fair comparator, reproducible GitHub Actions, and measured CPU/RSS; lossy claims additionally need rate-distortion comparisons at equal error/quality.

## Related artifacts

- Exp95 audited negative report: `BHUH_EXP95_AUDITED_REPORT_AND_EXP96_DECISION.md`
- Exp96: `experiment_96_supertux_transfer.py`, `.github/workflows/exp96.yml`
- Exp97: `experiment_97_godot_transfer.py`, `.github/workflows/exp97.yml`

Until audited results exist, decision is **HOLD**.
