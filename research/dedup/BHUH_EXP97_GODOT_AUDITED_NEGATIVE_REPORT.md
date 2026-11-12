# BLKH Exp97 — audited Godot cross-project transfer: negative

**Decision: HOLD.** No fair codec win. Retrospective project source previously used in Exp86, not a new blinded independent test.

**Implementation:** [5c0a328](https://github.com/Kronos1027/black-hole/commit/5c0a32828acd72dabf0f01b52d57f0cac92adf5e).
**Auditor repair:** [3119420](https://github.com/Kronos1027/black-hole/commit/311942044567e011f5eb27d4fc66c3563893b8af).
**Verified run:** [43330431199](https://github.com/Kronos1027/black-hole/actions/runs/43330431199) (success).
**Full artifact:** [13993429805](https://github.com/Kronos1027/black-hole/actions/runs/43330431199/artifacts/13993429805).

Frozen original 14 Godot Git parent/child PNG pairs (13 platformer, 1 Godotea). Two exact motion methods + 14 full-archive exact RGBA reference codecs for each of three overlapping groups = **48 fully persisted and independently audited archives**. Original source bytes are not the guarantee; pixelwise exact RGBA, including RGB under alpha zero, is.

| Group | Exp93 mismatch-min | Exp95 compressed-byte selector | Best conventional full-archive reference | Penalty vs best |
|---|---:|---:|---:|---:|
| all_14_pairs | 335,017 B | 335,102 B | 291,571 B | +43,531 B (+14.93%) |
| platformer_art_redraw | 81,068 B | 80,737 B | 69,399 B | +11,338 B (+16.34%) |
| godotea_sprite_revision | 221,995 B | 221,594 B | 170,663 B | +50,931 B (+29.84%) |

The 14-pair aggregate overlaps its two component groups, and the two groups share one source repository. Exp95's per-pair compressed-size selector is not guaranteed to minimize the whole archive, and in the aggregate it loses 85 B even to the mismatch selector.

The initial [43282376455](https://github.com/Kronos1027/black-hole/actions/runs/43282376455) run had a **workflow audit coding error** (`KeyError: 'best_reference_bytes'`): its result row did not contain that key. This was repaired to compare the recorded `best_existing_bytes` against the minimum of the two recorded baseline categories, and the new run passed full independent persisted archive verification. No data were altered to make the result positive.

**Limitations:** previously seen source, no measured CPU/RSS, no newly blinded independent sources, no fair lossy rate-distortion comparison. No release or professional superiority announcement. The negative result supports moving away from per-pair local motion scoring toward joint compressed-stream selection, subject to independent follow-up.
