# Exp232 fixed predictor ablation (2026-11-22)

Python experiment (local script: /mnt/data/exp232/bench.py; JSON: /mnt/data/exp232/results.json). Six scikit-image RGB 96x96 images; green-difference reversible channel transform; predictors none/left/up/average/MED, LZMA preset 6, 9-byte experimental header. All 30 experimental streams decoded and reconstructed exactly, plus 12 PNG/WebP roundtrips.

| Split | none | left | up | avg | MED | PNG | WebP lossless |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev |60551|53795|50239|51407|48583|57022|42662|
| held-out |50995|48779|45851|49827|45959|52362|40904|

Development-selected MED is 12.36% larger than WebP on held-out data (45959 vs 40904 bytes). Selecting UP post hoc on held-out would yield 45851, but this is NOT a valid generalization claim. Compared with Exp231 row-adaptive 45103 bytes, fixed MED is larger by 856 bytes; however codec framing, Pillow versions and run differences should be audited before asserting direct equivalence. PNG/WebP decoded exactly. This is an ablation, not a new competitive codec.

Limitations: local source and JSON not yet in repository, no independent external dataset, no isolated peak RAM, no AVIF/JXL, no Actions reproduction. Do not publish demo or LinkedIn. Next: commit script and deterministic data manifest, build CI, validate third-party corpus, and fair codec comparisons.
