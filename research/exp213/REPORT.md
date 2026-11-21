# Exp213 — adaptive per-tile predictor, negative result (2026-11-21)

Compared 4 causal reversible predictors (none, left, up, clipped gradient), using actual zlib9 compressed sizes for selection on each 16, 32, or 96 pixel tile. Fixed predictor baseline is none. All 48 configurations/images round-tripped RGB pixels exactly and truncated streams were rejected. 8 scikit-image bundled samples, resized to 96x96 RGB; first four development, last four held out. Chosen on development: tile=96, adaptive=true.

Held-out complete container bytes: BLKH prototype 37,136; Pillow WebP lossless method=6 24,608; BLKH +50.91% larger. No superiority. Pillow 12.3.0, Python 3.13.5. No independent external photographic holdout, AVIF/JXL, isolated RAM or RD curve. Not suitable for demo or LinkedIn.

Local runnable script and results JSON were generated for this run but are not included in this commit. Next: add scripts, dependency pinning, CI, independent dataset, then isolate memory and compare AVIF/JXL. Previous Exp211 row selection and Exp212 channel selection differ from this experiment: this selects *per tile by compressed byte count* rather than per-row proxy or channel transformation. The overlap in causal predictors is acknowledged.
