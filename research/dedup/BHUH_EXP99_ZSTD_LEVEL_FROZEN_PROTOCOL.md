# BLKH Exp99 — frozen Zstd level ablation (pre-results)

**Status:** code and GitHub Actions published, result unknown until workflow audit. **Retrospective, not independent; public release HOLD.**

Hypothesis: motion-patch archive competitiveness may depend on entropy-coder settings rather than the shift selector. The same exact frozen 21 Wesnoth old/new PNG pairs and metadata from Exp87/93 are used, without adding source families or changing the motion codec.

For each of four overlapping groups, encode two unchanged selectors (mismatch-minimizing and Exp95 per-pair compressed-byte-minimizing), then re-encode only their Zstandard patch payload at frozen levels **3, 9, 19**. Keep all source metadata, old RGBA WebP atlas, archive header, patch semantics and exact decoder unchanged. Persist **24** full archives. Compare actual complete archive bytes, not just patch sizes, to the strongest historical full-archive reference: all_21_pairs 64,802 B; thea_shadow_fix 43,783 B; dune_firetrooper_cleanup 13,119 B; ant_sprite_distinction 9,916 B.

For each saved archive record byte size, SHA-256, uncompressed patch bytes and patch recompression wall-clock seconds; also record Linux process peak RSS (not a per-codec peak). GitHub Actions independently re-fetches frozen original sources, checks source manifest and every archive's size/hash, independently decodes all 24 streams and verifies full RGBA including RGB beneath zero alpha. The aggregate overlaps its three subgroups and is not an independent replication.

A result only establishes *retrospective compressor-level sensitivity*. Even a win does not support claims of novel general compression, because stronger reference codecs may be retuned at comparable computational budgets and source families are previously seen. Before any public claim: new preregistered independent sources, fair same-fidelity codec tuning and rate-distortion when lossy, encode/decode timing and peak RSS, transparent failures. Do not claim a scientific breakthrough from a green CI.
