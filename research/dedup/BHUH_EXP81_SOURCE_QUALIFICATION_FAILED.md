# BLKH Exp81 — source qualification failed, no compression inference

**Status: source audit completed, preregistered sample requirement NOT MET. No compression benchmark was executed.** This is a transparent **source-qualification failure**, neither a codec win nor a codec loss. Do not claim 0/N performance.

- Preregistered protocol: [BHUH_EXP81_REAL_GIT_REVISIONS_PROTOCOL.md](BHUH_EXP81_REAL_GIT_REVISIONS_PROTOCOL.md).
- Official Actions source-audit run: https://github.com/Kronos1027/black-hole/actions/runs/40784190616 (CI **failed as intended** because the fixed source selection found zero eligible edited sprite pairs).
- Source audit artifact: `exp81-original-git-png-revision-differences`, includes `source_inventory.json` with all 12 immutable Git commit SHA256s, 11 consecutive parent-child diff checks, eligibility filters and exact rejection reason.
- Source project: https://github.com/pixel-boy/NinjaAdventure, creator Pixel-Boy/AAA, public CC0 sprite release documented via https://tmhsdigital.github.io/Free-Game-Dev-Assets/entry/ninja-adventure/.
- Pinned Git source HEAD: `6ac78232d5aedcc85ce5f27d060ea92366f7c24a`.
- **Inventory:** 11 historical commit transitions investigated; `git diff --diff-filter=M` yielded **one** same-path modified PNG (`system/environment/fx/fog.png`) but it fell outside the preregistered approved sprite paths (`content/character/`, `content/ui/`, `content/destroyable/`, `content/particle/`, `content/map/`). Other PNG source changes were primarily additions/renames and cannot be used as if they were before-and-after versions of the same source path. Qualified original RGBA edited sprite pairs = **zero**. This result is not an algorithm measurement.
- **CI diagnostics:** an initial decoder self-test found that several PNG/WebP decoders tolerate a single truncated trailing byte. The experimental `.bl81` format was corrected to count an equal SHA256 payload checksum in every arm, then it passed the random RGBA and truncation self-test. The source audit subsequently failed for the preregistered zero-eligible-source condition; no compressed result files exist.
- The user research branch may safely retain this prototype and honest failed Actions run for auditing. **Do not silently widen prefixes, change source commits or invent edits within Exp81** in response to source inspection. An alternative corpus must be a separately preregistered next experiment and must demonstrate its genuine historical PNG edits before any compression claim.

**Release:** no merge, no change to main or demo, no breakthrough claim. This failed qualification provides no evidence of size advantage or disadvantage against WebP/ZSTD for actual revisions.
