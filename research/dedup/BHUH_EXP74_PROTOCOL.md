# BLKH Experiment 74 — exact cross-frame block reuse vs genuine lossless animation

**Preregistered protocol committed BEFORE running any Exp74 measurements.** This is a benchmark of existing block deduplication ideas (content addressing plus WebP lossless dictionary coding), **not neural compression or an invention of new theory**.

## Fixed data construction and scope

Five known real-source 128×128 RGB photographs from the `scikit-image==0.25.2` corpus, exactly the same source/crop choice as Exp73: `astronaut`, `coffee`, `chelsea`, `hubble_deep_field`, `immunohistochemistry`. These source photos **are not external independent held-out images**. For each source, construct an 8-frame 128×128 RGB sequence from the real photograph's intact central crop; each new frame copies the previous image and replaces exactly **0, 2, 8 or 32 different 16×16 tiles** with original 16×16 pixel patches taken from the *next* real source photo (cyclic donor). Source pixels are real but the changes are **deterministically simulated edits**, not actual natural video or observed animation. Use NumPy Generator seed 740074 + index of source ×100 + edit-count ×10 + frame number; no additional noise; tile selection without replacement, donor tile selection at random independent uniform position in donor crop. This creates 20 **correlated** scenarios (5 sources ×4 edit rates), each containing eight 128×128 RGB frames. Frame duration is precisely 100ms. Frame origins, edit-rate, donor, complete pixel SHA256 and provenance are byte-counted.

## Experimental codec (two variants)

1. Exact global SHA256-indexed **16×16 RGB tile dictionary** across all eight frames. Serialize dictionary tiles in *first occurrence* order, and a complete 8×64 array of uint16 tile IDs, so all 512 tile references and every unique tile are counted. **No clipping, quantization, lossy pixel storage or uncounted metadata.**
2. Variant `raw_zlib`: concatenate unique tiles and compress with zlib9.
3. Variant `webp_dictionary`: pack unique RGB tiles into a 16-column WebP lossless atlas (unused cell slots explicitly zeroed); all dictionary atlas pixels, tile IDs, headers and manifest are serialized in the file. Reconstruct frames only by decoding the saved dictionary pixels and the saved uint16 indices.
4. Include a complete header and manifest in each file and verify SHA256 of every reconstructed full RGB frame, preserving all eight frame indices and their 100ms durations.

## Fair conventional reference schemes

For each 8-frame RGB sequence, save complete archive versions with identical JSON source manifest:
- **animated WebP lossless**, with frame durations and frame order verified after decoding. It may coalesce identical consecutive frames if their total duration can be reconstructed exactly.
- **animated PNG/APNG lossless**, likewise fully decoded and verified, when supported in the pinned Pillow version; failure is a visible benchmark failure, not a silent omission.
- **one lossless WebP atlas** with the 8 RGB frames arranged in **4 columns × 2 rows**, with the exact frame ordering and duration manifest.
- **one optimized PNG atlas** using the same arrangement.
- A simple full-frame zlib9 stream is an additional control, counted with its common manifest, not a straw-man win criterion.

Do not compare the custom dictionary only against eight isolated PNG images, or neglect animation standards. Compare complete file sizes inclusive of identical ordered manifest for all arms, and full decoded equality for all 160 reconstructed RGB frames. The winner for each scenario is the minimum complete byte count across valid references, not a convenient weak baseline.

## Declared evaluation

Report all 20 per-scenario byte counts, number of unique tiles, number of references, exact source/output SHA256, average encode/decode time measured in the same runner (exploratory only), and winning arm. Analyze each edit rate separately, not just favorable 0/2-tile results. An affirmative **limited engineering** finding would require complete custom archive smaller than the strongest valid standard for at least 4/5 sources **at the same edit rate**, and must explicitly state the edit rate and artificial edit construction. Even a narrow win must not be generalized to video codecs, natural animations, production decompression speed or global compression superiority.

**Demo/LinkedIn/production:** HOLD regardless of Exp74 alone; these data are simulated frame edits on known photos and require external representative workloads and independent codec evaluation.
