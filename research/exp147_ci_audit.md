# BLKH Exp147: independent GitHub Actions rerun audit

- Verified run: https://github.com/Kronos1027/black-hole/actions/runs/44371314266
- Artifact: https://github.com/Kronos1027/black-hole/actions/runs/44371314266/artifacts/14459221327
- Branch: `research/exp147-bounded-rd-20261118`, commit `9d947371428fd09a63203f8e438b5d7ab492fb10`
- Local machine Python 3.13.5; Actions machine Python 3.13.18.
- All 10 SHA-256 hashes of RGB input match exactly in both environments (4 development, 6 holdout).
- Both environments selected the same configuration using development only: `green_left/brotli`, uniform (not adaptive).
- Both environments: 162 test roundtrips, 6 malformed-input rejects, 540 full-image BLKH roundtrips, 180 baseline encodes + decodes.

## Full lossless bytes (six holdout RGB images combined)
| Codec | Local bytes | Actions bytes | Difference |
|---|---:|---:|---:|
| BLKH | 210,333 | 210,333 | +0 |
| WEBP | 150,116 | 150,116 | +0 |
| JPEGXL | 146,460 | 145,038 | -1,422 |

## Equal-quality interpolation at 40dB RGB PSNR
| Baseline | Compared images | Local BLKH excess | CI BLKH excess |
|---|---:|---:|---:|
| WEBP | 5/6 | +134.92% | +134.92% |
| AVIF | 5/6 | +134.55% | +134.55% |
| JPEGXL | 6/6 | +88.67% | +91.24% |

## Interpretation
WebP and BLKH match exactly in total file bytes, while JPEG XL varies across runner environments despite exact identical RGB inputs. The exact driver/library source of this variation is not isolated; differences in libjxl/ffmpeg stack are plausible but unproven.
Interpolated RD comparison is not a true encoded file at that exact PSNR; coverage varies by codec. Only common supported quality windows are compared. A green workflow does not establish superiority.
No codec advantage was found. External third-party images, native full-resolution images, standardized quality metrics beyond RGB PSNR, codec-version pinning, and isolated memory benchmarks remain missing.

## Provenance
Actual local source blob SHA: `d86528f55913eadd0e4ef503b75f78c24e5856b4`.
Actual local report blob SHA: `ba6441fdaf570d3e45bf97db31cee688a4b677a7`.
Github Actions output was separately downloaded and decoded as ZIP to compare numerical results and source-image hashes (not inferred from green CI alone).
No public demo, LinkedIn publication, or scientific alert is warranted.
