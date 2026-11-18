# Exp134 — reversible RGB bitplane transposition (2026-11-18)

Status: **negative; exploratory local execution, not CI-reproduced**.

Hypothesis: global bitplane transposition might improve LZMA entropy coding after reversible green-relative RGB decorrelation and optional vertical delta. Four candidates: base, up, basebits, upbits. LZMA preset 6. Full 13-byte custom header included in every BLKH size. Bitplane transposition is invertible, tested by encode/decode exact RGB equality. WebP lossless Pillow method=6 decode also checked.

Development: astronaut, coffee, chelsea, rocket. Holdout: camera, coins, moon, page. Original scikit-image dimensions, RGB conversion. These samples are **not external unseen photographs**; their identities have been explored in prior experiments.

| image | WebP bytes | base | up | basebits | upbits |
|---|---:|---:|---:|---:|---:|
| astronaut | 335224 | 432713 | 388417 | 454085 | 402877 |
| coffee | 336534 | 438797 | 403093 | 467509 | 410569 |
| chelsea | 153732 | 198873 | 176477 | 248817 | 216913 |
| rocket | 247300 | 283513 | 277313 | 325761 | 285193 |
| camera | 125522 | 142761 | 137653 | 158749 | 143397 |
| coins | 67428 | 81397 | 74229 | 88421 | 77969 |
| moon | 43012 | 41201 | 38645 | 41785 | 35829 |
| page | 39078 | 45165 | 40873 | 48097 | 42917 |

Selection by development bytes only: **up**. Holdout: BLKH **291400** bytes, WebP **275040** bytes; BLKH **5.95% larger**. Four exact BLKH decodes in holdout; 32 exact BLKH decodes overall. The bitplane variants failed to improve aggregate compression. Single moon image improves versus WebP, insufficient to claim general superiority.

Local artifact paths at execution: `/mnt/data/exp134/experiment.py`, `/mnt/data/exp134/results.json`. The experiment source and JSON are **not yet committed** because the local runtime cannot resolve github.com; this report was added through the connected GitHub API. Do not interpret this report commit as CI reproduction. No AVIF/JXL, no rate-distortion curve, no isolated RSS, no external independent corpus. Next run: upload source/JSON via connector, add workflow, rerun and expand datasets. No demo/LinkedIn publication.
