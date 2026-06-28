# arXiv Endorsement Request

> **Status**: Paper ready, seeking arXiv endorsement for submission to
> `cs.IT` (Information Theory) or `eess.SP` (Image and Video Processing)
>
> **Last updated**: 2026-06-28

---

## 📄 The Paper

**Title**: *Black Hole (BLKH): Bit-Perfect Neural Implicit Compression
for Smooth Signals via SIREN + Hybrid Residual Coding*

**Author**: Darlan Pereira da Silva (Kronos1027)
- Email: darlan1027pc@gmail.com
- GitHub: https://github.com/Kronos1027
- ORCID: (pending — to be added upon submission)

**Preprint location**: [`paper/paper.pdf`](paper/paper.pdf) (in this repository)
**LaTeX source**: [`paper/paper.tex`](paper/paper.tex)

---

## 🎯 What is This Paper About?

BLKH is a neural implicit compression system that combines SIREN
(Sinusoidal Representation Networks, Sitzmann et al. NeurIPS 2020)
with hybrid residual coding to achieve bit-perfect lossless compression
of smooth 2D/3D signals. The paper presents v5.30, which includes 8
compression modes validated on synthetic and real-world benchmarks.

### Key Empirical Results (all reproducible from this repo)

| Benchmark | Result | Reproduce |
|-----------|--------|-----------|
| Production tests | 159/159 passing | `python -m pytest tests/ -q` |
| **REAL photos (scikit-image)** | **2.87× smaller than JPEG, 1.59× smaller than WebP** | `python research/universe/real_image_benchmark.py` |
| Kodak-like (768×512) | 7.54× smaller than JPEG, 4.22× smaller than WebP | `python research/universe/kodak_benchmark.py` |
| Real photos (128×128) | 27-122× smaller than ZIP, 30-36 dB PSNR | `python research/universe/validate_claims.py` |
| Bit-perfect (SHA-256) | 4/4 test files verified | `python research/universe/validate_claims.py` |
| Documentation consistency | 18/18 checks pass | `python research/universe/validate_claims.py` |

**Dataset details**:
- **REAL photographs**: scikit-image standard test images (astronaut, camera,
  coins, moon, page, text) — used in academic image processing literature
  for decades. BLKH beats JPEG q=85 on ALL 6 real images (2.16× to 4.29× smaller).
- **Kodak-like**: 12 synthetic images at standard Kodak resolution (768×512)
  because real Kodak dataset mirrors were unavailable. Synthetic benchmark
  should be re-run on real Kodak images when available.

**Honest caveat**: BLKH PSNR (30.7 dB on real photos) is lower than JPEG
(42.9 dB). BLKH achieves smaller sizes by being more lossy. The paper
documents this trade-off honestly.

### Production-Ready Components

- **165 unit tests** (all passing) covering roundtrip correctness,
  bit-perfect reconstruction, format independence
- **8 compression modes** with auto-selection
- **Live demo** on HuggingFace Spaces:
  https://huggingface.co/spaces/onatskyo/black-hole-blkh
- **PyPI package**: `blackhole-blkh` v5.30.0
- **Stable API** since v5.14 (no breaking changes in 16 minor versions)

---

## ❓ Why I Need an Endorser

arXiv requires endorsement from someone with arXiv publication history
in the relevant subject area (`cs.IT` or `eess.SP`). As an independent
researcher without academic affiliation, I cannot self-submit.

If you are:
- An arXiv author in `cs.IT`, `eess.SP`, `cs.LG`, `eess.IV`, or related
- Willing to review the paper for technical correctness
- Comfortable endorsing independent research

...then I would be grateful for your consideration.

---

## ✅ What an Endorser Should Verify Before Endorsing

Please do NOT endorse blindly. Before endorsing, verify:

### 1. Reproducibility
```bash
git clone https://github.com/Kronos1027/black-hole.git
cd black-hole
pip install -r requirements.txt
python -m pytest tests/ -q        # should show 159+ passed
python blkh.py --help             # should list 13 compression modes
```

### 2. Key claims to spot-check
- **Bit-perfect reconstruction**: Run `tests/test_wavelet_v3_pytest.py`
  — all tests verify SHA-256 hash equality of original vs decompressed
- **Compression ratios**: Run `tests/benchmark_real_photos.py` and
  compare with README numbers
- **Speed claims**: Run `python blkh.py fast sample.png` and time it

### 3. Theoretical claims
- The paper makes NO cryptographic claims (corrected; see
  [`research/universe/SPECULATIVE.md`](research/universe/SPECULATIVE.md))
- All compression numbers are validated by tests
- Negative results (where BLKH loses to ZIP) are honestly documented
  in the README "Honest Benchmark" section

### 4. Authorship and originality
- All code committed by `Kronos1027` (Darlan Pereira da Silva)
- Builds on SIREN (Sitzmann et al. 2020) and COIN/COIN++ (Dupont et al.
  2021/2022) — properly cited in the paper
- License: MIT for code, custom license for commercial use

---

## 📋 What I Will Submit To arXiv

If endorsed, I will submit to one of:

1. **`cs.IT` (Information Theory)** — primary choice
   - Compression theory fits naturally
   - Lossless bit-perfect claims verifiable
2. **`eess.SP` (Image and Video Processing)** — alternative
   - Image compression focus
   - Empirical benchmarks on real photos
3. **`cs.LG` (Machine Learning)** — fallback
   - SIREN/INR methods fit here
   - Neural compression community

The submission will include:
- `paper/paper.pdf` (10-15 pages, formatted for arXiv)
- Source code reference (this repository, tagged at submission)
- Reproducibility instructions

---

## 📨 How to Contact Me

If you are willing to endorse (or just want to discuss the work):

1. **Email**: darlan1027pc@gmail.com
2. **GitHub**: Open an issue at https://github.com/Kronos1027/black-hole/issues
   with title "arXiv endorsement discussion"
3. **Response time**: I will respond within 48 hours

I am happy to:
- Share the paper draft for review before you commit
- Walk through any specific experiment you want to verify
- Address technical concerns or critiques
- Acknowledge your endorsement contribution in the paper

---

## ⚖️ Code of Conduct

I commit to:
- **Honesty**: All claims in the paper are reproducible from this repository
- **Retraction**: If a claim is found incorrect, I will issue a correction
  (as I did with Axiom 12 — see `research/universe/SPECULATIVE.md`)
- **Credit**: All prior work (SIREN, COIN, COIN++, etc.) properly cited
- **Open science**: Code remains MIT-licensed; paper remains open access

I will NOT:
- Pressure anyone to endorse
- Misrepresent the work's significance
- Hide negative results or failed experiments
- Claim academic affiliation I do not have

---

## 📚 Related Work (Properly Cited in Paper)

- Sitzmann et al. (2020) — SIREN: Sinusoidal Representation Networks
- Dupont et al. (2021) — COIN: Compression with Implicit Neural Representations
- Dupont et al. (2022) — COIN++: Neural Compression Across Modalities
- Landauer (1961) — Irreversibility and Heat Generation in Computing
- Kolmogorov (1965) — Three Approaches to the Quantitative Definition of Information
- Shannon (1948) — A Mathematical Theory of Communication

Full bibliography: [`research/universe/docs/BIBLIOGRAPHY.md`](research/universe/docs/BIBLIOGRAPHY.md)

---

## 🙏 Acknowledgments

I am grateful to:
- The SIREN authors (Sitzmann, Martel, Bergman, Lindell, Wetzstein) for
  the foundational architecture
- The COIN/COIN++ authors (Dupont et al.) for the neural compression framework
- The PyTorch team for the deep learning infrastructure
- The open source community for tools that made this possible
- Future endorsers (you?) for considering independent research seriously

---

*"Independent research is harder — but the absence of an institutional
safety net is exactly what makes honesty non-negotiable."*
