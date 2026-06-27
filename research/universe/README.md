# 🌌 Black Hole Universe — Research Program

> **"The data is not a static block of bytes waiting to be dragged. The data is a living mathematical function, kept in potential state and ejected instantly on demand."**

**Status**: Active Research (started 2026-06-25)
**Lead Researcher**: Darlan Pereira da Silva (Kronos1027)
**Assistance**: GLM (AI research assistant)
**License**: MIT (research) + commercial license for production

---

## 🎯 The Hypothesis

### Black Hole Universe Hypothesis (BHUH)

> **Toda informação estruturada pode ser representada como uma "singularidade" — um gerador matemático mínimo — que, ao ser "executado", faz o arquivo "nascer" do zero como uma árvore cresce da semente. Múltiplos arquivos compartilham "raízes" no espaço de geradores, formando um "multiverso" onde cada arquivo é uma trajetória. A compressão torna-se descoberta da semente; a descompressão torna-se gênese.**

---

## 📐 The 5 Principles

### 1. **Singularity** (Singularidade)
Every structured file collapses to a minimal mathematical generator — its "seed" or "atom". This is a practical approximation of Kolmogorov complexity K(x).

### 2. **Genesis** (Gênese)
Decompression is not "unpacking" — it is organic growth. The file is born from the seed through execution, like a tree from a seed.

### 3. **Multiverse** (Multiverso)
Files are not isolated. Multiple files share "roots" — common structure in the generator space. Each file is a trajectory through a shared latent universe.

### 4. **Universality** (Universalidade)
Works for any data type via type-specific architectures (SIREN for images, transformers for text, etc.) with symbolic fallback for non-neural-friendly data.

### 5. **Hybridism** (Hibridismo)
No single approach works for everything. The Universe combines:
- **Neural** (SIREN, INRs) — for smooth/continuous data
- **Symbolic** (program synthesis, LLM search) — for discrete/structured data
- **Statistical** (diffusion, VAE) — for stochastic/natural data

---

## 🚪 Entry Points

- **[Hypothesis & Theory](docs/THEORY.md)** — Full mathematical foundation
- **[Research Roadmap](docs/ROADMAP.md)** — 5-10 year plan
- **[Experiments](experiments/)** — Active experimental code
- **[Bibliography](docs/BIBLIOGRAPHY.md)** — 50+ references ancient to modern
- **[Experiment Log](docs/EXPERIMENT_LOG.md)** — Results tracking
- **[Production BLKH](../phase1_inr_compressor/)** — Stable production code (untouched)

### Scientific Honesty Documents
- **[SPECULATIVE.md](SPECULATIVE.md)** — Unvalidated claims, clearly marked
- **[DOCUMENTATION_PROTOCOL.md](DOCUMENTATION_PROTOCOL.md)** — Protocol to prevent knowledge loss
- **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)** — Honest summary of all results

---

## ⚠️ Scientific Honesty Notice

The BHUH research program has 14 candidate axioms. Not all are validated:

- **8 validated** (empirically tested with reproducible code)
- **3 partial** (some aspects work, others fail)
- **2 failed/rejected** (honest negatives, documented)

Additionally, some earlier theoretical claims have been **retracted or
corrected**:
- Axiom 12 was originally called "One-Way Function" and compared to AES/RSA.
  This was **technically incorrect** (BHUH inverse is polynomial-time).
  It has been renamed to "Computational Asymmetry" and the comparison removed.
- Axiom 11 (Subspace Compression) was rejected twice (Phase 80 linear,
  Phase 82 nonlinear).
- Physics analogies (AdS/CFT, Bekenstein) are CONCEPTUAL, not formal
  derivations.

See [SPECULATIVE.md](SPECULATIVE.md) for full details on what is and
isn't validated.

---

## 🧪 Active Experiments

| Phase | Experiment | Status | Description |
|-------|------------|--------|-------------|
| 1 | Multi-File SIREN | 🔄 In progress | 1 network for 100 images |
| 2 | Universal Hypernetwork | ⏳ Planned | Generalize to text/audio/binary |
| 3 | LLM Program Search | ⏳ Planned | LLM writes file generators |
| 4 | Diffusion Seed | ⏳ Planned | Diffusion as compression |
| 5 | Universe Prototype | ⏳ Planned | Combine all approaches |

---

## 🛡️ Safety Guarantees

### Production code is SAFE
- `phase1_inr_compressor/` — STABLE, used by BLKH v5.30
- `tests/` — 165 passing tests
- `blkh.py` — CLI untouched
- **This research does NOT modify production code**

### Research is ISOLATED
- All experimental code in `research/universe/`
- Cannot break existing BLKH functionality
- Results feed back to production ONLY when proven

### Work is PRESERVED
- All experiments documented in `docs/EXPERIMENT_LOG.md`
- All theory in `docs/THEORY.md`
- All papers/references in `docs/BIBLIOGRAPHY.md`
- Git history preserves everything

---

## 📊 Honest Limitations

The Universe Hypothesis is NOT a magic bullet. It has fundamental limits:

| Limit | Reason | Mitigation |
|-------|--------|------------|
| Cannot compress random data | Shannon entropy limit | Fallback to zlib |
| Cannot compress encrypted data | Indistinguishable from random | Fallback to zlib |
| K(x) is incomputable | Chaitin's theorem | Approximate via search |
| Search is expensive | NP-hard at minimum | Use neural heuristics |
| Small files (<64B) | Overhead dominates | Use traditional codecs |

**What we CAN do**:
- Structured data → 10-100x compression (vs ZIP)
- Similar file corpora → additional 2-10x via shared roots
- Universal across types with appropriate architecture
- Real-time decompression once seed is found

---

## 🎓 Historical Lineage

This research builds on ideas spanning 2500 years:

| Era | Thinker | Contribution |
|-----|---------|--------------|
| Ancient | Pythagoras (~570 BCE) | "All is number" |
| Classical | Plato (~400 BCE) | Theory of Forms |
| Medieval | Fibonacci (~1200) | Recursive generation |
| Enlightenment | Leibniz (1666) | Characteristica Universalis |
| Modern | Euler (1748) | e^(iπ)+1=0 (minimal structure) |
| Modern | Cantor (1874) | Hierarchy of infinities |
| Modern | Turing (1936) | Universal machine |
| Information | Shannon (1948) | Information theory |
| Information | Kolmogorov (1965) | Algorithmic complexity |
| Information | Solomonoff (1960) | Universal induction |
| Information | Chaitin (1966) | Incompleteness |
| Modern | von Neumann (1948) | Cellular automata |
| Modern | Wolfram (2002) | Computational universe |
| Modern | Susskind (1994) | Holographic principle |
| Modern | Sitzmann (2020) | SIREN |
| Modern | DeepMind (2023) | FunSearch |

**The Universe Hypothesis is the synthesis of all these threads.**

---

## 🤝 Contributing to the Research

This is open research. Contributions welcome:

1. **Replicate experiments** — see `experiments/`
2. **Suggest new approaches** — open an issue with `[RESEARCH]` prefix
3. **Share datasets** — for benchmarking
4. **Cite the work** — see citation below

### Citation

```bibtex
@misc{blkh_universe_2026,
    title={Black Hole Universe: A Hypothesis on Generative Compression via Mathematical Singularities},
    author={Pereira da Silva, Darlan},
    year={2026},
    url={https://github.com/Kronos1027/black-hole/tree/main/research/universe}
}
```

---

## 📞 Contact

- **Author**: Darlan Pereira da Silva (Kronos1027)
- **Email**: darlan1027pc@gmail.com
- **GitHub**: https://github.com/Kronos1027/black-hole
- **Live demo**: https://huggingface.co/spaces/onatskyo/black-hole-blkh

---

## 📄 arXiv Publication Status

A paper draft is ready at [`paper/paper.pdf`](../../paper/paper.pdf) but
**arXiv endorsement is needed** to submit. As an independent researcher
without academic affiliation, I cannot self-submit.

### Research Documentation Status
- ✅ 84 experimental phases documented (71-84 from Phase II)
- ✅ 14 axiom candidates (8 validated, 3 partial, 2 rejected, 1 corrected)
- ✅ Honest disclosure of failed/retracted claims in [`SPECULATIVE.md`](SPECULATIVE.md)
- ✅ Reproducibility protocol in [`DOCUMENTATION_PROTOCOL.md`](DOCUMENTATION_PROTOCOL.md)
- ✅ 165 production tests passing (untouched by research)

### What an arXiv endorser would verify
1. Empirical claims (compression ratios, PSNR) are reproducible from code
2. Theoretical claims (Phase 73 thermodynamic framework, Phase 84
   Kolmogorov approximation) are honestly caveated
3. Failed experiments (Axiom 11 subspace, Axiom 6 self-modification)
   are documented as negatives
4. The corrected Axiom 12 (Computational Asymmetry, not "One-Way Function")
   makes no invalid cryptographic claims

### How to help
- **Email**: darlan1027pc@gmail.com
- **GitHub issue**: title "arXiv endorsement discussion"
- **Full details**: [`ARXIV_ENDORSEMENT.md`](../../ARXIV_ENDORSEMENT.md)

I'm happy to share the paper draft privately before any commitment.
Independent research is harder — but the absence of an institutional
safety net is exactly what makes honesty non-negotiable.

---

*"We are not just compressing data. We are discovering the mathematical DNA of information itself."*
