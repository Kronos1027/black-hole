# Black Hole Whitepaper — Repositório de Pesquisa

Este repositório contém o whitepaper técnico v1.0 do projeto **Black Hole (BLKH)** — uma proposta de software de compressão neural oportunista com pré-cálculo em ciclos ociosos.

## Conteúdo

- **BlackHole_Whitepaper.pdf** — Whitepaper técnico completo (51 páginas, 7 capítulos, 7 figuras)
- **BlackHole_Whitepaper.md** — Versão Markdown editável do whitepaper
- **scripts/** — Código Python reprodutível:
  - `00_setup_test_data.py` — Gera dados de teste (texto PT, imagem, binário, diretório)
  - `01_siren_core.py` — Implementação SIREN em PyTorch (Sitzmann et al. 2020)
  - `03_generate_charts.py` — Gera os 7 gráficos comparativos
- **results/raw_results.json** — Resultados crus dos testes empíricos (reprodutíveis)
- **chart_01..07.png** — Gráficos usados no whitepaper

## Resumo dos Achados Empíricos

| Cenário | SIREN Lossy | SIREN Lossless | gzip | lzma |
|---------|-------------|----------------|------|------|
| Texto 1KB | 0.08x (8x pior) | 0.08x (8x pior) | 1.69x | 1.43x |
| Texto 4KB | 0.08x (12x pior) | 0.08x (12x pior) | 2.10x | 2.02x |
| Texto 16KB | 0.06x (16x pior) | 0.06x (16x pior) | 7.51x | 7.52x |
| Imagem 32x32 (vs raw) | 1.33x | n/a | n/a | n/a |
| Diretório 10KB | 1.19x (lossy) | 0.66x (expande) | 2.08x | 2.14x |

**Veredicto**: SIREN puro é inviável como compressor universal. Black Hole reformulado como camada híbrida adaptativa é viável.

## Como Reproduzir

```bash
python3.13 scripts/00_setup_test_data.py    # gera dados de teste
python3.13 scripts/01_siren_core.py          # testa SIREN em 256 bytes
python3.13 scripts/02_run_all_tests.py       # bateria completa (lento)
python3.13 scripts/03_generate_charts.py     # gera gráficos
```

Requer: Python 3.13, PyTorch (CPU), numpy, matplotlib, Pillow.

## Citar

```
Projeto Black Hole. "Black Hole Whitepaper v1.0: Compressão Neural Oportunista 
com Pré-cálculo em Ciclos Ociosos." Research Artefact, Junho 2026.
```

## Licença

Conteúdo do whitepaper: Creative Commons Attribution 4.0 (CC BY 4.0).
Código: MIT License.

## Contato

Para colaboração acadêmica ou técnica, abrir issue no repositório GitHub.
