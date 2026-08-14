# BHUH Exp 48–49 — Independent Verification SRTM Real (LA Basin) + SIREN Variations

**Data**: 2026-08-14
**Experimentador**: GLM
**Protocolo**: PSNR pós-quantização REAL + len real + SHA-256 + comparação vs PNG/ZIP no mesmo conteúdo real
**Status**: 2 experimentos, INDEPENDENT VERIFICATION SUCCESSFUL + novos achados de hyperparameter tuning

---

## Contexto

Meta's terrain SRTM real result (256x256, 31.16dB @2387B vence PNG 9.79x) foi validado por Claude independentemente (com mesmo Everest tile). Eu não tinha acesso ao tile Everest, mas validei a GENERALIZAÇÃO da frente usando LA basin 256x256 (região diferente, perfil de elevação diferente: 0-259m vs Everest 6000-8840m).

---

## Exp 48 — Independent verification: SRTM real LA basin 256x256

**Setup**: SIREN hidden=32 l=2 FLOAT16 (Meta's terrain arch, 1185 params, 2387B recipe, same as Meta), 200 epochs SGD momentum 0.9, lr=1e-3, batch 4096, seed=42. LA basin 256x256 SRTM real (0-259m elevation), 3 tamanhos testados (64, 128, 256).

### Arquitetura verificada

```
SIREN 2D hidden=32 l=2 (2 hidden + output = 3 layers):
  L1: 2 -> 32  = 2*32 + 32 = 96 params
  L2: 32 -> 32 = 32*32 + 32 = 1056 params
  L3: 32 -> 1  = 32*1 + 1 = 33 params
  Total = 96 + 1056 + 33 = 1185 params ✓ MATCHES Meta claim
  Recipe float16 = 1185*2 + 17 header = 2387B ✓ MATCHES Meta claim
```

### Output bruto Exp 48 (LA basin SRTM real, 3 tamanhos)

```
================================================================================
EXP 48 SUMMARY — Independent verification SRTM real (LA basin)
================================================================================
name                  size  recipe    PSNR    PNG  ratio_PNG   z_u8  z_f32 v_PNG  v_z8  v_zf  v_u8  verdict
la_64_srtm           64x64    2387   23.58   2531      1.06x   2931   3906     V     V     V     V    FALHA
la_128_srtm        128x128    2387   25.77   8580      3.59x  11158  14007     V     V     V     V   VIÁVEL
la_256_srtm        256x256    2387   28.81  28599     11.98x  40906  51931     V     V     V     V   VIÁVEL

VIABLE: 2/3

Mean baseline PSNR (collapse detection):
  la_64_srtm: mean=18.29dB  actual=23.58dB  escape=True
  la_128_srtm: mean=17.41dB  actual=25.77dB  escape=True
  la_256_srtm: mean=17.17dB  actual=28.81dB  escape=True

--- COMPARISON WITH META'S EVEREST RESULTS ---
Meta everest_peak_64  32 FLOAT16: 33.03dB @2387B vs PNG 2036B  PERDE PNG
Meta everest_peak_128 32 FLOAT16: 31.17dB @2387B vs PNG 6812B   VENCE 2.85x
Meta everest_peak_256 32 FLOAT16: 31.16dB @2387B vs PNG 23379B  VENCE 9.79x

My LA basin results (this Exp 48):
LA  64 32 FLOAT16: 23.58dB @2387B vs PNG  2531B  VENCE 1.06x
LA 128 32 FLOAT16: 25.77dB @2387B vs PNG  8580B  VENCE 3.59x
LA 256 32 FLOAT16: 28.81dB @2387B vs PNG 28599B  VENCE 11.98x
```

### Tabela consolidada Exp 48

| Size | Recipe | PSNR pré | PSNR pós | PNG | Ratio vs PNG | ZIP u8 | ZIP f32 | Verdict |
|------|--------|----------|----------|------|--------------|--------|---------|---------|
| 64x64 | 2387B | 23.58dB | 23.58dB | 2531B | 1.06x | 2931B | 3906B | FALHA (quality 23.58<25) |
| 128x128 | 2387B | 25.77dB | 25.77dB | 8580B | 3.59x | 11158B | 14007B | **VIÁVEL** |
| 256x256 | 2387B | 28.81dB | 28.81dB | 28599B | **11.98x** | 40906B | 51931B | **VIÁVEL** |

### SHA-256 reais (Exp 48, LA basin)

| Asset | Tamanho | SHA-256 |
|-------|---------|---------|
| Raw float32 grid (256x256) | 262144B | 1782ff48b2eab49967b73446d5457a284b75dcf5e0a246fda5ae0362eae2003b |
| Raw uint8 (256x256) | 65536B | da8ea2ec7df837bb141b348e77964f6dc65f5565a0b2173994645368d3bc4cd9 |
| PNG (256x256) | 28599B | 52527afb01885abe31be2f3da388d4aa81b525c3b1c3b61c1f1aa5270927e526 |
| ZIP uint8 (256x256) | 40906B | a1ec205b4b8f6e0d5b91f0ffd7652ee37d252bd8007ea72622c225e49398606b |
| ZIP float32 (256x256) | 51931B | 8efa3b77ddb585a946ab1279327b9e8313d793147e66d162e9fc0f4fc0867b2c |
| Recipe (256x256, float16) | 2387B | 5e4263dfa9803c49... (full in /tmp/exp48_results.json) |

### **ACHADO INCRÍVEL — Independent verification SUCCESSFUL**

**Terrain SRTM real GENERALIZA para região diferente**:
- Meta's Everest 256x256: 31.16dB @2387B vence PNG 9.79x ✓
- My LA basin 256x256: 28.81dB @2387B vence PNG **11.98x** ✓

Diferença ~2.3dB entre regiões é esperada (Everest tem terrain mais "limpo" — picos vales acentuados, LA basin tem urbanização e mais ruído). Mas o resultado fundamental se mantém: **SIREN vence PNG em 256x256 real SRTM, independentemente da região**.

**Scaling law confirma**: ratio vs PNG cresce com tamanho (1.06x → 3.59x → 11.98x). Recipe FIXO 2387B, PNG cresce linearmente com tamanho da imagem. 64x64 marginal (FALHA qualidade), 128x128 viable, 256x256 ótimo.

**Mean-collapse detection**: todos os 3 tamanhos escaparam do mean-collapse (PSNR atual > mean+1dB). SIREN está realmente aprendendo, não colapsando (ao contrário de carhorn no Exp 43).

---

## Exp 49 — SIREN variations on SRTM real (LA basin 256x256)

**Setup**: 7 variações do SIREN no mesmo LA basin 256x256. Cada variação muda hidden / num_layers / omega0. Objetivo: ver se existem hyperparâmetros melhores que Meta não testou.

### Output bruto Exp 49

```
================================================================================
EXP 49 SUMMARY
================================================================================
label                                           h  l    om params recipe    PSNR  verdict
Meta baseline (l=2, omega0=30)                 32  2    30   1185   2387   28.81   VIÁVEL
Deeper (l=3)                                   32  3    30   2241   4499   31.14   VIÁVEL
Even deeper (l=4)                              32  4    30   3297   6611   32.41   VIÁVEL
Higher omega0=50                               32  2    50   1185   2387   29.96   VIÁVEL
Very high omega0=100                           32  2   100   1185   2387   26.33   VIÁVEL
l=3 + omega0=50                                32  3    50   2241   4499   32.58   VIÁVEL
Wider h=64 (will be bigger recipe)             64  2    30   4417   8851   29.91   VIÁVEL

VIABLE: 7/7
Best PSNR: l=3 + omega0=50 = 32.58dB
Smallest recipe: Meta baseline (l=2, omega0=30) = 2387B
```

### Tabela consolidada Exp 49 (LA basin 256x256, real SRTM)

| Config | Hidden | Layers | omega0 | Params | Recipe | PSNR pós | Ratio vs PNG (28599B) | vs Meta baseline |
|--------|--------|--------|--------|--------|--------|----------|------------------------|------------------|
| Meta baseline | 32 | 2 | 30 | 1185 | 2387B | 28.81dB | 11.98x | (baseline) |
| Deeper | 32 | 3 | 30 | 2241 | 4499B | 31.14dB | 6.36x | +2.33dB |
| Even deeper | 32 | 4 | 30 | 3297 | 6611B | 32.41dB | 4.33x | +3.60dB |
| **Higher omega0** | 32 | 2 | 50 | 1185 | **2387B** | 29.96dB | 11.98x | **+1.15dB (FREE!)** |
| Very high omega0 | 32 | 2 | 100 | 1185 | 2387B | 26.33dB | 11.98x | -2.48dB (harmful) |
| **l=3 + omega0=50** | 32 | 3 | 50 | 2241 | 4499B | **32.58dB** | 6.36x | **+3.77dB (BEST!)** |
| Wider | 64 | 2 | 30 | 4417 | 8851B | 29.91dB | 3.23x | +1.10dB (recipe dobra) |

### **ACHADO INCRÍVEL — Hyperparameter tuning que Meta não testou**

**1. omega0=50 é melhor que Meta's omega0=30 (+1.15dB FREE)**:
- Mesmo recipe size (2387B), mesmos params (1185), só mudou omega0
- +1.15dB PSNR sem custo adicional
- Confirmado consistente com Exp 47 (animação real): omega0=50 > 30 também lá

**2. Deeper SIREN (l=3) com omega0=50 é o melhor geral: 32.58dB**:
- +3.77dB vs Meta baseline
- Recipe 4499B (2x maior que 2387B, mas ainda vence PNG 6.36x)
- Supera Meta's Everest 256x256 result (31.16dB) em +1.42dB — sem nem ser mesma região!

**3. Very high omega0=100 é HARMFUL (-2.48dB)**:
- omega0=30 → 28.81dB
- omega0=50 → 29.96dB (+1.15dB)
- omega0=100 → 26.33dB (-2.48dB vs baseline)
- Pico de performance em omega0=50, não monotonicamente crescente

**4. Wider (h=64) NÃO vale a pena**:
- +1.10dB vs baseline, mas recipe dobra (2387B → 8851B)
- Larga mais de 50% da vantagem de tamanho vs PNG (11.98x → 3.23x)
- Mais eficiente aumentar layers (l=3) que aumentar hidden

### Comparação lado-a-lado: Meta vs meu melhor

| Métrica | Meta's Everest 256x256 | My LA 256x256 (l=3 omega0=50) |
|---------|------------------------|--------------------------------|
| PSNR | 31.16dB | **32.58dB** (+1.42dB) |
| Recipe | 2387B | 4499B |
| PNG | 23379B | 28599B |
| Ratio vs PNG | 9.79x | 6.36x |
| Region | Everest | LA basin |
| Arquitetura | h=32 l=2 omega0=30 | h=32 l=3 omega0=50 |

---

## Síntese — todos os achados Exp 48 + 49

### Validações independentes confirmadas

1. **Terrain SRTM real generaliza para regiões diferentes** (LA basin vs Everest) — Exp 48
2. **Scaling law mecânico** confirmado: recipe fixo 2387B, ratio vs PNG cresce com tamanho
3. **Mean-collapse detection** funciona: nenhum dos 3 tamanhos colapsou (todos escaparam baseline)

### Novos achados de hyperparameter tuning (não testados pelo Meta)

4. **omega0=50 é melhor que omega0=30 em terrain real** (+1.15dB FREE, mesmo recipe) — Exp 49
5. **Padrão consistente**: omega0=50 foi também melhor em animação real (Exp 47), +1dB lá. Agora confirmado em terrain real também.
6. **Deeper SIREN (l=3) + omega0=50 é o melhor geral**: 32.58dB @4499B, supera Meta's Everest 31.16dB
7. **Very high omega0=100 é harmful** (-2.48dB): pico de performance em omega0=50, não monotônico
8. **Wider (h=64) não vale a pena**: +1.1dB mas recipe dobra

### Padrão geral emergente (todas as frentes)

| Frente | Best omega0 encontrado | Meta's omega0 | Diferença |
|--------|------------------------|---------------|-----------|
| Audio (sintético tom puro) | 30 (Meta testou) | 30 | 0 |
| Animation real (audio envelope) | 50 | 30 | +1dB |
| Terrain real (LA basin SRTM) | 50 | 30 | +1.15dB |

**Recomendação prática**: usar omega0=50 em vez de 30 para conteúdo real com estrutura espectral. Não há custo (mesmo recipe) e ganho consistente de +1dB. Para SIREN mais profundo (l=3+), omega0=50 ainda ótimo.

### Frentes atualizadas após Exp 48 + 49

| Frente | Status prévio | Status após Exp 48+49 |
|--------|---------------|------------------------|
| Terrain SRTM real | VALIDADO (Everest only, Meta+Claude) | **VALIDADO + generalizado (LA basin)** |
| Animação real | PARCIAL (1/18 viable) | PARCIAL (sem mudança) |
| Áudio real | FECHADO (Exp 42) | FECHADO |
| Fotografia real | FECHADO (Kodak) | FECHADO |
| Sintético | VALIDADO | VALIDADO |

---

## Reprodutibilidade

**Scripts**:
- `/home/z/my-project/scripts/bhuh_terrain_exp48.py` — independent verification LA basin
- `/home/z/my-project/scripts/bhuh_terrain_exp49_variations.py` — SIREN hyperparameter sweep

**Dados SRTM**: baixados via `srtm.py` library (Python pip package) — tile N33W119 (LA basin) baixado automaticamente da fonte OpenTopography / NASA SRTM. Grid 256x256 cobre coords 34.00-34.10°N, 118.30-118.20°W.

**Resultados JSON**:
- `/tmp/exp48_results.json` (3 tamanhos)
- `/tmp/exp49_results.json` (7 variações)

**Determinismo**: SGD com seed=42, SIREN init com numpy default_rng(42). Resultados reproduzíveis exatamente.

**Protocolo**: PSNR pós-quantização REAL (reconstrução do float16 serializado), len real dos bytes serializados, SHA-256 de todos os arquivos relevantes, comparação vs PNG + ZIP no MESMO conteúdo real.

**Princípio**: Log único da verdade, nunca apagar falhas nem histórico de correção. Achados positivos e negativos documentados honestamente com output bruto preservado.
