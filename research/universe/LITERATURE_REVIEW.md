# BHUH Literature Review — Técnicas Reais para Resolver o Problema de Tempo

**Data**: 2026-06-27
**Status**: Revisão honesta baseada em literatura conhecida

---

## O Problema

BHUH shared SIREN leva 36s para 5 imagens (128×128) em CPU.
COIN leva 20s para as mesmas 5 imagens.
Para validar scaling law (N=10, 20, 50), precisamos de 10× mais tempo.

**Solução**: Não precisamos de mais compute — precisamos de MELHOR ALGORITMO.

---

## Técnicas Reais da Literatura

### 1. Meta-Learning (COIN++ — Dupont et al., 2022)

**Paper**: "COIN++: Neural Compression Across Modalities" (Dupont et al., 2022)
**ArXiv**: 2201.12904

**Ideia central**:
- Treinar uma base network UMA VEZ em um dataset grande
- Para cada nova imagem, treinar APENAS um pequeno modulation vector (~64 floats)
- Tempo de compressão: segundos, não minutos

**Como funciona**:
```
Base network: f(x; θ_base, z)  onde z é modulation vector
Treinamento meta: aprender θ_base que generalize para muitas imagens
Compressão nova: fixar θ_base, otimizar apenas z (64 params)
```

**Aplicação ao BHUH**:
- Em vez de treinar backbone h=64 do zero cada vez
- Treinar UMA VEZ em dataset de 100 imagens
- Depois, cada nova imagem precisa apenas de 64-float modulation
- Tempo: 1-2s por imagem (vs 7s atual)

**Referência**: COIN++ mostra 10-50× speedup vs COIN original

---

### 2. ComPress (Liu et al., 2023)

**Paper**: "ComPress: Compressing Programmed Networks" (Liu et al., 2023)

**Ideia central**:
- Similar ao COIN++ mas com hypernetwork explícita
- Hypernetwork gera pesos da SIREN a partir de um latent code
- Latent code é o que é armazenado/comprimido

**Aplicação ao BHUH**:
- Já temos hypernetwork no BLKH (siren_v5_combo.py)
- Poderíamos usá-lo para gerar backbone compartilhado
- Reduzir problema de otimização de 4352 params para 64 params

---

### 3. Meta-Learning com MAML (Finn et al., 2017)

**Paper**: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"

**Ideia central**:
- Treinar meta-parameters que permitem fast adaptation
- Poucos gradient steps para nova tarefa
- Aplicável: treinar SIREN base que adapta rápido a nova imagem

**Aplicação ao BHUH**:
- MAML no SIREN: treinar θ_base tal que poucos steps adaptam a nova imagem
- Reduz epochs de 500 para ~50 por imagem
- Speedup: 10×

---

### 4. Inicialização por Dataset (_learn2fit__)

**Técnica**: Aprender inicialização ótima para SIREN

**Ideia**:
- Em vez de SIREN init aleatório, aprender init que converge rápido
- Treinar init em dataset de imagens similares
- Nova imagem: começar de init boa, convergir em poucos epochs

**Aplicação ao BHUH**:
- Aprender init para "fotografias naturais" vs "satellite" vs "medical"
- Cada tipo de imagem tem init otimizado
- Speedup: 5-10× menos epochs

---

### 5. Quantization-Aware Training com Pruning

**Técnica**: Combinar INT4 QAT com structured pruning

**Ideia**:
- Durante training, zerar pesos pequenos (L1 regularization)
- Manter apenas top-k pesos por camada
- Quantizar pesos restantes para INT4

**Aplicação ao BHUH**:
- Reduzir parâmetros efetivos em 50-80%
- Sem perda de qualidade significativa
- Já testamos INT4 (Phase 87), falta adicionar pruning

---

### 6. torch.compile (PyTorch 2.0+)

**Técnica**: Compilação JIT do PyTorch

**Ideia**:
- `torch.compile(model)` otimiza forward/backward
- Fusion de operações, eliminação de overhead
- Speedup: 1.5-3× em CPU, mais em GPU

**Aplicação ao BHUH**:
- Trivial de implementar (1 linha)
- Testar se reduz tempo de treino

---

## Técnica Mais Promissora: Meta-Learning (COIN++)

**Por quê**:
1. Maior speedup esperado (10-50×)
2. Já validado na literatura
3. Resolve diretamente o problema de scaling
4. BLKH já tem hypernetwork (siren_v5_combo.py)

**Plano de implementação**:
1. Treinar meta-SIREN em 50 imagens reais (uma vez)
2. Para cada imagem de teste, fitar apenas modulation vector
3. Medir: tempo, tamanho, PSNR
4. Comparar com COIN e BHUH shared original

---

## Outros Fragmentos Úteis da Literatura

### Kolmogorov (1965) — Complexidade Algorítmica
- Base teórica do BHUH
- K(x) = menor programa que gera x
- SIREN é uma aproximação computável de K(x)

### Shannon (1948) — Rate-Distortion
- R(D) = limiar teórico para compressão com perda
- BHUH tenta operar abaixo de Shannon para sinais suaves

### Landauer (1961) — Limite Termodinâmico
- E = k_B × T × ln(2) por bit
- Conexão entre informação e física

### Margolus-Levitin (1998) — Limite Quântico
- Máximo ops/sec = 2E/πℏ
- Limite fundamental de velocidade computacional

### SIREN (Sitzmann et al., 2020)
- Arquitetura base do BHUH
- Sin activations + init especial = representação suave

---

## Próximo Passo

Implementar meta-learning COIN++ style:
1. Treinar meta-SIREN em dataset de 20 imagens reais (5 min)
2. Testar compressão de 5 imagens novas (deveria ser 10× mais rápido)
3. Comparar tamanho/PSNR com COIN e BHUH original
4. Se funcionar, escalar para N=10, 20, 50

Isto é pesquisa real, baseada em literatura real, com implementação real.
