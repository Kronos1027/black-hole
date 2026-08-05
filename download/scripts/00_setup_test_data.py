#!/usr/bin/env python3.13
"""
Gera dados de teste realistas para os experimentos do Black Hole.
Cria:
  - test_data/sample_text.txt       (texto literário em PT, ~50KB)
  - test_data/sample_binary.bin     (binário pseudoaleatório + estruturado, 50KB)
  - test_data/sample_image.png      (imagem procedural 128x128 RGB)
  - test_data/sample_code.c         (código fonte C, ~10KB)
  - test_data/sample_dir/           (diretório com 10 arquivos mistos)
"""
import os
import struct
import random
import zlib
import io
from PIL import Image
import math

random.seed(42)

OUT_DIR = "/home/z/my-project/test_data"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "sample_dir"), exist_ok=True)

# 1) Texto literário em português (~50KB)
# Vamos gerar um texto coerente repetindo e variando parágrafos
paragraphs = [
    "A determinação é o combustível que move a inovação tecnológica. Sem ela, nenhuma ideia por mais brilhante que seja consegue atravessar o vale da morte que separa o conceito da implementação real.",
    "O Black Hole não é apenas um compactador de arquivos. É uma nova filosofia de como o hardware gerencia a existência da informação. Cada byte deixa de ser um bloco estático e passa a ser uma função matemática viva, mantida em estado potencial e ejetada instantaneamente sob demanda.",
    "Em vez de empacotar arquivos em um contêiner rígido, o software destrói a estrutura original e a reconstrói como uma equação contínua. Isso rompe com o modelo de Von Neumann tradicional, onde o disco guarda dados frios e o processador os puxa sob estresse.",
    "A computação oportunista monitora os ciclos ociosos da CPU e GPU. Quando o computador está sem fazer nada, a IA interna usa essa energia que já seria gasta para manter as receitas dos arquivos mais prováveis rodando em segundo plano, sem causar atraso perceptível ao usuário.",
    "O jato de informação, análogo ao que ocorre em um buraco negro, dispara a partir do centro com foco absurdo. O dado é enraizado, construído e ejetado, sem causar um pico gigante de consumo, pois já estava sendo calculado de forma estável dentro da bolha.",
    "A arquitetura zero-copy elimina intermediários do sistema operacional. Instruções diretas de hardware como io_uring e DirectStorage permitem que o dado seja ejetado do núcleo da IA direto para a memória RAM ou VRAM de execução.",
    "O limite de Shannon estabelece o teto teórico de qualquer esquema de compressão sem perdas. Para dados aleatórios, esse teto é exatamente o tamanho original, pois não há redundância a ser explorada por nenhum algoritmo, por mais inteligente que seja.",
    "As representações neurais implícitas, conhecidas como INRs, oferecem um caminho alternativo. Ao invés de armazenar o dado diretamente, armazenam-se os pesos de uma rede neural treinada para reconstruí-lo. Isso é poderosíssimo para dados com estrutura espacial contínua.",
    "A função de ativação senoidal, proposta no paper SIREN de Sitzmann et al. em 2020, permite que as redes neurais representem com fidelidade sinais com altas frequências. Isso é fundamental para comprimir texturas, áudio e imagens sem perder detalhes.",
    "O paper COIN, publicado por Dupont et al. em 2021, demonstrou que é possível comprimir imagens usando INRs. Porém, os resultados mostram que para a maioria dos arquivos do mundo real, os métodos tradicionais como BPG e JPEG XL ainda ganham em razão de compressão.",
    "Para compressão sem perdas, o cenário é ainda mais desafiador. As INRs são inerentemente aproximativas, pois usam números de ponto flutuante. Reconstruir byte a byte um executável ou um arquivo de texto exige precisão absoluta que redes neurais não fornecem naturalmente.",
    "O custo de treinamento é o calcanhar de Aquiles dessa abordagem. Comprimir um único arquivo de imagem pode exigir minutos de GPU, enquanto o JPEG faz o mesmo em milissegundos. Esse fator de mil é difícil de superar com otimizações de engenharia.",
    "A descompressão, por outro lado, é onde as INRs poderiam brilhar. Avaliar uma rede neural é uma operação altamente paralelizável, especialmente em GPU. O fluxo de dados é previsível e pode ser otimizado em nível de hardware.",
    "A visão original do usuário, de ter receitas rodando em background para pré-calcular arquivos, é elegante mas esbarra na segunda lei da termodinâmica da computação. Manter redes neurais em execução constante consome energia, mesmo que em pequena quantidade.",
    "O caminho realista para o Black Hole não é substituir o WinRAR. É criar uma camada de cache inteligente em nível de sistema operacional que decida automaticamente qual estratégia de compressão usar para cada arquivo, baseado em seu conteúdo e padrão de acesso.",
]

text = ""
while len(text) < 50000:
    random.shuffle(paragraphs)
    text += "\n\n".join(paragraphs) + "\n\n"

text = text[:50000]
with open(os.path.join(OUT_DIR, "sample_text.txt"), "w", encoding="utf-8") as f:
    f.write(text)
print(f"[OK] sample_text.txt: {len(text)} bytes")

# 2) Binário misto: 30% aleatório + 40% estruturado (integers, floats) + 30% repetitivo
random_bytes = bytes(random.getrandbits(8) for _ in range(15000))
structured = b""
for i in range(2500):
    structured += struct.pack("<i", i * 7)  # 10000 bytes
for i in range(1250):
    structured += struct.pack("<d", math.sin(i * 0.1))  # 10000 bytes
repetitive = b"\x00" * 5000 + b"\xff" * 5000 + (b"BLACKHOLE" * 555)  # 15000 bytes
binary = random_bytes + structured + repetitive
with open(os.path.join(OUT_DIR, "sample_binary.bin"), "wb") as f:
    f.write(binary)
print(f"[OK] sample_binary.bin: {len(binary)} bytes")

# 3) Imagem procedural 128x128 - gradiente + ondas + ruído
W, H = 128, 128
img = Image.new("RGB", (W, H))
px = img.load()
for y in range(H):
    for x in range(W):
        r = int(127 + 127 * math.sin(x * 0.05))
        g = int(127 + 127 * math.cos(y * 0.05))
        b = int(127 + 127 * math.sin((x + y) * 0.05))
        noise = random.randint(-10, 10)
        px[x, y] = (max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise)))
img.save(os.path.join(OUT_DIR, "sample_image.png"))
print(f"[OK] sample_image.png: {os.path.getsize(os.path.join(OUT_DIR, 'sample_image.png'))} bytes (raw: {W*H*3} bytes)")

# 4) Código C ~10KB
code = """/* Black Hole - Prototype implementation
 * SIREN-based data compression proof of concept
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 1024
#define HIDDEN_DIM 256
#define NUM_LAYERS 3

typedef struct {
    float weights[NUM_LAYERS][HIDDEN_DIM][HIDDEN_DIM];
    float biases[NUM_LAYERS][HIDDEN_DIM];
} SirenNetwork;

float sin_activation(float x) {
    return sinf(x);
}

void forward(SirenNetwork* net, float* input, float* output) {
    float hidden[HIDDEN_DIM];
    float next[HIDDEN_DIM];
    memcpy(hidden, input, HIDDEN_DIM * sizeof(float));
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float acc = net->biases[l][i];
            for (int j = 0; j < HIDDEN_DIM; j++) {
                acc += net->weights[l][i][j] * hidden[j];
            }
            next[i] = sin_activation(acc);
        }
        memcpy(hidden, next, HIDDEN_DIM * sizeof(float));
    }
    memcpy(output, hidden, HIDDEN_DIM * sizeof(float));
}

int main(int argc, char** argv) {
    printf("Black Hole prototype v0.1\\n");
    SirenNetwork net;
    memset(&net, 0, sizeof(net));
    float input[HIDDEN_DIM] = {0};
    float output[HIDDEN_DIM] = {0};
    forward(&net, input, output);
    return 0;
}
"""
while len(code) < 10000:
    code += "\n// filler: " + "A" * 80 + "\n"
code = code[:10000]
with open(os.path.join(OUT_DIR, "sample_code.c"), "w") as f:
    f.write(code)
print(f"[OK] sample_code.c: {len(code)} bytes")

# 5) Diretório com 10 arquivos mistos
for i in range(10):
    fname = os.path.join(OUT_DIR, "sample_dir", f"file_{i:02d}.dat")
    # mistura tipos
    if i % 3 == 0:
        # texto
        with open(fname, "w") as f:
            f.write(text[i*1000:(i+1)*1000])
    elif i % 3 == 1:
        # binário
        with open(fname, "wb") as f:
            f.write(binary[i*1000:(i+1)*1000])
    else:
        # repetitivo
        with open(fname, "wb") as f:
            f.write(b"X" * 1000)
total = sum(os.path.getsize(os.path.join(OUT_DIR, "sample_dir", f)) for f in os.listdir(os.path.join(OUT_DIR, "sample_dir")))
print(f"[OK] sample_dir/: 10 files, {total} bytes total")

print("\n=== Setup complete ===")
