#!/usr/bin/env python3.13
"""
SIREN - Implicit Neural Representation para compressão de dados.
Implementação baseada em Sitzmann et al. 2020 (https://arxiv.org/abs/2006.09661).

A ideia central: dado um sinal f(x), em vez de armazenar f, armazenamos os pesos
de uma rede neural g(x; theta) que aproxima f. Se theta for menor que f, houve
compressão (com perda).

Para compressão SEM PERDAS (lossless), precisamos quantizar theta + armazenar
o resíduo (f - g(x; theta)) para reconstrução exata.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import json

torch.manual_seed(42)
np.random.seed(42)

# Usa CPU explicitamente (sem GPU disponível neste ambiente)
device = torch.device("cpu")


class SirenLayer(nn.Module):
    """Camada SIREN com inicialização específica para ativação senoidal."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Primeira camada: distribuição uniforme com borne maior
                bound = 1.0 / self.in_features
            else:
                # Camadas subsequentes: borne menor para preservar gradiente
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """MLP SIREN: in_features -> hidden -> hidden -> ... -> out_features"""
    def __init__(self, in_features=1, out_features=1, hidden_features=32,
                 hidden_layers=2, omega_0=30.0):
        super().__init__()
        layers = [SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features, omega_0=omega_0))
        layers.append(nn.Linear(hidden_features, out_features))  # camada final linear
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_bytes(self, dtype_bits=32):
        """Tamanho em bytes dos pesos, com quantização de dtype_bits bits por peso."""
        return self.param_count() * dtype_bits // 8


def train_siren_1d(data, hidden_features=32, hidden_layers=2, omega_0=30.0,
                   epochs=2000, lr=1e-3, verbose=True):
    """
    Treina SIREN para mapear índice i -> byte value data[i].
    data: numpy array 1D de floats normalizadas em [-1, 1].
    Retorna: modelo treinado, histórico de loss, tempo de treinamento.
    """
    N = len(data)
    # Coordenadas: posicionar em [-1, 1]
    coords = torch.linspace(-1, 1, N, device=device).unsqueeze(1)  # (N, 1)
    targets = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(1)

    model = SirenMLP(in_features=1, out_features=1,
                     hidden_features=hidden_features,
                     hidden_layers=hidden_layers,
                     omega_0=omega_0).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start = time.time()
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            history.append((epoch, loss.item()))
            print(f"  epoch {epoch:5d} | loss {loss.item():.6e}")
    elapsed = time.time() - start
    return model, history, elapsed


def evaluate_siren_1d(model, N):
    """Avalia SIREN em N pontos, retorna numpy array."""
    coords = torch.linspace(-1, 1, N, device=device).unsqueeze(1)
    with torch.no_grad():
        pred = model(coords).cpu().numpy().flatten()
    return pred


def decode_time_siren(model, N, repeats=5):
    """Mede tempo de decodificação (repeats vezes, retorna média)."""
    coords = torch.linspace(-1, 1, N, device=device).unsqueeze(1)
    times = []
    for _ in range(repeats):
        t0 = time.time()
        with torch.no_grad():
            _ = model(coords)
        times.append(time.time() - t0)
    return np.mean(times)


def quantize_weights(model, bits=8):
    """
    Quantiza pesos para 'bits' bits por peso usando min-max quantization.
    Retorna bytes estimados.
    """
    all_w = torch.cat([p.data.flatten() for p in model.parameters()])
    w_min, w_max = all_w.min().item(), all_w.max().item()
    n_levels = (1 << bits) - 1
    scale = (w_max - w_min) / n_levels if w_max > w_min else 1.0
    # Tamanho: param_count * bits / 8 + overhead (2 floats p/ min/max)
    param_count = sum(p.numel() for p in model.parameters())
    return param_count * bits // 8 + 8  # +8 bytes para min/max float32


if __name__ == "__main__":
    # Teste rápido: treinar SIREN em 256 bytes do texto
    with open("/home/z/my-project/test_data/sample_text.txt", "rb") as f:
        raw = f.read(256)
    data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
    print(f"Input: {len(data)} bytes")
    model, hist, t = train_siren_1d(data, hidden_features=32, hidden_layers=2,
                                     epochs=500, lr=1e-3, verbose=True)
    print(f"\nTreinamento: {t:.2f}s, params: {model.param_count()}, "
          f"pesos @8bit: {quantize_weights(model, 8)} bytes")
    pred = evaluate_siren_1d(model, len(data))
    mse = np.mean((pred - data) ** 2)
    print(f"MSE final: {mse:.6e}")
    print(f"PSNR (vs original normalizado): {10*math.log10(1.0/mse):.2f} dB")
