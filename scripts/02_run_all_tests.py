#!/usr/bin/env python3.13
"""
Bateria de testes completa do SIREN vs compressores tradicionais.
Testa:
  1) Texto em vários tamanhos (256B, 1KB, 4KB, 16KB, 50KB)
  2) Binário estruturado (50KB)
  3) Imagem (128x128 RGB)
  4) Diretório inteiro (10 arquivos mistos)

Para cada teste mede:
  - Tempo de compressão (treinamento para SIREN, encoding para tradicionais)
  - Tamanho comprimido
  - Razão de compressão
  - Tempo de descompressão
  - PSNR / erro para SIREN (lossy)
  - Lossless check para tradicionais
"""
import sys, os
sys.path.insert(0, "/home/z/my-project/scripts")
import time
import json
import zlib
import lzma
import gzip
import io
import math
import numpy as np
import torch
from PIL import Image
from importlib.machinery import SourceFileLoader
_siren = SourceFileLoader("siren_core", "/home/z/my-project/scripts/01_siren_core.py").load_module()
train_siren_1d = _siren.train_siren_1d
evaluate_siren_1d = _siren.evaluate_siren_1d
decode_time_siren = _siren.decode_time_siren
quantize_weights = _siren.quantize_weights
SirenMLP = _siren.SirenMLP

torch.manual_seed(42)
np.random.seed(42)

OUT = "/home/z/my-project/results"
os.makedirs(OUT, exist_ok=True)

results = {"tests": []}


def gzip_compress(data):
    return gzip.compress(data, compresslevel=9)

def gzip_decompress(data):
    return gzip.decompress(data)

def zlib_compress(data):
    return zlib.compress(data, 9)

def zlib_decompress(data):
    return zlib.decompress(data)

def lzma_compress(data):
    return lzma.compress(data, preset=9)

def lzma_decompress(data):
    return lzma.decompress(data)


def measure_traditional(name, raw, compress_fn, decompress_fn):
    """Roda compressor tradicional e mede tudo."""
    t0 = time.time()
    comp = compress_fn(raw)
    t_comp = time.time() - t0
    t0 = time.time()
    dec = decompress_fn(comp)
    t_dec = time.time() - t0
    lossless = (dec == raw)
    return {
        "method": name,
        "original_size": len(raw),
        "compressed_size": len(comp),
        "ratio": len(raw) / max(1, len(comp)),
        "compression_time_s": t_comp,
        "decompression_time_s": t_dec,
        "lossless": lossless,
        "throughput_MBps_decode": len(raw) / t_dec / 1e6 if t_dec > 0 else 0,
    }


def test_text_siren(raw, label, sizes_to_test=None):
    """Testa SIREN em texto, comparando com gzip/zlib/lzma."""
    if sizes_to_test is None:
        sizes_to_test = [256, 1024, 4096, 16384, 50000]
    out = []
    for size in sizes_to_test:
        if size > len(raw):
            continue
        data = raw[:size]
        # Normaliza bytes para [-1, 1]
        norm = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5

        # Configuração do SIREN escala com o tamanho - EPÓCAS REDUZIDAS P/ TEMPO
        if size <= 256:
            hf, hl, ep = 32, 2, 500
        elif size <= 1024:
            hf, hl, ep = 64, 3, 800
        elif size <= 4096:
            hf, hl, ep = 128, 3, 1200
        elif size <= 16384:
            hf, hl, ep = 256, 4, 1500
        else:
            hf, hl, ep = 512, 4, 2000

        print(f"\n--- {label} size={size} SIREN hf={hf} hl={hl} ep={ep} ---", flush=True)
        t0 = time.time()
        model, hist, t_train = train_siren_1d(norm, hidden_features=hf,
                                               hidden_layers=hl, epochs=ep,
                                               lr=1e-3, verbose=False)
        # Avalia
        pred = evaluate_siren_1d(model, len(norm))
        mse = float(np.mean((pred - norm) ** 2))
        psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

        # Tamanho dos pesos (8 bit e 16 bit)
        size_8bit = quantize_weights(model, 8)
        size_16bit = quantize_weights(model, 16)
        size_32bit = model.param_count() * 4

        # Tempo de decode
        t_decode = decode_time_siren(model, len(norm), repeats=3)

        # Lossless check: quantizar predição para uint8 e comparar
        pred_uint8 = np.clip(np.round((pred * 127.5) + 127.5), 0, 255).astype(np.uint8)
        # Calcular resíduo para lossless
        original_uint8 = np.frombuffer(data, dtype=np.uint8)
        residual = original_uint8.astype(np.int16) - pred_uint8.astype(np.int16)
        residual_bytes = residual.astype(np.int8).tobytes()  # 1 byte per residual
        # Comprimir resíduo
        residual_compressed = zlib.compress(residual_bytes, 9)
        lossless_size = size_8bit + len(residual_compressed)
        lossless_check = (pred_uint8 + residual.astype(np.int8).astype(np.uint8) == original_uint8).all()

        # Tradicionais
        gzip_r = measure_traditional("gzip", data, gzip_compress, gzip_decompress)
        zlib_r = measure_traditional("zlib", data, zlib_compress, zlib_decompress)
        lzma_r = measure_traditional("lzma", data, lzma_compress, lzma_decompress)

        siren_result = {
            "test": label,
            "type": "text",
            "original_size": size,
            "siren_lossy_8bit_size": size_8bit,
            "siren_lossy_16bit_size": size_16bit,
            "siren_lossy_32bit_size": size_32bit,
            "siren_lossy_ratio_8bit": size / max(1, size_8bit),
            "siren_lossless_size": lossless_size,
            "siren_lossless_ratio": size / max(1, lossless_size),
            "siren_train_time_s": t_train,
            "siren_decode_time_s": t_decode,
            "siren_mse": mse,
            "siren_psnr_db": psnr,
            "siren_param_count": model.param_count(),
            "siren_lossless_check": bool(lossless_check),
            "siren_residual_compressed_size": len(residual_compressed),
            "gzip": gzip_r,
            "zlib": zlib_r,
            "lzma": lzma_r,
        }
        out.append(siren_result)
        print(f"  SIREN lossy 8bit: {size_8bit}B  ratio={size/max(1,size_8bit):.3f}x  PSNR={psnr:.2f}dB", flush=True)
        print(f"  SIREN lossless:   {lossless_size}B  ratio={size/max(1,lossless_size):.3f}x  check={lossless_check}", flush=True)
        print(f"  gzip:  {gzip_r['compressed_size']}B  ratio={gzip_r['ratio']:.3f}x  decode={gzip_r['decompression_time_s']*1000:.2f}ms", flush=True)
        print(f"  lzma:  {lzma_r['compressed_size']}B  ratio={lzma_r['ratio']:.3f}x  decode={lzma_r['decompression_time_s']*1000:.2f}ms", flush=True)
    return out


def test_image_siren():
    """Testa SIREN em imagem RGB 128x128 - domínio onde INRs são mais promissores."""
    img = Image.open("/home/z/my-project/test_data/sample_image.png").convert("RGB")
    W, H = img.size
    arr = np.array(img).astype(np.float32) / 255.0 * 2 - 1  # [-1, 1]
    raw_size = W * H * 3  # raw bytes
    png_size = os.path.getsize("/home/z/my-project/test_data/sample_image.png")

    # Coordenadas normalizadas
    ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    coords = np.stack([xs.flatten(), ys.flatten()], axis=1).astype(np.float32)
    targets = arr.reshape(-1, 3)

    coords_t = torch.tensor(coords)
    targets_t = torch.tensor(targets)

    # Configurações SIREN - reduzidas para tempo
    configs = [
        ("small", 64, 3, 800),
        ("medium", 128, 4, 1200),
    ]
    out = []
    for name, hf, hl, ep in configs:
        print(f"\n--- image SIREN {name} hf={hf} hl={hl} ep={ep} ---")
        model = SirenMLP(in_features=2, out_features=3,
                          hidden_features=hf, hidden_layers=hl, omega_0=30.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
        t0 = time.time()
        for epoch in range(ep):
            opt.zero_grad()
            pred = model(coords_t)
            loss = torch.nn.functional.mse_loss(pred, targets_t)
            loss.backward()
            opt.step()
            sched.step()
            if epoch % max(1, ep // 5) == 0:
                print(f"  epoch {epoch:5d} | loss {loss.item():.6e}")
        t_train = time.time() - t0

        with torch.no_grad():
            pred = model(coords_t).cpu().numpy()
        mse = float(np.mean((pred - targets) ** 2))
        psnr = 10 * math.log10(4.0 / mse) if mse > 0 else float('inf')  # 4 = (max-min)^2

        size_8bit = quantize_weights(model, 8)
        size_16bit = quantize_weights(model, 16)

        # Decode time
        t0 = time.time()
        for _ in range(3):
            with torch.no_grad():
                _ = model(coords_t)
        t_decode = (time.time() - t0) / 3

        result = {
            "test": f"image_128x128_{name}",
            "type": "image",
            "original_size_raw": raw_size,
            "original_size_png": png_size,
            "siren_lossy_8bit_size": size_8bit,
            "siren_lossy_16bit_size": size_16bit,
            "siren_lossy_ratio_8bit_raw": raw_size / max(1, size_8bit),
            "siren_lossy_ratio_8bit_png": png_size / max(1, size_8bit),
            "siren_train_time_s": t_train,
            "siren_decode_time_s": t_decode,
            "siren_mse": mse,
            "siren_psnr_db": psnr,
            "siren_param_count": model.param_count(),
        }
        out.append(result)
        print(f"  SIREN lossy 8bit: {size_8bit}B  ratio_raw={raw_size/max(1,size_8bit):.3f}x  ratio_png={png_size/max(1,size_8bit):.3f}x  PSNR={psnr:.2f}dB")
        print(f"  PNG: {png_size}B  ratio_raw={raw_size/png_size:.3f}x")
    return out


def test_binary_siren():
    """Testa SIREN em binário estruturado (50KB)."""
    with open("/home/z/my-project/test_data/sample_binary.bin", "rb") as f:
        raw = f.read()
    return test_text_siren(raw, "binary_structured", sizes_to_test=[1024, 16384, 49995])


def test_directory():
    """Testa compressão de diretório inteiro (10 arquivos mistos, 10125 bytes total)."""
    dir_path = "/home/z/my-project/test_data/sample_dir"
    files = sorted(os.listdir(dir_path))
    raw_concat = b""
    file_boundaries = []
    for fn in files:
        with open(os.path.join(dir_path, fn), "rb") as f:
            content = f.read()
            file_boundaries.append((fn, len(raw_concat), len(raw_concat) + len(content)))
            raw_concat += content

    total_size = len(raw_concat)
    print(f"\n=== Directory test: {len(files)} files, {total_size} bytes total ===")

    # Tradicionais: comprimir tudo junto
    gzip_r = measure_traditional("gzip_dir", raw_concat, gzip_compress, gzip_decompress)
    lzma_r = measure_traditional("lzma_dir", raw_concat, lzma_compress, lzma_decompress)
    print(f"  gzip_dir: {gzip_r['compressed_size']}B  ratio={gzip_r['ratio']:.3f}x")
    print(f"  lzma_dir: {lzma_r['compressed_size']}B  ratio={lzma_r['ratio']:.3f}x")

    # SIREN lossless em concatenação
    norm = (np.frombuffer(raw_concat, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
    print(f"  Training SIREN on directory ({total_size} bytes)...")
    model, _, t_train = train_siren_1d(norm, hidden_features=256, hidden_layers=4,
                                        epochs=1500, lr=1e-3, verbose=False)
    pred = evaluate_siren_1d(model, len(norm))
    mse = float(np.mean((pred - norm) ** 2))
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

    size_8bit = quantize_weights(model, 8)
    pred_uint8 = np.clip(np.round((pred * 127.5) + 127.5), 0, 255).astype(np.uint8)
    original_uint8 = np.frombuffer(raw_concat, dtype=np.uint8)
    residual = original_uint8.astype(np.int16) - pred_uint8.astype(np.int16)
    residual_compressed = zlib.compress(residual.astype(np.int8).tobytes(), 9)
    lossless_size = size_8bit + len(residual_compressed)

    print(f"  SIREN lossy 8bit: {size_8bit}B  ratio={total_size/max(1,size_8bit):.3f}x  PSNR={psnr:.2f}dB")
    print(f"  SIREN lossless:   {lossless_size}B  ratio={total_size/max(1,lossless_size):.3f}x")

    return [{
        "test": "directory_mixed",
        "type": "directory",
        "num_files": len(files),
        "original_size": total_size,
        "siren_lossy_8bit_size": size_8bit,
        "siren_lossy_ratio": total_size / max(1, size_8bit),
        "siren_lossless_size": lossless_size,
        "siren_lossless_ratio": total_size / max(1, lossless_size),
        "siren_train_time_s": t_train,
        "siren_mse": mse,
        "siren_psnr_db": psnr,
        "gzip": gzip_r,
        "lzma": lzma_r,
    }]


if __name__ == "__main__":
    print("###### TEST 1: TEXT ######")
    with open("/home/z/my-project/test_data/sample_text.txt", "rb") as f:
        text_raw = f.read()
    results["tests"].extend(test_text_siren(text_raw, "text_pt"))

    print("\n###### TEST 2: BINARY ######")
    results["tests"].extend(test_binary_siren())

    print("\n###### TEST 3: IMAGE ######")
    results["tests"].extend(test_image_siren())

    print("\n###### TEST 4: DIRECTORY ######")
    results["tests"].extend(test_directory())

    # Salva resultados
    with open(os.path.join(OUT, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Resultados salvos em {OUT}/raw_results.json ===")
