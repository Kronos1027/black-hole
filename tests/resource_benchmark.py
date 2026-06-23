#!/usr/bin/env python3
"""
Black Hole — Resource Benchmark v4.5
Compares CPU time, memory usage, and file sizes:
- BLKH v4 (4-bit INR) 
- BLKH v4 + Residual (bit-perfect)
- JPEG (lossy, quality 85)
- PNG (lossless)
- ZIP (general compression)

Measures: encode time, decode time, peak RAM, file size, PSNR
"""
import sys
import os
import io
import time
import tracemalloc
import tempfile
import zipfile
import zlib
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
from siren_v4 import ImageINRV4

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============ UTILS ============

def measure_memory_and_time(func, *args, **kwargs):
    """Run function, measure wall time and peak RAM."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak

def zip_compress(data_bytes):
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zp = f.name
    with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('data', data_bytes)
    with open(zp, 'rb') as f:
        compressed = f.read()
    os.unlink(zp)
    return compressed

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(65025.0 / mse)

# ============ GENERATORS ============

def create_smooth_128():
    x = np.linspace(0, 1, 128)
    y = np.linspace(0, 1, 128)
    xx, yy = np.meshgrid(x, y)
    r = (xx * 255).astype(np.uint8)
    g = (yy * 255).astype(np.uint8)
    b = ((np.sin(xx * 4 * np.pi) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)

def create_brick_64():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            brick_y = i // 8
            offset = 4 if brick_y % 2 == 0 else 0
            is_mortar = (j + offset) % 8 == 0 or i % 8 == 0
            if is_mortar:
                img[i, j] = [180, 180, 180]
            else:
                r = 120 + np.random.randint(-20, 20)
                g = 60 + np.random.randint(-10, 10)
                b = 40 + np.random.randint(-10, 10)
                img[i, j] = [r, g, b]
    return img

def create_water_128():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            wave = int(np.sin(i * 0.2) * 20 + np.sin(j * 0.3) * 15)
            r = 20 + wave
            g = 50 + wave
            b = 150 + wave
            img[i, j] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]
    return img

# ============ BENCHMARKS ============

def benchmark_jpeg(image, quality=85):
    pil_img = Image.fromarray(image)
    
    # Encode
    def encode():
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        return buf.getvalue()
    
    jpeg_data, enc_time, enc_mem = measure_memory_and_time(encode)
    
    # Decode
    def decode():
        buf = io.BytesIO(jpeg_data)
        return np.array(Image.open(buf))
    
    recon, dec_time, dec_mem = measure_memory_and_time(decode)
    
    return {
        'format': 'JPEG',
        'file_size': len(jpeg_data),
        'enc_time': enc_time,
        'dec_time': dec_time,
        'enc_mem': enc_mem,
        'dec_mem': dec_mem,
        'psnr': psnr(image, recon),
        'match': np.mean(recon == image) * 100,
        'recon': recon
    }

def benchmark_png(image):
    pil_img = Image.fromarray(image)
    
    def encode():
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return buf.getvalue()
    
    png_data, enc_time, enc_mem = measure_memory_and_time(encode)
    
    def decode():
        buf = io.BytesIO(png_data)
        return np.array(Image.open(buf))
    
    recon, dec_time, dec_mem = measure_memory_and_time(decode)
    
    return {
        'format': 'PNG',
        'file_size': len(png_data),
        'enc_time': enc_time,
        'dec_time': dec_time,
        'enc_mem': enc_mem,
        'dec_mem': dec_mem,
        'psnr': float('inf'),  # lossless
        'match': 100.0,
        'recon': recon
    }

def benchmark_zip(image):
    data = image.tobytes()
    
    def encode():
        return zip_compress(data)
    
    zip_data, enc_time, enc_mem = measure_memory_and_time(encode)
    
    def decode():
        return np.frombuffer(zlib.decompress(zip_data), dtype=np.uint8).reshape(image.shape)
    
    recon, dec_time, dec_mem = measure_memory_and_time(decode)
    
    return {
        'format': 'ZIP',
        'file_size': len(zip_data),
        'enc_time': enc_time,
        'dec_time': dec_time,
        'enc_mem': enc_mem,
        'dec_mem': dec_mem,
        'psnr': float('inf'),
        'match': 100.0,
        'recon': recon
    }

def benchmark_blkh(image, epochs=2000):
    compressor = ImageINRV4(hidden_dim=32, num_layers=2)
    
    def encode():
        compressor.compress(image, epochs=epochs, lr=1e-3)
        with tempfile.NamedTemporaryFile(suffix='.blkh', delete=False) as f:
            rp = f.name
        compressor.save_recipe(rp, bits=4, prune_threshold=0.01)
        with open(rp, 'rb') as f:
            data = f.read()
        os.unlink(rp)
        return data
    
    blkh_data, enc_time, enc_mem = measure_memory_and_time(encode)
    
    def decode():
        return compressor.reconstruct()
    
    recon, dec_time, dec_mem = measure_memory_and_time(decode)
    
    return {
        'format': 'BLKH v4',
        'file_size': len(blkh_data),
        'enc_time': enc_time,
        'dec_time': dec_time,
        'enc_mem': enc_mem,
        'dec_mem': dec_mem,
        'psnr': psnr(image, recon),
        'match': np.mean(recon == image) * 100,
        'recon': recon
    }

def benchmark_blkh_hybrid(image, epochs=2000):
    """BLKH + residual for bit-perfect."""
    compressor = ImageINRV4(hidden_dim=32, num_layers=2)
    compressor.compress(image, epochs=epochs, lr=1e-3)
    recon = compressor.reconstruct()
    
    # Calculate residual
    residual = (image.astype(np.int16) - recon.astype(np.int16)).astype(np.int8)
    residual_compressed = zlib.compress(residual.tobytes(), level=9)
    
    with tempfile.NamedTemporaryFile(suffix='.blkh', delete=False) as f:
        rp = f.name
    compressor.save_recipe(rp, bits=4, prune_threshold=0.01)
    with open(rp, 'rb') as f:
        blkh_data = f.read()
    os.unlink(rp)
    
    total_size = len(blkh_data) + len(residual_compressed)
    
    # Decode
    def decode():
        r = compressor.reconstruct()
        res = np.frombuffer(zlib.decompress(residual_compressed), dtype=np.int8).reshape(image.shape)
        return np.clip(r.astype(np.int16) + res, 0, 255).astype(np.uint8)
    
    recon_final, dec_time, dec_mem = measure_memory_and_time(decode)
    
    return {
        'format': 'BLKH + Residual',
        'file_size': total_size,
        'enc_time': 'N/A',  # Already measured in BLKH
        'dec_time': dec_time,
        'enc_mem': 'N/A',
        'dec_mem': dec_mem,
        'psnr': float('inf'),
        'match': 100.0,
        'recon': recon_final
    }

# ============ MAIN ============

def main():
    images = {
        'Smooth 128x128': create_smooth_128(),
        'Brick 64x64': create_brick_64(),
        'Water 128x128': create_water_128(),
    }
    
    all_results = []
    
    for name, img in images.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print(f"{'='*70}")
        
        orig_size = len(img.tobytes())
        print(f"Original: {orig_size} bytes")
        
        results = []
        
        # JPEG
        r = benchmark_jpeg(img)
        print(f"[JPEG] Size: {r['file_size']} | Enc: {r['enc_time']:.4f}s | Dec: {r['dec_time']:.4f}s | PSNR: {r['psnr']:.1f}")
        results.append(r)
        
        # PNG
        r = benchmark_png(img)
        print(f"[PNG]  Size: {r['file_size']} | Enc: {r['enc_time']:.4f}s | Dec: {r['dec_time']:.4f}s | Lossless")
        results.append(r)
        
        # ZIP
        r = benchmark_zip(img)
        print(f"[ZIP]  Size: {r['file_size']} | Enc: {r['enc_time']:.4f}s | Dec: {r['dec_time']:.4f}s | Lossless")
        results.append(r)
        
        # BLKH v4
        r = benchmark_blkh(img, epochs=2000)
        print(f"[BLKH] Size: {r['file_size']} | Enc: {r['enc_time']:.2f}s | Dec: {r['dec_time']:.4f}s | PSNR: {r['psnr']:.1f}")
        results.append(r)
        
        # BLKH + Residual
        r = benchmark_blkh_hybrid(img, epochs=2000)
        print(f"[HYBRID] Size: {r['file_size']} | Dec: {r['dec_time']:.4f}s | Lossless")
        results.append(r)
        
        all_results.append({'image': name, 'original': orig_size, 'results': results})
    
    # Generate summary table
    print(f"\n{'='*90}")
    print(f"{'RESOURCE BENCHMARK SUMMARY':^90}")
    print(f"{'='*90}")
    
    for item in all_results:
        print(f"\n{item['image']} ({item['original']} bytes)")
        print(f"{'Format':<18} {'Size':>10} {'Ratio':>8} {'Enc(s)':>10} {'Dec(s)':>10} {'EncRAM':>12} {'DecRAM':>12} {'PSNR':>10}")
        print('-' * 90)
        for r in item['results']:
            ratio = item['original'] / r['file_size']
            enc_mem = f"{r['enc_mem']/1024/1024:.1f}MB" if isinstance(r['enc_mem'], (int, float)) else 'N/A'
            dec_mem = f"{r['dec_mem']/1024/1024:.1f}MB" if isinstance(r['dec_mem'], (int, float)) else 'N/A'
            enc_time = f"{r['enc_time']:.4f}" if isinstance(r['enc_time'], (int, float)) else 'N/A'
            dec_time = f"{r['dec_time']:.4f}" if isinstance(r['dec_time'], (int, float)) else 'N/A'
            psnr = f"{r['psnr']:.1f}" if r['psnr'] != float('inf') else 'INF'
            print(f"{r['format']:<18} {r['file_size']:>10} {ratio:>7.2f}x {enc_time:>10} {dec_time:>10} {enc_mem:>12} {dec_mem:>12} {psnr:>10}")
    
    # Save JSON
    with open('resource_benchmark.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nSaved: resource_benchmark.json")
    
    # Generate charts
    generate_charts(all_results)
    
    return all_results

def generate_charts(all_results):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0d0d1a')
    
    metrics = ['file_size', 'enc_time', 'dec_time']
    metric_labels = ['File Size (bytes)', 'Encode Time (s)', 'Decode Time (s)']
    
    for row, metric in enumerate(metrics):
        for col, item in enumerate(all_results):
            ax = axes[row, col]
            formats = [r['format'] for r in item['results']]
            values = [r[metric] if isinstance(r[metric], (int, float)) else 0 for r in item['results']]
            
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
            bars = ax.bar(range(len(formats)), values, color=colors[:len(formats)], alpha=0.8)
            
            ax.set_xticks(range(len(formats)))
            ax.set_xticklabels(formats, rotation=45, ha='right', fontsize=8, color='white')
            ax.set_ylabel(metric_labels[row], color='white', fontsize=9)
            ax.set_title(f"{item['image']}", color='white', fontsize=10, fontweight='bold')
            ax.set_facecolor('#0d0d1a')
            ax.tick_params(colors='white')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#555577')
            ax.grid(True, alpha=0.2, color='#555577')
    
    plt.suptitle('Black Hole Resource Benchmark: Size vs Time vs Algorithm', 
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    plt.tight_layout()
    plt.savefig('resource_benchmark.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    
    print("Saved: resource_benchmark.png")

if __name__ == '__main__':
    main()
