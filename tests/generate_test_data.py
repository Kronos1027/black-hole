#!/usr/bin/env python3
"""
Black Hole - Generate Real-World Test Data
Creates sample files for benchmarking: images, text, audio, patterns, random.
"""
import numpy as np
import os
import random

def generate_all():
    test_dir = os.path.join(os.path.dirname(__file__), 'real_data')
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Synthetic image (16x16 RGB - small for fast testing)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            r = int((i / 15.0) * 255)
            g = int((j / 15.0) * 255)
            b = int(128 + 64 * np.sin(i * 0.5) * np.cos(j * 0.5))
            img[i, j] = [r, g, b]
    
    with open(os.path.join(test_dir, "test_image.raw"), "wb") as f:
        f.write(img.tobytes())
    
    # 2. Structured text
    text = "Black Hole Architecture. The data is a living mathematical function. " * 100
    with open(os.path.join(test_dir, "test_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    
    # 3. Audio-like signal (440Hz + 880Hz sine)
    sr = 8000
    duration = 0.1
    N = int(sr * duration)
    t = np.linspace(0, duration, N)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio_bytes = ((audio + 1.0) / 2.0 * 255).astype(np.uint8).tobytes()
    with open(os.path.join(test_dir, "test_audio.raw"), "wb") as f:
        f.write(audio_bytes)
    
    # 4. Random data (Kolmogorov limit)
    random.seed(42)
    rand_data = bytes([random.randint(0, 255) for _ in range(1000)])
    with open(os.path.join(test_dir, "test_random.bin"), "wb") as f:
        f.write(rand_data)
    
    # 5. Repeating pattern
    pattern = bytes([0, 255, 128, 64, 32, 16, 8, 4] * 125)
    with open(os.path.join(test_dir, "test_pattern.bin"), "wb") as f:
        f.write(pattern)
    
    print("Test data generated:")
    for f in sorted(os.listdir(test_dir)):
        size = os.path.getsize(os.path.join(test_dir, f))
        print(f"  {f}: {size} bytes")
    
    return test_dir

if __name__ == '__main__':
    generate_all()
