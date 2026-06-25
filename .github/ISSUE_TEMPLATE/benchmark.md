---
name: "📊 Benchmark Result"
about: "Share your benchmark results with the community"
title: "[BENCHMARK] <dataset/image description>"
labels: ["benchmark", "data"]
assignees: []
---

## 📊 Benchmark Setup

<!-- Describe your benchmark setup -->

- **BLKH version**: [e.g., 5.30.0]
- **Image type**: [e.g., natural photo, satellite, medical, synthetic]
- **Image dimensions**: [e.g., 512x512]
- **Dataset**: [e.g., CIFAR-10, ImageNet, custom]
- **Number of images**: [e.g., 10000]
- **Hardware**: [e.g., Intel i7, 16GB RAM, no GPU]

## 🎛️ Modes Tested

<!-- Which modes did you benchmark? -->

- [ ] Auto v5.30
- [ ] Fast v5.23
- [ ] DCT v5.22 (q=0.9)
- [ ] DCT v5.22 (q=0.5)
- [ ] Photo v5.21
- [ ] Wavelet v3 (lossless)
- [ ] Hybrid
- [ ] Other: _________

## 📈 Results

<!-- Fill in the table with your results -->

| Mode | Avg Size | Avg PSNR | Avg Time | vs ZIP |
|------|----------|----------|----------|--------|
| ZIP (baseline) | _____B | N/A | _____ms | 1.00x |
| BLKH _____ | _____B | _____dB | _____ms | _____x |
| BLKH _____ | _____B | _____dB | _____ms | _____x |
| BLKH _____ | _____B | _____dB | _____ms | _____x |

## 🔍 Comparison with Other Tools (optional)

| Tool | Size | PSNR | Time |
|------|------|------|------|
| BLKH | _____B | _____dB | _____ms |
| JPEG | _____B | _____dB | _____ms |
| WebP | _____B | _____dB | _____ms |
| AVIF | _____B | _____dB | _____ms |
| PNG | _____B | lossless | _____ms |

## 📝 Notes

<!-- Any observations, surprises, or insights from the benchmark? -->

## 🔗 Reproducibility

<!-- How can others reproduce your benchmark? -->

```bash
# Commands to reproduce
```

## ➕ Additional Context

<!-- Add any charts, graphs, or additional data here. -->
