# Black Hole v5 Roadmap

> **Target:** GPU acceleration, video compression, and game engine integration.

---

## v5.1: GPU Acceleration (CUDA Kernels)

### Objective
Train INRs on GPU instead of CPU. Current v4 takes ~25s per 128x128 texture on CPU. Target: <1s on GPU.

### Technical Approach
- PyTorch implementation of SIREN (replace numpy)
- CUDA kernels for forward/backward pass
- Batch processing of multiple textures
- Mixed precision (FP16) training

### Expected Impact
- 50x speedup in training
- Enable real-time encoding for video
- Unlock 512x512 and 1024x1024 textures

---

## v5.2: Video Compression (NeRV-Style Temporal INRs)

### Objective
Compress video sequences using temporal INRs. Instead of storing frames as images, store a single INR that evolves over time.

### Technical Approach
- Add time coordinate `t` to (x, y) → (x, y, t)
- Temporal SIREN: f(x, y, t) → RGB
- Keyframe INR + motion INR (like NeRV)
- Quantize temporal weights separately

### Expected Impact
- Video compression ratios of 100x-1000x
- Direct playback from INR (no frame decoding)
- Resolution-independent video

---

## v5.3: Game Engine Integration

### Objective
Plug BLKH into a real game engine (Unity, Unreal, or Godot).

### Technical Approach
- Plugin for texture loading at runtime
- Shader-based INR reconstruction (GPU inference)
- Background pre-calculation daemon
- Hybrid mode: INR for smooth textures, ZIP for complex details

### Expected Impact
- 10x smaller game installs
- Faster loading times
- Resolution-adaptive textures (LOD from INR)

---

## v5.4: DirectStorage / io_uring Integration

### Objective
True zero-copy ejection from INR to GPU memory.

### Technical Approach
- Kernel driver for INR recipe loading
- DirectStorage: GPU decompresses INR directly
- io_uring: async recipe loading
- DMA transfer from storage to GPU VRAM

### Expected Impact
- No CPU involvement in decompression
- VRAM-to-VRAM INR inference
- Console-level optimization (Xbox, PlayStation)

---

## v5.5: Multi-Resolution & LOD

### Objective
Generate textures at any resolution from a single INR recipe.

### Technical Approach
- Train INR once, query at any resolution
- Automatic LOD: closer = higher resolution query
- Mipmaps generated on-the-fly from INR
- No pre-computed mipmaps needed

### Expected Impact
- Infinite resolution scaling
- No texture memory bloat
- Perfect LOD transitions

---

## Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| v5.1 GPU | 2-3 weeks | PyTorch CUDA SIREN |
| v5.2 Video | 3-4 weeks | Temporal INR demo |
| v5.3 Engine | 4-6 weeks | Unity/Unreal plugin |
| v5.4 Kernel | 6-8 weeks | DirectStorage driver |
| v5.5 LOD | 2-3 weeks | Resolution-agnostic demo |

---

## Research Questions

1. Can we use Tensor Cores for SIREN inference?
2. How does temporal INR handle scene cuts?
3. What's the memory bandwidth tradeoff of INR vs traditional textures?
4. Can we pre-compute INR recipes on cloud and stream them?

---

*Document created by Darlan Pereira da Silva. Black Hole v5 planning.*
