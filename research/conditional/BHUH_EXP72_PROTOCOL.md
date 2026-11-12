# BLKH Experiment 72 — four independent real RGB photo sources (preregistered)

**Preregistration committed before any new Exp72 measurements.** Research comparison, not a new codec invention or a production demo release.

## Sources and provenance

Use exactly the following **four independent original photographs** from the scikit-image standard data collection (`scikit-image==0.25.2`), with source description/licenses as noted by the [upstream scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html):

1. `astronaut`: NASA photograph of astronaut Eileen Collins, public domain.
2. `coffee`: photo of a coffee cup, CC0 by Rachel Michetti.
3. `chelsea`: photo of a cat, CC0 by Stefan van der Walt.
4. `rocket`: launch photograph, public-domain release, **external source holdout** relative to the first three in this experiment.

For each image, take the central **128×128 RGB crop** (no resampling, normalization of RGB integer pixels, or post-hoc crop selection); split into 16 nonoverlapping 32×32 tiles, and preserve the original 4×4 layout as a single **128×128 RGB atlas**. The 16 tiles within a photograph are NOT independent photos: four separate source photographs are the independence unit. Record full RGB source photo SHA-256 and exact cropped-atlas SHA-256, image library versions, and license provenance. No extra images may be silently substituted.

## Frozen neural conditions

A **genuine shared nonlinear conditional coordinate network**, one per source photograph. Inputs: position x,y mapped to [-1,1] and a separate learned latent vector for each of the 16 tiles; two sine-activated hidden layers with fixed frequency=10; 3-channel sigmoid RGB output. Frozen sizes: `(hidden, latent)=(32,8),(64,8),(96,12)`; Adam learning rate 0.003, 1500 training steps, 2048 random samples per step, seed `720072`. Train the model per source photo, storing every shared parameter and all tile embeddings as float16 under zlib plus the complete manifest and file header. **Only decode actual saved bytes in independent NumPy inference** to calculate final PSNR and mean per-tile *RGB* SSIM. Do not report PyTorch training PSNR as serialized quality.

## Strong conventional baselines and fixed win criteria

Each comparator gets the SAME 128×128 RGB atlas in its original spatial arrangement plus the same serialized source manifest:

- PNG lossless, WebP lossless, JPEG2000 reversible, JPEG lossy at quality `[5,20,40,60,80,95]`.
- WebP lossy at quality `[1,5,10,20,30,40,50,60,70,80,90,95,100]`.
- Actual byte sizes include manifest and encoded image bytes; decode all saved output first and confirm exact lossless format behavior.
- For each neural model, identify the smallest conventional atlas with BOTH **global RGB MSE <= neural RGB MSE** (equivalent to PSNR >= model) and **mean RGB per-tile SSIM >= model**. Report the neural size relative to that best compliant atlas, not to PNG alone.
- **High-quality gate:** decoded RGB PSNR >=30 dB AND mean RGB tile SSIM >=0.90 AND strictly smaller than quality-matched conventional atlas. Evaluate all four, but hold out the `rocket` source for external validation and **do not select hyperparameters after examining any Exp72 results**. A potential broader claim would require reproducible high-quality success in at least 3/4 *independent source photos*, including rocket; 16 crops per photograph do not increase independent n.

## Bias and limits

These are real photographs, **not sprites or textures created for the BLKH**; generalization to sprite artists, text, alpha channels, photos in other domains or training across scenes is NOT established. The source corpora are small and model parameters are optimized against each source's own pixels (valid for compression, NOT independent predictive generalization). This pilot tests a plausible use case of shared learned coordinates with honest rate–distortion and atlas baselines. Negative results should be published exactly and should guide a pivot, not retrospective re-tuning.

**No `main` merge, demo update, breakthrough or LinkedIn marketing claim** based solely on Exp72.
