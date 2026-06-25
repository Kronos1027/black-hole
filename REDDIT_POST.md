# 📱 Reddit Post — BLKH v5.30

**Copy everything below this line and paste on Reddit:**

---

## 🕳️ I built a neural compression tool that's FASTER than ZIP and smaller than JPEG — try it free in your browser!

Hey everyone! 👋

I've been working on **Black Hole (BLKH)** — an open-source image compression tool that combines neural networks (SIREN) with traditional codecs. After months of development, I just deployed a live web demo and would love your feedback!

## 🎯 What makes it different

- **3-4x FASTER than ZIP** while being 6-13x smaller
- **2-7x smaller than JPEG** at similar quality
- **Competitive with AVIF** (the modern standard)
- Has a **TRUE bit-perfect lossless** mode (SHA-256 verified)

## 🌐 Try it now (no install needed)

**👉 https://huggingface.co/spaces/onatskyo/black-hole-blkh**

Just upload an image and it auto-picks the best compression mode. You'll see:
- Side-by-side comparison with ZIP, PNG, WebP
- PSNR and compression ratio
- Download the compressed file

## 🎛️ 8 compression modes

1. **Auto** (default) — smart-picks the best mode for your image
2. **Fast** — 3x faster than ZIP, great for real-time
3. **DCT** — JPEG-like, max compression (20-50x smaller than PNG)
4. **Photo** — beats PNG 2-4x on natural photos
5. **Wavelet v3** — TRUE bit-perfect lossless
6. **Hybrid (Instant/Turbo/Quality)** — SIREN neural network

## 📊 Some real numbers

On a 128×128 photo:
- ZIP: 36,237 bytes
- PNG: 25,507 bytes
- JPEG q=80: 1,657 bytes
- **BLKH DCT: 418 bytes** (86x smaller than ZIP, 4x smaller than JPEG!)

On CIFAR-10 (10,000 images, 32×32):
- ZIP: 26.4 MB
- **BLKH Fast: 4.3 MB** (6.1x smaller, 3x faster per image)

## 🔗 Links

- **Live demo**: https://huggingface.co/spaces/onatskyo/black-hole-blkh
- **GitHub**: https://github.com/Kronos1027/black-hole
- **License**: MIT (free for research/education)

## 🙏 What I need from you

1. **Test it** — upload your images, try different modes
2. **Report bugs** — GitHub issues welcome: https://github.com/Kronos1027/black-hole/issues
3. **Feedback** — what works, what doesn't, what's missing
4. **Ideas** — what compression features would you want?

## ⚠️ Honest limitations

- Lossless mode loses to PNG on natural photos (PNG's filtering is hard to beat)
- Audio mode loses to ZIP on speech (Shannon limit)
- GPU mode is written but not tested on actual CUDA hardware
- Small images (<64px) don't compress well (overhead dominates)

I'm a solo developer doing this as a research project. Any feedback, bug reports, or feature requests would mean a lot! 🙏

**Author**: Darlan Pereira da Silva (Kronos1027)

---

## 📋 Notas para você (NÃO copiar para o Reddit)

### Título alternativo (mais técnico):
> "I built BLKH v5.30 — a neural image compressor with 13 modes (SIREN + DCT + wavelet + AVIF). 3x faster than ZIP, 7x smaller than JPEG. Live demo inside."

### Título alternativo (mais clickbait):
> "I made a free image compressor that beats JPEG 7x and runs faster than ZIP. Try it in your browser — no install needed!"

### Melhores subreddits para postar (em ordem de relevância):

1. **r/MachineLearning** (2.5M members) — público técnico, vai dar feedback sério
   - ⚠️ Tem regras estritas, leia antes de postar
   - Use flair "Research" ou "Project"

2. **r/compsci** (1.2M) — ciência da computação geral
   - Bom para discussão técnica

3. **r/programming** (4.2M) — programadores em geral
   - Foco no "tool" e "open source"

4. **r/Python** (700K) — comunidade Python
   - Destaque que é Python puro

5. **r/opensource** (450K) — projetos open source
   - Foco no MIT license e colaboração

6. **r/dataisbeautiful** (22M) — se fizer gráficos bonitos
   - Poste visualizações dos resultados

7. **r/SideProject** (200K) — projetos paralelos
   - Comunidade acolhedora, bom para começar

8. **r/InternetIsBeautiful** (18M) — coisas legais na internet
   - Foco no "try it in browser"

### Estratégia de postagem recomendada:

1. **Comece pelo r/SideProject** (mais acolhedor, feedback inicial)
2. **Depois r/Python** (comunidade grande, técnico)
3. **Depois r/compsci** (discussão mais profunda)
4. **Por último r/MachineLearning** (mais exigente, precisa de resultados sólidos)

### Dicas para o post:

- ✅ **Poste entre 9-11h EST** (horário de pico dos EUA)
- ✅ **Responda TODO comentário** nos primeiros 2h (algoritmo favorece)
- ✅ **Tenha screenshots prontos** para responder perguntas
- ✅ **Seja humilde** — aceite críticas graciosamente
- ✅ **Não defenda** seu trabalho agressivamente — agradeça feedback

### O que NÃO fazer:

- ❌ Não poste em r/programming sem ler as regras
- ❌ Não faça cross-post imediato (espera 2-3 dias entre subs)
- ❌ Não defenda que "BLKH é melhor que X" — mostre números, deixe outros julgarem
- ❌ Não ignore críticas — use elas para melhorar
