# 📦 Guia PyPI — Como publicar BLKH no PyPI

**Versão**: 5.30.0
**Pacote**: `blackhole-blkh`
**Status**: ✅ Build testado e aprovado

---

## ✅ Pré-requisitos (já feitos)

- [x] `pyproject.toml` configurado
- [x] `MANIFEST.in` criado
- [x] `LICENSE` presente
- [x] Build testado: `python -m build` ✅
- [x] Twine check: `python -m twine check dist/*` ✅ PASSED
- [x] Pacotes gerados em `dist/`

---

## 🚀 Passo a passo para publicar no PyPI

### Passo 1: Criar conta no PyPI (se não tiver)

1. Acesse: https://pypi.org/account/register/
2. Preencha: username, email, senha
3. Confirme o email (clique no link recebido)
4. **Anote seu username e senha**

### Passo 2: Criar API Token (MAIS SEGURO que senha)

1. Acesse: https://pypi.org/manage/account/token/
2. Clique em **"Add API token"**
3. Preencha:
   - **Scope**: "Entire account" (primeira vez) ou "Project" (depois)
   - **Token name**: `blkh-pypi`
4. Clique em **"Create token"**
5. **COPIE o token** (começa com `pypi-...`)
   - ⚠️ Guarde bem! Você não verá novamente.
   - NÃO compartilhe com ninguém (nem comigo!)

### Passo 3: Criar arquivo de configuração

Crie um arquivo `~/.pypirc` (no Windows: `C:\Users\SEU_USER\.pypirc`):

```ini
[pypi]
username = __token__
password = pypi-COLE_SEU_TOKEN_AQUI
```

⚠️ **NUNCA** commite este arquivo no git! Adicione ao `.gitignore`:
```
.pypirc
```

### Passo 4: Fazer o upload

No terminal, na pasta do projeto:

```bash
cd /home/z/my-project/blackhole_repo

# Upload para PyPI (produção)
python -m twine upload dist/*

# OU se não criou .pypirc, vai pedir token interativamente:
python -m twine upload dist/* -u __token__ -p pypi-COLE_SEU_TOKEN_AQUI
```

### Passo 5: Verificar publicação

1. Acesse: https://pypi.org/project/blackhole-blkh/
2. Deve aparecer seu pacote!
3. Teste instalação:
```bash
pip install blackhole-blkh
blkh --help
```

---

## 🔄 Como atualizar no futuro (nova versão)

### 1. Bump versão no `pyproject.toml`:
```toml
version = "5.31.0"  # mude para nova versão
```

### 2. Commit e push:
```bash
git add pyproject.toml
git commit -m "bump: version 5.31.0"
git push
```

### 3. Limpar build antigo:
```bash
rm -rf dist/ build/ *.egg-info
```

### 4. Build novo:
```bash
python -m build
```

### 5. Verificar:
```bash
python -m twine check dist/*
```

### 6. Upload:
```bash
python -m twine upload dist/*
```

---

## 🧪 Testar antes de publicar (TestPyPI)

Antes de publicar no PyPI oficial, teste no TestPyPI:

### 1. Criar conta no TestPyPI:
- https://test.pypi.org/account/register/

### 2. Criar token:
- https://test.pypi.org/manage/account/token/

### 3. Build (igual):
```bash
python -m build
```

### 4. Upload para TestPyPI:
```bash
python -m twine upload --repository testpypi dist/*
```

### 5. Testar instalação do TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ blackhole-blkh
```

---

## 📋 Comandos resumo (para copiar)

```bash
# === BUILD ===
cd /home/z/my-project/blackhole_repo
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine check dist/*

# === UPLOAD TESTPYPI (primeira vez) ===
python -m twine upload --repository testpypi dist/*

# === UPLOAD PYPI (oficial) ===
python -m twine upload dist/*

# === TESTAR ===
pip install blackhole-blkh
blkh --help
```

---

## 📦 O que está incluído no pacote

```
blackhole-blkh-5.30.0/
├── blkh.py                    # CLI principal
├── blkh_web_demo.py           # Web demo
├── phase1_inr_compressor/     # 37 módulos de compressão
│   ├── siren_v5_torch.py
│   ├── siren_v5_hybrid.py
│   ├── siren_v5_wavelet*.py
│   ├── siren_v5_photo.py
│   ├── siren_v5_dct.py
│   ├── siren_v5_fast.py
│   ├── siren_v5_rle.py
│   ├── siren_v5_palette.py
│   ├── siren_v5_auto.py
│   ├── siren_v5_avif.py
│   └── ... (37 módulos)
├── phase2_opportunistic_daemon/
├── phase3_ejection_engine/
├── game_engine/
└── dist-info/
    ├── METADATA
    ├── LICENSE
    └── entry_points.txt
```

**Tamanho**: 318KB (wheel) / 547KB (sdist)

---

## 🎯 Comandos disponíveis após install

```bash
blkh compress input.png output.blkh8     # Comprimir
blkh decompress input.blkh8 output.png   # Descomprimir
blkh wavelet3 input.png output.blkw3     # Wavelet v5.20
blkh photo input.png output.blkp         # Photo v5.21
blkh dct input.png output.blkd           # DCT v5.22
blkh fast input.png output.blkf          # Fast v5.23
blkh rle input.png output.blkr           # RLE v5.28
blkh palette input.png output.blkq       # Palette v5.29
blkh auto input.png output.blkh          # Auto v5.30
blkh avif input.png output.blhav         # AVIF v5.26
blkh batch input/ output/                # Batch v5.24
blkh info input.blkh8                    # Info do arquivo
blkh benchmark input.png                 # Benchmark
blkh doctor                              # Diagnóstico
```

---

## ⚠️ Importante

1. **NUNCA** faça upload da mesma versão duas vezes
   - PyPI não permite re-upload de mesma versão
   - Sempre bump a versão antes de novo upload

2. **Mantenha `.pypirc` privado**
   - Adicione ao `.gitignore`
   - NUNCA commite tokens

3. **Teste no TestPyPI primeiro**
   - Evita erros no PyPI oficial
   - Pode re-upload no TestPyPI

4. **Versão semântica**
   - `5.30.0` → `5.30.1` (patch: bug fix)
   - `5.30.0` → `5.31.0` (minor: nova feature)
   - `5.30.0` → `6.0.0` (major: breaking change)

---

## 📞 Próximos passos

1. **Criar conta PyPI**: https://pypi.org/account/register/
2. **Criar API token**: https://pypi.org/manage/account/token/
3. **Rodar**: `python -m twine upload dist/*`
4. **Verificar**: https://pypi.org/project/blackhole-blkh/
5. **Testar**: `pip install blackhole-blkh`

**Bom lançamento! 🚀**
