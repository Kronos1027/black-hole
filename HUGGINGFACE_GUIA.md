# 🚀 Guia Completo: Subir BLKH para HuggingFace Spaces

**Autor**: Darlan Pereira da Silva (Kronos1027)
**Versão**: v5.30
**Tempo estimado**: 10-15 minutos

---

## 📋 Pré-requisitos

1. **Conta no HuggingFace** (gratuita)
2. **Access Token** com permissão "Write"
3. **BLKH** clonado do GitHub

---

## Passo 1: Criar conta no HuggingFace (se não tiver)

1. Acesse: https://huggingface.co/join
2. Preencha: nome, email, senha
3. Confirme o email (clique no link que receber)
4. Anote seu **username** (ex: `darlan1027`)

---

## Passo 2: Criar Access Token

1. Acesse: https://huggingface.co/settings/tokens
2. Clique em **"New token"**
3. Preencha:
   - **Name**: `blkh-upload`
   - **Type**: ⚠️ **Write** (NÃO escolha "Read" — precisa ser Write!)
4. Clique em **"Create"**
5. **COPIE o token** (começa com `hf_...`)
   - ⚠️ Guarde bem! Você não verá novamente.
   - Se perder, pode criar outro.

---

## Passo 3: Clonar o BLKH (se ainda não fez)

```bash
git clone https://github.com/Kronos1027/black-hole.git
cd black-hole
```

---

## Passo 4: Subir para HuggingFace (3 opções)

### Opção A: Script Automático (MAIS FÁCIL) ⭐

```bash
cd black-hole
python upload_to_huggingface.py
```

O script vai pedir:
1. Seu token (cole o `hf_...`)
2. Seu username do HuggingFace

Ele faz tudo automaticamente:
- ✅ Cria o Space
- ✅ Sobe todos os arquivos
- ✅ Mostra a URL final

---

### Opção B: Upload Manual via Site (sem código)

1. Acesse: https://huggingface.co/new-space
2. Preencha:
   - **Owner**: seu username
   - **Space name**: `black-hole-blkh`
   - **License**: MIT
   - **SDK**: Gradio
   - **Visibility**: Public
3. Clique em **"Create Space"**
4. Na página do Space, clique em **"Files"** → **"+ Add file"** → **"Upload files"**
5. Arraste todos os arquivos da pasta `huggingface/`:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - Pasta `phase1_inr_compressor/` (todos os .py)
6. Clique em **"Commit changes"**

---

### Opção C: Git (para quem conhece Git)

```bash
# 1. Clonar o Space vazio (substitua SEU_USERNAME)
git clone https://huggingface.co/spaces/SEU_USERNAME/black-hole-blkh
cd black-hole-blkh

# 2. Copiar arquivos do BLKH
cp -r ../black-hole/huggingface/* .

# 3. Commit e push
git add .
git commit -m "Initial BLKH v5.30 upload"
git push
```

Quando pedir senha, use seu **Access Token** (não a senha da conta).

---

## Passo 5: Aguardar Build (2-5 minutos)

Após o upload, o HuggingFace vai:
1. Instalar as dependências (do `requirements.txt`)
2. Buildar o app Gradio
3. Iniciar o servidor

Você pode acompanhar o status em:
- **"Logs"** na página do Space
- Quando mostrar **"Running"**, está pronto!

---

## Passo 6: Acessar seu App! 🎉

Sua app estará pública em:
```
https://huggingface.co/spaces/SEU_USERNAME/black-hole-blkh
```

Exemplo: `https://huggingface.co/spaces/darlan1027/black-hole-blkh`

---

## 🔄 Como Atualizar no Futuro

Se você fizer mudanças no código:

```bash
# Opção A: Script automático (recomendado)
python upload_to_huggingface.py

# Opção B: Git
cd black-hole-blkh
cp -r ../black-hole/huggingface/* .
git add .
git commit -m "Update to v5.XX"
git push
```

---

## 🐛 Problemas Comuns

### Erro: "Token does not have write permission"
- **Solução**: Recrie o token com tipo **"Write"** (não "Read")

### Erro: "Space not building"
- **Solução**: Verifique os logs na aba "Logs" do Space
- Pode ser dependência faltando no `requirements.txt`

### Erro: "Module not found"
- **Solução**: Verifique se todos os arquivos da pasta `phase1_inr_compressor/` foram subidos

### App carrega mas não comprime
- **Solução**: Veja os logs. Pode ser falta de memória (use imagens menores)
- HuggingFace Spaces free tem 16GB RAM

---

## 📞 Suporte

- **GitHub**: https://github.com/Kronos1027/black-hole
- **Email**: darlan1027pc@gmail.com
- **Issues**: https://github.com/Kronos1027/black-hole/issues

---

## 🎯 Resumo Rápido

```bash
# 1. Instalar huggingface_hub
pip install huggingface_hub

# 2. Rodar script (vai pedir token + username)
python upload_to_huggingface.py

# 3. Aguardar 2-5 min
# 4. Acessar: https://huggingface.co/spakes/SEU_USERNAME/black-hole-blkh
```

**Pronto! Seu BLKH estará no ar! 🕳️**
