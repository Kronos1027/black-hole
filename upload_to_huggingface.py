#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BLKH HuggingFace Spaces Uploader
=================================
Faz upload automático do BLKH para HuggingFace Spaces.

PRÉ-REQUISITOS:
1. Conta no HuggingFace (crie em https://huggingface.co/join)
2. Access Token (crie em https://huggingface.co/settings/tokens)
   - Clique em "New token"
   - Nome: "blkh-upload"
   - Tipo: "Write" (importante!)
   - Copie o token (começa com "hf_...")

USO:
    python upload_to_huggingface.py

O script vai:
1. Pedir seu token (cole quando solicitado)
2. Pedir seu username do HuggingFace
3. Criar o Space automaticamente
4. Fazer upload de todos os arquivos
5. Mostrar a URL final do app

Após o upload, o Space leva 2-5 minutos para buildar.
Depois disso, seu app estará público em:
    https://huggingface.co/spaces/SEU_USERNAME/black-hole-blkh

Author: Darlan Pereira da Silva (Kronos1027)
"""
import os
import sys
import shutil
from pathlib import Path


def main():
    print("=" * 70)
    print("🚀 BLKH HuggingFace Spaces Uploader")
    print("=" * 70)
    print()
    print("Este script vai subir o BLKH v5.30 para o HuggingFace Spaces.")
    print()
    print("PRÉ-REQUISITOS:")
    print("  1. Conta no HuggingFace (https://huggingface.co/join)")
    print("  2. Access Token com permissão 'Write'")
    print("     (crie em https://huggingface.co/settings/tokens)")
    print()

    # Verificar se huggingface_hub está instalado
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub não está instalado.")
        print("   Instale com: pip install huggingface_hub")
        sys.exit(1)

    # Pedir token
    print("-" * 70)
    token = input("📋 Cole seu HuggingFace Access Token (hf_...): ").strip()
    if not token.startswith("hf_"):
        print("⚠️  Token deve começar com 'hf_'. Verifique se copiou corretamente.")
        print("   Crie um token em: https://huggingface.co/settings/tokens")
        print("   Tipo: Write (não Read!)")
        sys.exit(1)

    # Pedir username
    username = input("👤 Seu username do HuggingFace: ").strip()
    if not username:
        print("❌ Username inválido.")
        sys.exit(1)

    # Nome do Space
    space_name = "black-hole-blkh"
    repo_id = f"{username}/{space_name}"

    print()
    print(f"📦 Criando Space: {repo_id}")
    print()

    # Verificar diretório huggingface
    script_dir = Path(__file__).parent
    hf_dir = script_dir / "huggingface"
    if not hf_dir.exists():
        # Tentar caminho alternativo
        hf_dir = script_dir.parent / "huggingface"
    if not hf_dir.exists():
        print(f"❌ Diretório 'huggingface/' não encontrado em {script_dir}")
        sys.exit(1)

    print(f"📂 Diretório fonte: {hf_dir}")
    print()

    # Login
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ Login efetuado com sucesso!")
    except Exception as e:
        print(f"❌ Erro no login: {e}")
        print("   Verifique se o token é válido e tem permissão 'Write'")
        sys.exit(1)

    # Criar repositório (Space)
    print(f"🔧 Criando Space '{space_name}'...")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,  # não falha se já existir
        )
        print(f"✅ Space criado: {repo_url}")
    except Exception as e:
        print(f"❌ Erro ao criar Space: {e}")
        sys.exit(1)

    # Upload de arquivos
    print()
    print("📤 Fazendo upload dos arquivos...")
    api = HfApi(token=token)

    files_to_upload = []
    for root, dirs, files in os.walk(hf_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f.startswith('.'):
                continue
            full_path = Path(root) / f
            rel_path = full_path.relative_to(hf_dir)
            files_to_upload.append((str(full_path), str(rel_path)))

    print(f"   {len(files_to_upload)} arquivos para upload")
    print()

    success = 0
    failed = 0
    for full_path, rel_path in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
            print(f"   ✅ {rel_path}")
            success += 1
        except Exception as e:
            print(f"   ❌ {rel_path}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"✅ Upload concluído: {success} arquivos enviados, {failed} falhas")
    print("=" * 70)
    print()
    print(f"🌐 Seu app estará disponível em:")
    print(f"   https://huggingface.co/spaces/{username}/{space_name}")
    print()
    print("⏳ O Space leva 2-5 minutos para buildar na primeira vez.")
    print("   Você pode acompanhar o status na página do Space.")
    print()
    print("📱 Para compartilhar:")
    print(f"   https://huggingface.co/spaces/{username}/{space_name}")
    print()
    print("🔄 Para atualizar no futuro, rode este script novamente.")
    print()
    print("🎉 Pronto! Seu BLKH está no HuggingFace!")


if __name__ == "__main__":
    main()
