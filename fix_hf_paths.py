# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Fix Windows backslash issue in HuggingFace Space.

Problem: Files uploaded from Windows have backslash in path names:
  phase1_inr_compressor\siren_v5_auto.py  (WRONG - treated as single file)

Solution: Delete wrong files and re-upload with forward slash:
  phase1_inr_compressor/siren_v5_auto.py  (CORRECT - file in folder)
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi


def main():
    print("=" * 70)
    print("🔧 BLKH Fix: Corrigir paths com backslash (Windows → Linux)")
    print("=" * 70)

    token = input("\n📋 Token (hf_...): ").strip()
    username = input("👤 Username: ").strip()
    repo_id = f"{username}/black-hole-blkh"

    api = HfApi(token=token)

    print(f"\n🔍 Listando arquivos no Space...")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="space")
    except Exception as e:
        print(f"❌ Erro ao listar: {e}")
        sys.exit(1)

    print(f"📊 Total de arquivos: {len(files)}")

    # Identificar arquivos com backslash
    wrong_files = [f for f in files if '\\' in f]
    print(f"❌ Arquivos com backslash (errados): {len(wrong_files)}")

    if wrong_files:
        print("\nArquivos errados encontrados:")
        for f in wrong_files[:5]:
            print(f"  {f}")
        if len(wrong_files) > 5:
            print(f"  ... e mais {len(wrong_files) - 5}")

        print("\n🗑️  Deletando arquivos errados...")
        for f in wrong_files:
            try:
                api.delete_file(
                    path_in_repo=f,
                    repo_id=repo_id,
                    repo_type="space",
                    token=token,
                )
                print(f"  ✅ Deletado: {f}")
            except Exception as e:
                print(f"  ❌ Erro ao deletar {f}: {e}")

    # Re-uploadar com paths corretos
    print("\n📤 Re-uploadando arquivos com paths corretos...")
    hf_dir = Path(__file__).parent / "huggingface"
    if not hf_dir.exists():
        print(f"❌ Pasta não encontrada: {hf_dir}")
        sys.exit(1)

    # Coletar arquivos
    upload_files = []
    for root, dirs, filenames in os.walk(hf_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in filenames:
            if fname.startswith('.') or fname.endswith('.pyc'):
                continue
            full = Path(root) / fname
            # IMPORTANTE: usar forward slash SEMPRE
            rel = str(full.relative_to(hf_dir)).replace('\\', '/')
            upload_files.append((str(full), rel))

    print(f"📊 Arquivos para upload: {len(upload_files)}")

    success = 0
    failed = 0
    for full, rel in upload_files:
        try:
            api.upload_file(
                path_or_fileobj=full,
                path_in_repo=rel,  # SEMPRE com /
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
            print(f"  ✅ {rel}")
            success += 1
        except Exception as e:
            print(f"  ❌ {rel}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"✅ Upload concluído: {success} sucesso, {failed} falhas")
    print("=" * 70)
    print()
    print(f"🌐 App: https://huggingface.co/spaces/{username}/black-hole-blkh")
    print()
    print("⏳ Aguarde 3-5 min para rebuildar.")
    print()
    print("🔍 Verifique se os arquivos estão com / (não \\):")
    print(f"   https://huggingface.co/spaces/{username}/black-hole-blkh/tree/main/phase1_inr_compressor")
    print()
    print("✅ Deve aparecer a pasta 'phase1_inr_compressor' (clicável)")
    print("   e DENTRO dela os arquivos .py")


if __name__ == "__main__":
    main()
