# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Upload ALL files from huggingface/ directory to HuggingFace Space."""
import os
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🚀 BLKH Upload COMPLETO (todos os arquivos)")
    print("=" * 70)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Instale: pip install huggingface_hub")
        sys.exit(1)

    token = input("\n📋 Token (hf_...): ").strip()
    username = input("👤 Username: ").strip()
    repo_id = f"{username}/black-hole-blkh"

    print(f"\n📂 Procurando arquivos em huggingface/...")
    hf_dir = Path(__file__).parent / "huggingface"
    if not hf_dir.exists():
        print(f"❌ Pasta não encontrada: {hf_dir}")
        sys.exit(1)

    # Coletar TODOS os arquivos
    files = []
    for root, dirs, filenames in os.walk(hf_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in filenames:
            if fname.startswith('.') or fname.endswith('.pyc'):
                continue
            full = Path(root) / fname
            rel = full.relative_to(hf_dir)
            files.append((str(full), str(rel).replace('\\', '/')))

    print(f"📊 Encontrados: {len(files)} arquivos")
    print()

    api = HfApi(token=token)

    # Upload folder completo de uma vez (mais confiável)
    print("📤 Subindo TODOS os arquivos de uma vez...")
    try:
        api.upload_folder(
            folder_path=str(hf_dir),
            repo_id=repo_id,
            repo_type="space",
            token=token,
            commit_message="Upload completo BLKH v5.30",
        )
        print("✅ Upload concluído com SUCESSO!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        # Fallback: upload um por um
        print("\n📤 Tentando upload um por um...")
        success = 0
        failed = 0
        for full, rel in files:
            try:
                api.upload_file(
                    path_or_fileobj=full,
                    path_in_repo=rel,
                    repo_id=repo_id,
                    repo_type="space",
                    token=token,
                )
                print(f"  ✅ {rel}")
                success += 1
            except Exception as e2:
                print(f"  ❌ {rel}: {e2}")
                failed += 1
        print(f"\n✅ Sucesso: {success}, ❌ Falhas: {failed}")

    print()
    print("=" * 70)
    print(f"🌐 App: https://huggingface.co/spaces/{username}/black-hole-blkh")
    print("=" * 70)
    print()
    print("⏳ Aguarde 3-5 min para rebuildar.")
    print("📝 Verifique os arquivos em:")
    print(f"   https://huggingface.co/spaces/{username}/black-hole-blkh/tree/main")
    print()
    print("🔍 Confirme que phase1_inr_compressor/siren_v5_auto.py está lá!")


if __name__ == "__main__":
    main()
