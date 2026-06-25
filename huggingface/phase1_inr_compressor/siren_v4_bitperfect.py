"""
siren_v4_bitperfect.py — Extensão do v4 com camada residual bit-perfect
=====================================================================
Permite roundtrip 100% idêntico ao original (verificado por SHA-256),
combinando a compressão neural do SIREN com um residual XOR comprimido.

Pipeline:
    1. Treina SIREN v4 normalmente (lossy)
    2. Roda inferência → bytes previstos
    3. Calcula residual = original XOR predicted
    4. zlib-compress o residual
    5. Salva recipe = pesos quantizados + residual comprimido

Decompressão:
    1. Carrega pesos + residual
    2. Inferência → bytes previstos
    3. XOR(predicted, residual) → original exato
    4. Verifica SHA-256

Trade-off: o residual cresce com a taxa de erro do SIREN. Para sinais suaves
(o caso onde SIREN brilha), o residual é pequeno e comprime bem. Para sinais
caóticos, o residual ≈ tamanho original e BLKH perde para ZIP.
"""
import os
import sys
import zlib
import struct
import hashlib
import time
import numpy as np

# Reusa tudo do v4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v4 import ImageINRV4, SIREN2DV4


# -------- residual helpers --------
def compute_residual(original_bytes: bytes, predicted_bytes: bytes) -> bytes:
    """XOR byte-a-byte. Mesmo comprimento."""
    assert len(original_bytes) == len(predicted_bytes), \
        f"length mismatch: {len(original_bytes)} vs {len(predicted_bytes)}"
    a = np.frombuffer(original_bytes, dtype=np.uint8)
    b = np.frombuffer(predicted_bytes, dtype=np.uint8)
    return (a ^ b).tobytes()


def apply_residual(predicted_bytes: bytes, residual_bytes: bytes) -> bytes:
    """Reverte o XOR."""
    a = np.frombuffer(predicted_bytes, dtype=np.uint8)
    b = np.frombuffer(residual_bytes, dtype=np.uint8)
    assert len(a) == len(b)
    return (a ^ b).tobytes()


# -------- formato binário bit-perfect --------
# Layout:
#   [4B  magic 'BLKB']
#   [1B  version = 1]
#   [4B  H]
#   [4B  W]
#   [4B  C]
#   [4B  bits]                    # 4 ou 8 (do SIREN)
#   [4B  prune_threshold_x10000] # float em int
#   [4B  num_layers_siren]
#   [4B  hidden_dim]
#   [4B  omega_0_x1000]           # float em int
#   [4B  siren_recipe_size]
#   [siren_recipe_size bytes]     # recipe v4 empacotado
#   [8B  residual_compressed_size]
#   [residual_compressed_size bytes]
#   [8B  sha256_original (32B actually)]
#   [32B sha256]
MAGIC = b'BLKB'
VERSION = 1


class ImageINRV4BitPerfect:
    """v4 + camada residual XOR → 100% bit-perfect roundtrip."""

    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega_0 = omega_0
        self.H = None
        self.W = None
        self.C = None
        self._compressor = None

    def _make_compressor(self):
        return ImageINRV4(hidden_dim=self.hidden_dim,
                          num_layers=self.num_layers,
                          omega_0=self.omega_0)

    def compress(self, image_array: np.ndarray,
                 epochs: int = 2000, lr: float = 1e-3,
                 bits: int = 4, prune_threshold: float = 0.01,
                 zlib_level: int = 9,
                 verbose: bool = False) -> dict:
        """
        Comprime imagem para recipe bit-perfect.
        image_array: (H, W, C) uint8.
        """
        assert image_array.dtype == np.uint8, "imagem deve ser uint8"
        assert image_array.ndim == 3, "shape deve ser (H, W, C)"

        H, W, C = image_array.shape
        self.H, self.W, self.C = H, W, C
        original_bytes = image_array.tobytes()

        # 1. Treina SIREN v4 (lossy)
        t0 = time.time()
        self._compressor = self._make_compressor()
        meta = self._compressor.compress(image_array, epochs=epochs, lr=lr)
        train_time = time.time() - t0

        # 2. Salva recipe v4 em arquivo temp e lê de volta
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.blkh', delete=False) as f:
            tmp_path = f.name
        try:
            self._compressor.save_recipe(tmp_path, bits=bits,
                                          prune_threshold=prune_threshold)
            with open(tmp_path, 'rb') as f:
                siren_recipe = f.read()
        finally:
            os.unlink(tmp_path)

        # 3. Recarrega o recipe em um SIREN FRESCO — CRÍTICO!
        # O residual deve ser calculado contra o modelo que será usado na
        # descompressão (ou seja, com pesos quantizados/dequantizados).
        # Se usarmos o modelo pré-save (pesos FP32), o residual não vai bater.
        reload_comp = self._make_compressor()
        reload_comp.H, reload_comp.W = H, W
        with tempfile.NamedTemporaryFile(suffix='.blkh', delete=False) as f:
            tmp_path2 = f.name
        try:
            with open(tmp_path2, 'wb') as f:
                f.write(siren_recipe)
            reload_comp.load_recipe(tmp_path2)
        finally:
            os.unlink(tmp_path2)

        # 4. Inferência com modelo recarregado (quantizado)
        t0 = time.time()
        predicted_img = reload_comp.reconstruct()
        predict_time = time.time() - t0
        predicted_bytes = predicted_img.tobytes()

        # 4. Residual XOR
        residual = compute_residual(original_bytes, predicted_bytes)
        residual_compressed = zlib.compress(residual, zlib_level)

        # 5. SHA-256 do original para verificação
        sha = hashlib.sha256(original_bytes).digest()

        # 6. Pack tudo
        recipe = self._pack_recipe(H, W, C, bits, prune_threshold,
                                    siren_recipe, residual_compressed, sha)

        # 7. Stats
        orig_size = len(original_bytes)
        siren_size = len(siren_recipe)
        resid_size = len(residual_compressed)
        resid_raw = len(residual)
        # bit accuracy do modelo puro (antes do residual)
        bit_acc = float(np.mean(np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8)) ==
                                np.unpackbits(np.frombuffer(predicted_bytes, dtype=np.uint8)))) * 100

        return {
            'recipe_bytes': recipe,
            'original_size': orig_size,
            'recipe_size': len(recipe),
            'siren_recipe_size': siren_size,
            'residual_compressed_size': resid_size,
            'residual_raw_size': resid_raw,
            'model_bit_accuracy': bit_acc,
            'train_time_s': train_time,
            'predict_time_s': predict_time,
            'psnr_db': meta['psnr'],
            'sha256': sha.hex(),
        }

    def _pack_recipe(self, H, W, C, bits, prune_threshold,
                     siren_recipe, residual_compressed, sha) -> bytes:
        out = bytearray()
        out += MAGIC
        out += struct.pack('<B', VERSION)
        out += struct.pack('<I', H)
        out += struct.pack('<I', W)
        out += struct.pack('<I', C)
        out += struct.pack('<I', bits)
        out += struct.pack('<i', int(prune_threshold * 10000))
        out += struct.pack('<I', self.num_layers)
        out += struct.pack('<I', self.hidden_dim)
        out += struct.pack('<i', int(self.omega_0 * 1000))
        out += struct.pack('<I', len(siren_recipe))
        out += siren_recipe
        out += struct.pack('<Q', len(residual_compressed))
        out += residual_compressed
        out += sha  # 32 bytes
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes) -> tuple[np.ndarray, dict]:
        """Decodifica recipe bit-perfect. Retorna (imagem_reconstruída, meta)."""
        buf = recipe_bytes
        offset = 0
        if buf[:4] != MAGIC:
            raise ValueError(f"magic inválido: {buf[:4]!r}")
        offset += 4
        version = struct.unpack('<B', buf[offset:offset+1])[0]; offset += 1
        assert version == VERSION, f"versão não suportada: {version}"
        H = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        W = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        C = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        bits = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        prune_threshold = struct.unpack('<i', buf[offset:offset+4])[0] / 10000.0; offset += 4
        num_layers = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        hidden_dim = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        omega_0 = struct.unpack('<i', buf[offset:offset+4])[0] / 1000.0; offset += 4
        siren_size = struct.unpack('<I', buf[offset:offset+4])[0]; offset += 4
        siren_recipe = buf[offset:offset+siren_size]; offset += siren_size
        resid_size = struct.unpack('<Q', buf[offset:offset+8])[0]; offset += 8
        residual_compressed = buf[offset:offset+resid_size]; offset += resid_size
        sha = buf[offset:offset+32]; offset += 32

        # Recria o compressor SIREN
        comp = ImageINRV4(hidden_dim=hidden_dim, num_layers=num_layers, omega_0=omega_0)
        comp.H, comp.W = H, W
        # salva recipe v4 em arquivo temp e carrega
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.blkh', delete=False) as f:
            f.write(siren_recipe)
            tmp_path = f.name
        try:
            comp.load_recipe(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Inferência
        predicted_img = comp.reconstruct()
        predicted_bytes = predicted_img.tobytes()

        # Aplica residual
        residual = zlib.decompress(residual_compressed)
        if len(residual) != len(predicted_bytes):
            raise ValueError(f"residual len {len(residual)} != predicted {len(predicted_bytes)}")
        recovered_bytes = apply_residual(predicted_bytes, residual)

        # Verifica SHA-256
        recovered_sha = hashlib.sha256(recovered_bytes).digest()
        sha_match = (recovered_sha == sha)

        meta = {
            'H': H, 'W': W, 'C': C,
            'bits': bits,
            'prune_threshold': prune_threshold,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'omega_0': omega_0,
            'siren_recipe_size': siren_size,
            'residual_compressed_size': resid_size,
            'sha256_expected': sha.hex(),
            'sha256_recovered': recovered_sha.hex(),
            'sha256_match': sha_match,
            'exact_match': sha_match,
        }
        return np.frombuffer(recovered_bytes, dtype=np.uint8).reshape(H, W, C), meta

    @staticmethod
    def save_recipe(recipe_bytes: bytes, path: str) -> None:
        with open(path, 'wb') as f:
            f.write(recipe_bytes)

    @staticmethod
    def load_recipe(path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()


# -------- CLI rápido --------
def _self_test():
    """Auto-teste rápido do bit-perfect."""
    print("[bit-perfect] Self-test em imagem sintética 32x32...")
    # gradiente suave (caso favorável)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        for j in range(32):
            img[i, j] = [int(i*8), int(j*8), int((i+j)*4)]

    comp = ImageINRV4BitPerfect(hidden_dim=32, num_layers=2, omega_0=30.0)
    res = comp.compress(img, epochs=500, lr=1e-3, bits=4, prune_threshold=0.01)
    print(f"  Original:    {res['original_size']:>6,} B")
    print(f"  SIREN v4:    {res['siren_recipe_size']:>6,} B")
    print(f"  Residual:    {res['residual_compressed_size']:>6,} B (raw {res['residual_raw_size']:,})")
    print(f"  Recipe total:{res['recipe_size']:>6,} B")
    print(f"  Bit accuracy do modelo: {res['model_bit_accuracy']:.2f}%")
    print(f"  PSNR (lossy): {res['psnr_db']:.2f} dB")

    # Roundtrip
    recon, meta = ImageINRV4BitPerfect.decompress(res['recipe_bytes'])
    print(f"\n  Roundtrip:")
    print(f"    SHA-256 original:    {res['sha256'][:32]}...")
    print(f"    SHA-256 recuperado:  {meta['sha256_recovered'][:32]}...")
    print(f"    Match: {meta['exact_match']}")
    print(f"    Shape: orig={img.shape} recon={recon.shape}")

    # ZIP comparison
    import zlib as _z
    zip_size = len(_z.compress(img.tobytes(), 9))
    print(f"\n  Comparação:")
    print(f"    ZIP:        {zip_size:>6,} B  (ratio {res['original_size']/zip_size:.2f}x)")
    print(f"    BLKH+Res:   {res['recipe_size']:>6,} B  (ratio {res['original_size']/res['recipe_size']:.2f}x)")
    winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
    print(f"    Vencedor:   {winner}")


if __name__ == '__main__':
    _self_test()
