"""
arithmetic_codec.py
====================
Arithmetic coding for SIREN weights.

Clean-room implementation of integer arithmetic coding per:
Witten, Neal, and Cleary (1987), "Arithmetic Coding for Data Compression".

This is the codec referenced in Experiment 19 of the BHUH research effort.
Reimplemented in this environment because the original Exp 19 source was not
available in the working directory.

Usage:
    from arithmetic_codec import ArithmeticEncoder, ArithmeticDecoder
    encoder = ArithmeticEncoder(frequency_table)
    compressed = encoder.encode(symbols)
    symbols_decoded = decoder.decode(compressed, num_symbols)
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple


# 32-bit arithmetic coding constants (Witten/Neal/Cleary)
CODE_VALUE_BITS = 32
TOP_MASK = 1 << (CODE_VALUE_BITS - 1)
SECOND_MASK = 1 << (CODE_VALUE_BITS - 2)
BOTTOM_MASK = (1 << CODE_VALUE_BITS) - 1
MAX_FREQ = 1 << 16  # cap to prevent overflow in 32-bit math


class FrequencyTable:
    """Symbol frequency table with adaptive updates."""

    def __init__(self, num_symbols: int = 257):
        # 256 symbols + 1 EOF marker
        self.num_symbols = num_symbols
        # Initialize with uniform frequency 1 for each symbol
        self.freq = np.ones(num_symbols, dtype=np.int64)
        self.cum_freq = np.zeros(num_symbols + 1, dtype=np.int64)
        self._rebuild_cum()
        self.total = int(self.cum_freq[-1])

    def _rebuild_cum(self):
        # cum_freq[i] = sum of freq[0..i-1]
        np.cumsum(self.freq, out=self.cum_freq[1:])
        self.cum_freq[0] = 0
        self.total = int(self.cum_freq[-1])

    def update(self, symbol: int):
        self.freq[symbol] += 1
        # Rescale if total exceeds MAX_FREQ to prevent overflow
        if self.total >= MAX_FREQ:
            self.freq = np.maximum(self.freq // 2, 1)
        self._rebuild_cum()

    def get_range(self, symbol: int) -> Tuple[int, int, int]:
        """Returns (low, high, total) for arithmetic coding."""
        return int(self.cum_freq[symbol]), int(self.cum_freq[symbol + 1]), self.total


class ArithmeticEncoder:
    """Integer arithmetic encoder (Witten/Neal/Cleary 1987)."""

    def __init__(self, freq_table: FrequencyTable = None):
        self.ft = freq_table if freq_table is not None else FrequencyTable()

    def encode(self, symbols: List[int]) -> bytes:
        low = 0
        high = BOTTOM_MASK
        pending_bits = 0
        output_bits: List[int] = []
        ft = self.ft

        for s in symbols:
            sym_low, sym_high, total = ft.get_range(s)
            rng = high - low + 1
            high = low + (rng * sym_high) // total - 1
            low = low + (rng * sym_low) // total
            ft.update(s)

            while True:
                if high < TOP_MASK:
                    # Top bit is 0: emit 0 + pending 1s
                    output_bits.append(0)
                    output_bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= TOP_MASK:
                    # Top bit is 1: emit 1 + pending 0s
                    output_bits.append(1)
                    output_bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= TOP_MASK
                    high -= TOP_MASK
                elif low >= SECOND_MASK and high < (TOP_MASK | SECOND_MASK):
                    # Pending bit case
                    pending_bits += 1
                    low -= SECOND_MASK
                    high -= SECOND_MASK
                else:
                    break
                low = (low << 1) & BOTTOM_MASK
                high = ((high << 1) | 1) & BOTTOM_MASK

        # Flush
        pending_bits += 1
        if low < SECOND_MASK:
            output_bits.append(0)
            output_bits.extend([1] * pending_bits)
        else:
            output_bits.append(1)
            output_bits.extend([0] * pending_bits)

        # Pack bits into bytes
        return _bits_to_bytes(output_bits)


class ArithmeticDecoder:
    """Integer arithmetic decoder."""

    def __init__(self, freq_table: FrequencyTable = None):
        self.ft = freq_table if freq_table is not None else FrequencyTable()

    def decode(self, data: bytes, num_symbols: int) -> List[int]:
        bits = _bytes_to_bits(data)
        bit_idx = 0
        value = 0
        for _ in range(CODE_VALUE_BITS):
            value = (value << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
            bit_idx += 1

        low = 0
        high = BOTTOM_MASK
        decoded: List[int] = []
        ft = self.ft

        for _ in range(num_symbols):
            rng = high - low + 1
            # Find symbol whose range contains (value - low) * total / rng
            target = ((value - low + 1) * ft.total - 1) // rng
            # Binary search in cum_freq
            sym = _find_symbol(ft.cum_freq, target)
            decoded.append(sym)
            sym_low, sym_high, total = ft.get_range(sym)
            high = low + (rng * sym_high) // total - 1
            low = low + (rng * sym_low) // total
            ft.update(sym)

            while True:
                if high >= TOP_MASK:
                    pass
                elif low >= TOP_MASK:
                    value -= TOP_MASK
                    low -= TOP_MASK
                    high -= TOP_MASK
                elif low >= SECOND_MASK and high < (TOP_MASK | SECOND_MASK):
                    value -= SECOND_MASK
                    low -= SECOND_MASK
                    high -= SECOND_MASK
                else:
                    break
                low = (low << 1) & BOTTOM_MASK
                high = ((high << 1) | 1) & BOTTOM_MASK
                next_bit = bits[bit_idx] if bit_idx < len(bits) else 0
                bit_idx += 1
                value = ((value << 1) | next_bit) & BOTTOM_MASK

        return decoded


def _find_symbol(cum_freq: np.ndarray, target: int) -> int:
    """Binary search: find symbol s such that cum_freq[s] <= target < cum_freq[s+1]."""
    # cum_freq is sorted ascending
    lo, hi = 0, len(cum_freq) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum_freq[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _bits_to_bytes(bits: List[int]) -> bytes:
    # Pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def encode_weights(weights: np.ndarray, bits_per_weight: int = 8) -> bytes:
    """
    Quantize weights to `bits_per_weight` levels and arithmetic-code them.

    Returns the full compressed byte payload (header + arithmetic-coded body).
    Layout:
      - 4 bytes: number of weights (uint32 LE)
      - 4 bytes: original dtype info — here we use 0 for float32 source
      - 4 bytes: bits_per_weight (uint32 LE)
      - 4 bytes: min value (float32 LE)
      - 4 bytes: max value (float32 LE)
      - 4 bytes: number of quantization levels (uint32 LE) = 1<<bits_per_weight
      - N bytes: arithmetic-coded symbols
    """
    if bits_per_weight < 1 or bits_per_weight > 16:
        raise ValueError("bits_per_weight must be in [1, 16]")

    w = weights.astype(np.float32).flatten()
    n = w.size
    w_min = float(w.min()) if w.size else 0.0
    w_max = float(w.max()) if w.size else 1.0
    if w_max == w_min:
        w_max = w_min + 1.0

    n_levels = 1 << bits_per_weight
    # Quantize to [0, n_levels-1]
    scaled = ((w - w_min) / (w_max - w_min) * (n_levels - 1)).round().astype(np.int64)
    scaled = np.clip(scaled, 0, n_levels - 1)

    symbols = scaled.tolist()
    # Append EOF marker if using 257-symbol table; we use n_levels-symbol table
    ft = FrequencyTable(num_symbols=n_levels)
    encoder = ArithmeticEncoder(freq_table=ft)
    body = encoder.encode(symbols)

    header = bytearray()
    header += int(n).to_bytes(4, 'little')
    header += int(0).to_bytes(4, 'little')  # float32 source
    header += int(bits_per_weight).to_bytes(4, 'little')
    header += np.float32(w_min).tobytes()
    header += np.float32(w_max).tobytes()
    header += int(n_levels).to_bytes(4, 'little')
    header += int(len(symbols)).to_bytes(4, 'little')
    header += int(len(body)).to_bytes(4, 'little')

    return bytes(header) + body


def decode_weights(payload: bytes) -> np.ndarray:
    """Inverse of encode_weights."""
    offset = 0
    n = int.from_bytes(payload[offset:offset + 4], 'little'); offset += 4
    _ = int.from_bytes(payload[offset:offset + 4], 'little'); offset += 4
    bits_per_weight = int.from_bytes(payload[offset:offset + 4], 'little'); offset += 4
    w_min = float(np.frombuffer(payload[offset:offset + 4], dtype=np.float32)[0]); offset += 4
    w_max = float(np.frombuffer(payload[offset:offset + 4], dtype=np.float32)[0]); offset += 4
    n_levels = int.from_bytes(payload[offset:offset + 4], 'little'); offset += 4
    n_symbols = int.from_bytes(payload[offset:offset + 4], 'little'); offset += 4
    body_len = int.from_bytes(payload[offset:offset + 4], 'little'); offset += 4
    body = payload[offset:offset + body_len]

    ft = FrequencyTable(num_symbols=n_levels)
    decoder = ArithmeticDecoder(freq_table=ft)
    symbols = decoder.decode(body, n_symbols)

    scaled = np.array(symbols, dtype=np.float32)
    w = w_min + scaled / max(1, n_levels - 1) * (w_max - w_min)
    return w.reshape(-1)[:n]


if __name__ == "__main__":
    # Self-test: round-trip
    rng = np.random.default_rng(42)
    w = rng.standard_normal(1000).astype(np.float32) * 0.1
    payload = encode_weights(w, bits_per_weight=8)
    w2 = decode_weights(payload)
    err = np.abs(w - w2[:w.size]).max()
    print(f"Round-trip test: {w.size} floats -> {len(payload)} bytes, max_err={err:.4f}")
    print(f"Compression: {w.size * 4 / len(payload):.2f}x vs raw float32")
