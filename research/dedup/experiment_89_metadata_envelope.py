"""
BLKH Experiment 89: reversible fully self-contained metadata-envelope ablation.

Retrospective analysis of previously known Exp87 Wesnoth and Exp88 SuperTux
source populations. This is a generic container/header compression
improvement, not a new compressed-pixel method or independent holdout.
Protocol: research/dedup/BHUH_EXP89_METADATA_ENVELOPE_PROTOCOL.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import time
import zlib

import numpy as np
import zstandard as zstd
from PIL import __version__ as PIL_VERSION

import experiment_83_pixelorama_git_history as legacy
import experiment_84_sparse_pixel_patches as sparse
import experiment_87_wesnoth_real_maintenance as wesnoth
import experiment_88_supertux_png_history as supertux

OLD_HEAD = struct.Struct("<5sBI")
NEW_HEAD = struct.Struct("<5sB5sII")
MAGIC = b"BL89R"
MAX_FILE = 25_000_000
MAX_META = {b"BL83P": legacy.MAX_METADATA, b"BL84S": sparse.MAX_METADATA}
COMPRESSORS = ("zstd9", "zlib9")
DATASETS = (
    ("exp87_wesnoth", wesnoth, "all_21_pairs", 21, 42),
    ("exp88_supertux", supertux, "all_29_pairs", 29, 58),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_original(raw: bytes) -> tuple[bytes, bytes, bytes]:
    if len(raw) <= OLD_HEAD.size or len(raw) > MAX_FILE:
        raise ValueError("Old full original archive size outside bounds")
    old_magic, version, meta_size = OLD_HEAD.unpack_from(raw)
    if old_magic not in MAX_META or version != 1:
        raise ValueError("Unknown historical archive version")
    if not 0 < meta_size < MAX_META[old_magic]:
        raise ValueError("Historical manifest outside original format bound")
    limit = OLD_HEAD.size + meta_size
    if limit >= len(raw):
        raise ValueError("Truncated original archive: no image data")
    full_metadata = raw[OLD_HEAD.size:limit]
    info = json.loads(full_metadata)
    if not isinstance(info, dict) or "codec" not in info or "source_sprites" not in info:
        raise ValueError("Original full manifest malformed")
    return old_magic, full_metadata, raw[limit:]


def rewrap(raw: bytes, compressor: str) -> bytes:
    old_magic, metadata, payload = parse_original(raw)
    if compressor == "zstd9":
        code = 1
        compressed = zstd.ZstdCompressor(level=9).compress(metadata)
    elif compressor == "zlib9":
        code = 2
        compressed = zlib.compress(metadata, level=9)
    else:
        raise ValueError("Unknown fixed header compressor")
    packed = (NEW_HEAD.pack(MAGIC, code, old_magic, len(metadata), len(compressed))
              + compressed + payload)
    if len(packed) > MAX_FILE:
        raise ValueError("Whole saved wrapped file exceeds original archive cap")
    return packed


def unrewrap(archive: bytes) -> bytes:
    if len(archive) <= NEW_HEAD.size or len(archive) > MAX_FILE:
        raise ValueError("Invalid wrapped archive size")
    magic, code, old_magic, raw_n, compressed_n = NEW_HEAD.unpack_from(archive)
    if magic != MAGIC or code not in (1, 2) or old_magic not in MAX_META:
        raise ValueError("Wrong BL89R version, algorithm or old archive magic")
    if not 0 < raw_n < MAX_META[old_magic]:
        raise ValueError("Oversized or empty uncompressed metadata")
    begin = NEW_HEAD.size
    end = begin + compressed_n
    if compressed_n < 1 or end >= len(archive):
        raise ValueError("Truncated compressed metadata or content payload")
    blob = archive[begin:end]
    if code == 1:
        d = zstd.ZstdDecompressor().decompressobj()
        manifest = d.decompress(blob)
        if not d.eof or d.unused_data:
            raise ValueError("Invalid or trailing Zstandard metadata frame")
    else:
        d = zlib.decompressobj()
        manifest = d.decompress(blob, raw_n + 1)
        manifest += d.flush()
        if not d.eof or d.unused_data or d.unconsumed_tail:
            raise ValueError("Invalid or trailing zlib metadata frame")
    if len(manifest) != raw_n:
        raise ValueError("Decompressed historical manifest wrong size")
    rebuilt = OLD_HEAD.pack(old_magic, 1, raw_n) + manifest + archive[end:]
    parse_original(rebuilt)
    return rebuilt


def decode_original(raw: bytes, mode: str) -> tuple[list[np.ndarray], dict]:
    old_magic, meta, body = parse_original(raw)
    if mode in sparse.NEW + (sparse.BINARY,):
        if old_magic != sparse.MAGIC:
            raise AssertionError("Sparse/binary wrong old archive magic")
        result, info = sparse.decode(raw)
    else:
        if old_magic != legacy.MAGIC:
            raise AssertionError("Prior reference wrong old archive magic")
        result, info = legacy.decode(raw)
    if info["codec"] != mode:
        raise AssertionError("Wrong decoded original codec")
    return result, info


def self_test() -> None:
    original_metadata = json.dumps({
        "codec": "synthetic_test", "source_sprites": [
            {"name": "test.png", "pixels": "123" * 100}
        ], "provenance": {"test": "metadata" * 60},
    }, sort_keys=True, separators=(",", ":")).encode()
    for old_magic in MAX_META:
        old_raw = (OLD_HEAD.pack(old_magic, 1, len(original_metadata))
                   + original_metadata + b"nonempty synthetic payload")
        for algorithm in COMPRESSORS:
            full = rewrap(old_raw, algorithm)
            assert unrewrap(full) == old_raw
            try:
                unrewrap(full[:NEW_HEAD.size + 2])
            except (ValueError, zstd.ZstdError, zlib.error):
                pass
            else:
                raise AssertionError("Truncated metadata erroneously accepted")
            try:
                unrewrap(full[:NEW_HEAD.size] + b"junk" + full[NEW_HEAD.size:])
            except (ValueError, zstd.ZstdError, zlib.error):
                pass
            else:
                raise AssertionError("Metadata corruption erroneously accepted")
    print("EXP89 SELFTEST PASS: both algorithms, both original archive formats, byte-for-byte original restoration and malformed input checks", flush=True)


def build(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for label, experiment, entire, expected_pairs, expected_source_count in DATASETS:
        dest = root / label
        dest.mkdir(exist_ok=True, parents=True)
        experiment.source_license(dest)
        sources, inventory = experiment.discover(dest)
        assert len(sources) == expected_source_count
        assert inventory["real_file_pair_count"] == expected_pairs
        cases = [(entire, sources)]
        for group, commit, paths in experiment.SOURCE_GROUPS:
            filtered = [item for item in sources if item["meta"]["commit_label"] == group]
            assert len(filtered) == len(paths) * 2
            cases.append((group, filtered))
        assert len(cases) == 4
        dataset_report = []
        for name, items in cases:
            original = sparse.evaluate(items, inventory, name, dest)
            files = {}
            for method in sparse.ALL:
                base = original["methods"][method]
                raw = (dest / name / base["filename"]).read_bytes()
                assert len(raw) == base["bytes"] and sha(raw) == base["sha256"]
                old_magic, metadata, payload = parse_original(raw)
                images, info = decode_original(raw, method)
                assert len(images) == len(items)
                assert all(np.array_equal(a["rgba"], b) for a, b in zip(items, images))
                variants = {}
                for algorithm in COMPRESSORS:
                    start = time.perf_counter_ns()
                    stream = rewrap(raw, algorithm)
                    target = dest / name / f"{method}.{algorithm}.bl89r"
                    target.write_bytes(stream)
                    reopened = target.read_bytes()
                    restored = unrewrap(reopened)
                    restored_images, restored_info = decode_original(restored, method)
                    seconds = (time.perf_counter_ns() - start) / 1e9
                    assert restored == raw and restored_info == info
                    assert len(restored_images) == len(items)
                    assert all(np.array_equal(a["rgba"], b) for a, b in zip(items, restored_images))
                    variants[algorithm] = {
                        "file": target.name, "bytes": len(reopened), "sha256": sha(reopened),
                        "difference_from_original_bytes": len(reopened) - len(raw),
                        "restored_original_sha256": sha(restored),
                        "full_repack_and_decode_seconds": seconds,
                        "bit_exact_original_archive_restored": True,
                        "exact_original_rgba_restored": True,
                    }
                files[method] = {
                    "original_bytes": len(raw), "original_sha256": sha(raw),
                    "old_archive_magic": old_magic.decode("ascii"),
                    "old_full_metadata_bytes": len(metadata),
                    "old_encoded_payload_bytes": len(payload),
                    "wrapped": variants,
                }
            compact_best = {algorithm: {
                "best_sparse": min((files[codec]["wrapped"][algorithm]["bytes"], codec)
                                   for codec in sparse.NEW),
                "best_reference": min((files[codec]["wrapped"][algorithm]["bytes"], codec)
                                      for codec in (sparse.BINARY,) + sparse.OLD),
            } for algorithm in COMPRESSORS}
            for value in compact_best.values():
                value["sparse_strict_win"] = value["best_sparse"][0] < value["best_reference"][0]
            row = {
                "group": name, "pair_count": len(items) // 2,
                "source_rgba_changed_pairs": sum(
                    not (a["rgba"].shape == b["rgba"].shape and np.array_equal(a["rgba"], b["rgba"]))
                    for a, b in zip(items[::2], items[1::2])
                ),
                "original_best_sparse_bytes": original["best_sparse_bytes"],
                "original_best_sparse_codec": original["best_sparse_codec"],
                "original_best_reference_bytes": original["best_reference_bytes"],
                "original_best_reference_codec": original["best_of_12_strong_references"],
                "files": files, "repacked_best": compact_best,
            }
            dataset_report.append(row)
            print("EXP89", label, name, "original", row["original_best_sparse_bytes"],
                  row["original_best_reference_bytes"],
                  "zstd9", compact_best["zstd9"]["best_sparse"],
                  compact_best["zstd9"]["best_reference"],
                  "zlib9", compact_best["zlib9"]["best_sparse"],
                  compact_best["zlib9"]["best_reference"], flush=True)
        summary.append({"dataset": label, "source_original_png_pairs": expected_pairs,
                        "source_inventory": inventory, "results": dataset_report})
        (root / "results.json").write_text(json.dumps({
            "experiment": 89,
            "protocol": "research/dedup/BHUH_EXP89_METADATA_ENVELOPE_PROTOCOL.md",
            "not_independent_new_holdout": True,
            "methods": list(sparse.ALL), "repacked_algorithms": list(COMPRESSORS),
            "software": {"python": platform.python_version(),
                         "Pillow": PIL_VERSION, "numpy": np.__version__,
                         "zstandard": zstd.__version__, "zlib": zlib.ZLIB_VERSION},
            "datasets": summary,
        }, indent=2))
    assert len(summary) == 2 and all(len(s["results"]) == 4 for s in summary)
    print("EXP89 GENERATED: 112 original and 224 fully saved self-contained wrapper archives; independent verification required", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", default="/tmp/blkh_exp89")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        build(Path(args.out))


if __name__ == "__main__":
    main()
