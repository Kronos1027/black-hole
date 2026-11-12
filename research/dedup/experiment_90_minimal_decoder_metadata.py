"""
BLKH Experiment 90 — decode-minimal per-codec exact RGBA historical archives.

One retrospective per-method manifest ablation across the same 21 Wesnoth and
29 SuperTux historical PNG pairs from Exp87/88. LEAN deliberately DROPS
historical provenance, FULL does not. Neither is a new image codec.
Frozen protocol: BHUH_EXP90_DECODER_MINIMAL_METADATA_PROTOCOL.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import zstandard as zstd
from PIL import __version__ as PIL_VERSION

import experiment_83_pixelorama_git_history as legacy
import experiment_84_sparse_pixel_patches as sparse
import experiment_89_metadata_envelope as earlier

METHODS = tuple(sparse.ALL)
SPRITE_FIELDS = ("x", "y", "width", "height", "pixel_rgba_sha256")
OLD_ATLAS_FIELDS = ("x", "y", "width", "height")
VALID_MODES = ("full_zstd", "lean_raw", "lean_zstd")


def sha(stream: bytes) -> str:
    return hashlib.sha256(stream).hexdigest()


def lean_meta(original: dict) -> dict:
    mode = original["codec"]
    if mode not in METHODS:
        raise ValueError("Unexpected old source compression method")
    original_sprites = original["source_sprites"]
    if not isinstance(original_sprites, list) or len(original_sprites) < 2 or len(original_sprites) % 2:
        raise ValueError("Source pairing missing or invalid")
    lean = {
        "codec": mode,
        "atlas_shape": original["atlas_shape"],
        "atlas_sha256": original["atlas_sha256"],
        "source_sprites": [
            {key: entry[key] for key in SPRITE_FIELDS}
            for entry in original_sprites
        ],
    }
    if mode in legacy.ALL_METHODS:
        lean["source_png_count"] = original["source_png_count"]
        lean["historical_pair_count"] = original["historical_pair_count"]
    if mode == "original_png_zstd" or mode == sparse.BINARY:
        for i, entry in enumerate(original_sprites):
            lean["source_sprites"][i]["source_png_sha256"] = entry["source_png_sha256"]
    elif mode == "sparse_pngold_zstd":
        for i, entry in enumerate(original_sprites):
            if i % 2 == 0:
                lean["source_sprites"][i]["source_png_sha256"] = entry["source_png_sha256"]
    elif mode == "sparse_webpold_zstd":
        lean["old_atlas_shape"] = original["old_atlas_shape"]
        lean["old_atlas_rgba_sha256"] = original["old_atlas_rgba_sha256"]
        lean["old_atlas_positions"] = [
            {key: position[key] for key in OLD_ATLAS_FIELDS}
            for position in original["old_atlas_positions"]
        ]
    return lean


def lean_archive(old: bytes) -> bytes:
    old_magic, metadata, payload = earlier.parse_original(old)
    full = json.loads(metadata)
    lean = lean_meta(full)
    compact = json.dumps(lean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not 0 < len(compact) < earlier.MAX_META[old_magic]:
        raise ValueError("Lean manifest outside old archive safety bounds")
    result = earlier.OLD_HEAD.pack(old_magic, 1, len(compact)) + compact + payload
    if result[len(result)-len(payload):] != payload:
        raise AssertionError("Legacy codec image payload changed")
    return result


def check_pixel_identity(raw: bytes, mode: str, original: list[dict]) -> None:
    pictures, meta = earlier.decode_original(raw, mode)
    if len(pictures) != len(original):
        raise AssertionError("Wrong old/new exact PNG source pair count")
    if any(x.shape != y["rgba"].shape or not np.array_equal(x, y["rgba"])
           for x, y in zip(pictures, original)):
        raise AssertionError("Decoded source is not exact original RGBA")
    for rgba, source in zip(pictures, original):
        if sha(rgba.tobytes()) != source["meta"]["pixel_rgba_sha256"]:
            raise AssertionError("Original historic uint8 RGBA hash mismatch")


def self_test() -> None:
    sparse.self_test()
    old = {
        "codec": "original_png_zstd", "source_png_count": 2,
        "historical_pair_count": 1, "atlas_shape": [16, 32, 4],
        "atlas_sha256": "0" * 64,
        "source_sprites": [
            {"x": 0, "y": 0, "width": 16, "height": 16,
             "pixel_rgba_sha256": "1" * 64, "source_png_sha256": "2" * 64,
             "filename": "sensitive.png", "source_repository": "example/unused"},
            {"x": 16, "y": 0, "width": 16, "height": 16,
             "pixel_rgba_sha256": "3" * 64, "source_png_sha256": "4" * 64,
             "filename": "sensitive2.png", "source_repository": "example/unused"},
        ],
        "source_repository": "example/unused",
    }
    compact = lean_meta(old)
    assert "source_repository" not in compact
    assert "filename" not in compact["source_sprites"][0]
    assert "source_png_sha256" in compact["source_sprites"][0]
    raw = json.dumps(old, sort_keys=True, separators=(",", ":")).encode()
    historical = earlier.OLD_HEAD.pack(legacy.MAGIC, 1, len(raw)) + raw + b"nonempty test payload"
    minimal = lean_archive(historical)
    m, small, payload = earlier.parse_original(minimal)
    assert m == legacy.MAGIC and payload == b"nonempty test payload"
    assert json.loads(small) == compact
    for record in (minimal, historical):
        wrapped = earlier.rewrap(record, "zstd9")
        assert earlier.unrewrap(wrapped) == record
    for mode in METHODS:
        v = dict(old, codec=mode)
        if mode == "sparse_webpold_zstd":
            v["old_atlas_shape"] = [16, 16, 4]
            v["old_atlas_rgba_sha256"] = "5" * 64
            v["old_atlas_positions"] = [{"x": 0, "y": 0, "width": 16, "height": 16}]
        mini = lean_meta(v)
        if mode in legacy.ALL_METHODS:
            assert mini["source_png_count"] == 2 and mini["historical_pair_count"] == 1
        else:
            assert "source_png_count" not in mini
        assert bool("old_atlas_shape" in mini) == (mode == "sparse_webpold_zstd")
    print("EXP90 SELFTEST PASS: 14 codec-specific manifests, payload unchanged, original and lean archival envelope checks", flush=True)


def build(out: Path) -> None:
    out.mkdir(exist_ok=True, parents=True)
    datasets = []
    original_count = 0
    treated_count = 0
    for label, exp, aggregate, pairs, n_images in earlier.DATASETS:
        root = out / label
        root.mkdir(parents=True, exist_ok=True)
        exp.source_license(root)
        images, inventory = exp.discover(root)
        assert len(images) == n_images and len(images) // 2 == pairs
        groups = [(aggregate, images)]
        for name, _, paths in exp.SOURCE_GROUPS:
            selection = [x for x in images if x["meta"]["commit_label"] == name]
            assert len(selection) == len(paths) * 2
            groups.append((name, selection))
        assert len(groups) == 4
        all_cases = []
        for name, selection in groups:
            res = sparse.evaluate(selection, inventory, name, root)
            cases = {}
            for method in METHODS:
                record = res["methods"][method]
                original = (root/name/record["filename"]).read_bytes()
                assert len(original) == record["bytes"] and sha(original) == record["sha256"]
                check_pixel_identity(original, method, selection)
                original_magic, old_meta, original_payload = earlier.parse_original(original)
                minimal = lean_archive(original)
                _, lean_bytes, lean_payload = earlier.parse_original(minimal)
                assert original_payload == lean_payload
                assert json.loads(lean_bytes) == lean_meta(json.loads(old_meta))
                payload_sha = sha(original_payload)
                flavors = {
                    "full_zstd": earlier.rewrap(original, "zstd9"),
                    "lean_raw": minimal,
                    "lean_zstd": earlier.rewrap(minimal, "zstd9"),
                }
                measures = {}
                for flavor, stream in flavors.items():
                    started = time.perf_counter_ns()
                    path = root/name/(method + "." + flavor + ".bl90")
                    path.write_bytes(stream)
                    saved = path.read_bytes()
                    restored = (earlier.unrewrap(saved) if flavor != "lean_raw" else saved)
                    expected = (original if flavor == "full_zstd" else minimal)
                    assert restored == expected
                    _, _, recovered_payload = earlier.parse_original(restored)
                    assert sha(recovered_payload) == payload_sha
                    check_pixel_identity(restored, method, selection)
                    measures[flavor] = {
                        "filename": path.name,
                        "bytes": len(saved),
                        "sha256": sha(saved),
                        "restored_archive_sha256": sha(restored),
                        "original_codec_payload_sha256": payload_sha,
                        "original_codec_payload_byte_identical": True,
                        "whole_archive_is_decodable_to_exact_rgba": True,
                        "decode_check_seconds": (time.perf_counter_ns()-started)/1e9,
                    }
                    treated_count += 1
                cases[method] = {
                    "original_filename": record["filename"],
                    "original_bytes": len(original),
                    "original_sha256": sha(original),
                    "original_full_manifest_bytes": len(old_meta),
                    "lean_required_manifest_bytes": len(lean_bytes),
                    "source_image_payload_bytes": len(original_payload),
                    "original_magic": original_magic.decode("ascii"),
                    "flavors": measures,
                }
                original_count += 1
            ranks = {}
            for flavor in VALID_MODES:
                left = min((row["flavors"][flavor]["bytes"], mode)
                           for mode, row in cases.items() if mode in sparse.NEW)
                right = min((row["flavors"][flavor]["bytes"], mode)
                            for mode, row in cases.items()
                            if mode in (sparse.BINARY,) + sparse.OLD)
                ranks[flavor] = {
                    "best_sparse": left, "best_reference": right,
                    "sparse_strict_win": left[0] < right[0],
                }
            row = {
                "group": name, "pair_count": len(selection)//2, "files": cases,
                "original_best_sparse": [res["best_sparse_bytes"], res["best_sparse_codec"]],
                "original_best_reference": [res["best_reference_bytes"], res["best_of_12_strong_references"]],
                "arm_rankings": ranks,
            }
            all_cases.append(row)
            print("EXP90", label, name, "original", row["original_best_sparse"],
                  row["original_best_reference"], "arms",
                  {f: (ranks[f]["best_sparse"], ranks[f]["best_reference"])
                   for f in VALID_MODES}, flush=True)
        datasets.append({
            "dataset": label, "source_original_png_pairs": pairs,
            "source_inventory": inventory, "cases": all_cases,
        })
        (out/"results.json").write_text(json.dumps({
            "experiment": 90, "protocol": "research/dedup/BHUH_EXP90_DECODER_MINIMAL_METADATA_PROTOCOL.md",
            "known_previously_studied_source_populations": True,
            "methods": list(METHODS), "arms": list(VALID_MODES),
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "pillow": PIL_VERSION, "zstd": zstd.__version__},
            "datasets": datasets,
        }, indent=2))
    assert original_count == 112 and treated_count == 336
    print("EXP90 COMPLETE: 112 original + 336 treated saved archives = 448 persisted complete streams; independent verification pending", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", default="/tmp/blkh_exp90")
    args = parser.parse_args()
    self_test() if args.self_test else build(Path(args.out))


if __name__ == "__main__":
    main()
