"""
BLKH Exp86 — preregistered unrelated Godot demo project's historical PNG edits.

Original source exact file paths and commits frozen in
BHUH_EXP86_GODOT_SOURCE_HOLDOUT_PROTOCOL.md before accessing source image bytes.
Reuse Exp84's EXACT 14 existing codec methods, parameters and decoders.
This is a domain-shift stress test, not claim of universal superiority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import urllib.request
import numpy as np
from PIL import __version__ as PIL_VERSION
import zstandard as zstd

import experiment_83_pixelorama_git_history as baseline
import experiment_84_sparse_pixel_patches as sparse

SOURCE_REPO = "godotengine/godot-demo-projects"
SOURCE_URL = "https://github.com/godotengine/godot-demo-projects"
SOURCE_LICENSE = "MIT (project-root LICENSE.md; asset-specific ownership not independently determined)"
SOURCE_GROUPS = [
    ("platformer_art_redraw", "545d63cd2dbdb87222bfde5371c947483df811e7", [
        "2d/platformer/assets/art/background/cloud_1.png",
        "2d/platformer/assets/art/background/cloud_2.png",
        "2d/platformer/assets/art/background/cloud_3.png",
        "2d/platformer/assets/art/background/distant_platforms_1.png",
        "2d/platformer/assets/art/background/distant_platforms_2.png",
        "2d/platformer/assets/art/background/sky.png",
        "2d/platformer/assets/art/coin/coin.png",
        "2d/platformer/assets/art/enemy/enemy.png",
        "2d/platformer/assets/art/platforms/moving_platform.png",
        "2d/platformer/assets/art/platforms/one_way_platform.png",
        "2d/platformer/assets/art/player/bullet/bullet.png",
        "2d/platformer/assets/art/player/robot_demo.png",
        "2d/platformer/assets/art/tileset/tiles_demo.png",
    ]),
    ("godotea_sprite_revision", "765ca64680da6a251241cc66bb1c0ea5d7225a40", [
        "2d/sprite_shaders/godotea.png",
    ]),
]
EXPECTED_PAIRS = 14
PROTOCOL = "research/dedup/BHUH_EXP86_GODOT_SOURCE_HOLDOUT_PROTOCOL.md"
_ORIGINAL_MANIFEST_FOR = baseline.manifest_for


def godot_manifest_for(items: list[dict], inventory: dict) -> dict:
    # Only correct the inherited Pixelorama-specific provenance label.
    # Original encoder algorithms, parameters and archive structures remain unchanged.
    manifest = _ORIGINAL_MANIFEST_FOR(items, inventory)
    manifest["paired_source_order"] = (
        "platformer_art_redraw then godotea_sprite_revision; fixed path order; "
        "first-parent version then Git child version"
    )
    return manifest


def configure_sources() -> None:
    baseline.REPOSITORY = SOURCE_REPO
    baseline.REPOSITORY_URL = SOURCE_URL
    baseline.LICENSE = SOURCE_LICENSE
    sparse.manifest_for = godot_manifest_for


def first_parent(commit_sha: str) -> str:
    # Unlike Pixelorama's original commits, Godot's art redraw is a merge.
    # Its first-parent diff is fixed in the preregistered protocol.
    endpoint = f"https://api.github.com/repos/{SOURCE_REPO}/commits/{commit_sha}"
    info = json.loads(baseline.request_bytes(endpoint, 500_000))
    if info.get("sha") != commit_sha or not info.get("parents"):
        raise ValueError("Missing expected historical source commit/first parent")
    parent = info["parents"][0]["sha"]
    if len(parent) != 40:
        raise ValueError("Invalid source Git first parent SHA")
    return parent


def source_license(out: Path) -> None:
    ref = SOURCE_GROUPS[0][1]
    url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{ref}/LICENSE.md"
    blob = baseline.request_bytes(url, 150_000)
    if b"Permission is hereby granted" not in blob or "MIT" not in SOURCE_LICENSE:
        raise ValueError("Unverified original project's root MIT license")
    (out / "THIRD_PARTY_GODOT_PROJECT_ROOT_LICENSE.md").write_bytes(blob)
    (out / "SOURCE_ATTRIBUTION.txt").write_text(
        "Original authored Git PNG revisions: godotengine/godot-demo-projects\\n"
        "https://github.com/godotengine/godot-demo-projects\\n"
        "Git original contributors, commit and first-parent SHA, paths and PNG hashes "
        "are preserved in real_git_source_inventory.json.\\n"
        "Root MIT license does not override separately attributed third-party art.\\n"
    )


def discover(out: Path) -> tuple[list[dict], dict]:
    configure_sources()
    directory = out / "sources"
    directory.mkdir(exist_ok=True, parents=True)
    items = []
    provenance = {}
    for group, sha, paths in SOURCE_GROUPS:
        parent = first_parent(sha)
        paths_info = []
        altered = 0
        for index, path in enumerate(paths):
            pair = []
            for role, ref in (("old", parent), ("new", sha)):
                raw, rgba = baseline.source_png(ref, path)
                (directory / f"{group}_{index:02d}_{role}.png").write_bytes(raw)
                meta = {
                    "source_repository": SOURCE_REPO,
                    "source_original_rights": SOURCE_LICENSE,
                    "commit_label": group,
                    "commit_sha": sha,
                    "parent_sha": parent,
                    "file_path": path,
                    "pair_index": index,
                    "role": role,
                    "filename": f"{group}:{index}:{role}:{path}",
                    "source_png_sha256": baseline.sha(raw),
                    "pixel_rgba_sha256": baseline.sha(rgba.tobytes()),
                    "width": int(rgba.shape[1]),
                    "height": int(rgba.shape[0]),
                    "alpha_zero_pixels": int(np.count_nonzero(rgba[:, :, 3] == 0)),
                }
                pair.append({"rgba": rgba, "raw": raw, "meta": meta})
            identical = (pair[0]["rgba"].shape == pair[1]["rgba"].shape
                         and np.array_equal(pair[0]["rgba"], pair[1]["rgba"]))
            altered += int(not identical)
            paths_info.append({"path": path, "decoded_rgba_identical": bool(identical)})
            items.extend(pair)
        provenance[group] = {
            "commit": sha, "first_parent": parent,
            "png_pairs_modified_in_git": len(paths),
            "png_pairs_really_changed_rgba": altered,
            "paths": paths_info,
        }
    inventory = {
        "source_repository": SOURCE_REPO,
        "license": SOURCE_LICENSE,
        "revision_groups": provenance,
        "source_revisions": provenance,
        "source_png_count": len(items),
        "real_file_pair_count": len(items) // 2,
        "pixel_changed_pairs": sum(x["png_pairs_really_changed_rgba"] for x in provenance.values()),
        "external_repository_independent_of_Pixelorama_and_Mindustry": True,
    }
    if len(items) != 2 * EXPECTED_PAIRS:
        raise AssertionError("Frozen source set incomplete; do not replace source paths")
    (out / "real_git_source_inventory.json").write_text(json.dumps(inventory, indent=2))
    return items, inventory


def self_test() -> None:
    configure_sources()
    assert [len(g[2]) for g in SOURCE_GROUPS] == [13, 1]
    assert sum(len(g[2]) for g in SOURCE_GROUPS) == EXPECTED_PAIRS
    assert len({path for _, _, group in SOURCE_GROUPS for path in group}) == EXPECTED_PAIRS
    assert len({sha for _, sha, _ in SOURCE_GROUPS}) == 2
    assert len(sparse.ALL) == 14 and len(sparse.NEW) == 2
    sparse.self_test()
    print("EXP86 SELFTEST PASS: 14 fixed Git PNG paths; same 14 codecs")


def execute(out: Path) -> None:
    out.mkdir(exist_ok=True, parents=True)
    source_license(out)
    items, inventory = discover(out)
    cases = [("all_14_pairs", items)]
    for group, _, paths in SOURCE_GROUPS:
        selected = [item for item in items if item["meta"]["commit_label"] == group]
        assert len(selected) == 2 * len(paths)
        cases.append((group, selected))
    results = []
    for name, selected in cases:
        row = sparse.evaluate(selected, inventory, name, out)
        results.append(row)
        print("EXP86", name, "changed fraction",
              row["changed_rgba_pixel_statistics"]["changed_pixel_fraction"],
              "sparse", row["best_sparse_bytes"], row["best_sparse_codec"],
              "reference", row["best_reference_bytes"],
              row["best_of_12_strong_references"],
              "win", row["sparse_smaller_than_best_reference"], flush=True)
    changed_pairs = inventory["pixel_changed_pairs"]
    source_qualified = changed_pairs >= 2
    aggregate_win = results[0]["sparse_smaller_than_best_reference"]
    group_wins = sum(int(x["sparse_smaller_than_best_reference"]) for x in results[1:])
    transfer_win = bool(source_qualified and aggregate_win and group_wins == 2)
    decision = ("TRANSFER_WIN_UNDER_LIMITED_TWO_COMMIT_RULE" if transfer_win else
                "SOURCE_INCONCLUSIVE" if not source_qualified else
                "PREREGISTERED_TRANSFER_GATE_FAILED")
    payload = {
        "experiment": 86, "protocol": PROTOCOL,
        "source_repo": SOURCE_REPO, "source_license": SOURCE_LICENSE,
        "source_revisions": inventory,
        "total_git_status_modified_png_pairs": EXPECTED_PAIRS,
        "source_images": len(items),
        "candidate_codecs": list(sparse.NEW),
        "reference_codecs": list((sparse.BINARY,) + sparse.OLD),
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pillow": PIL_VERSION, "zstandard": zstd.__version__,
        },
        "source_qualified": source_qualified,
        "aggregate_win": bool(aggregate_win),
        "subgroup_wins": group_wins,
        "preregistered_transfer_win": transfer_win,
        "decision": decision, "results": results,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    assert len(results) == 3
    print("EXP86 COMPLETE:", decision, "; 42 persisted full exact archives", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/blkh_exp86")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        execute(Path(args.out))


if __name__ == "__main__":
    main()
