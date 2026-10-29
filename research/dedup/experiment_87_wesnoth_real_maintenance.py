"""
BLKH Exp87 — preregistered unrelated Wesnoth demo project's historical PNG edits.

Original source exact file paths and commits frozen in
BHUH_EXP87_WESNOTH_SOURCE_HOLDOUT_PROTOCOL.md before accessing source image bytes.
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

SOURCE_REPO = "wesnoth/wesnoth"
SOURCE_URL = "https://github.com/wesnoth/wesnoth"
SOURCE_LICENSE = "Wesnoth project root GPL-2.0-or-later; individual artwork rights must be checked"
SOURCE_GROUPS = [
    ("thea_shadow_fix", "ee27cba746684bd72fb0499226db381bf85db5bd", [
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-bow-attack-1.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-bow-attack-2.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-bow-attack-3.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-bow-attack-4.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-bow-defend.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-bow.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-melee-attack-1.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-melee-attack-2.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-melee-attack-3.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-melee-attack-4.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-melee-defend-1.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman-melee-defend-2.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/longbowman.png",
        "data/campaigns/Of_Pearls_and_Pirates/images/units/thea3/masterbowman.png",
    ]),
    ("dune_firetrooper_cleanup", "5d0c0ec39384c364d716fa5039d53913da4b1e20", [
        "data/core/images/units/dunefolk/burner/firetrooper+female.png",
        "data/core/images/units/dunefolk/burner/firetrooper.png",
        "data/core/images/units/dunefolk/burner/scorcher+female.png",
        "data/core/images/units/dunefolk/burner/scorcher.png",
    ]),
    ("ant_sprite_distinction", "811a2c5b05df2d7e48c57f49c4f76c1c3f3923fc", [
        "data/core/images/units/monsters/ant/soldier-attack.png",
        "data/core/images/units/monsters/ant/soldier-defend.png",
        "data/core/images/units/monsters/ant/soldier.png",
    ]),
]
EXPECTED_PAIRS = 21
PROTOCOL = "research/dedup/BHUH_EXP87_WESNOTH_REAL_MAINTENANCE_PROTOCOL.md"
_ORIGINAL_MANIFEST_FOR = baseline.manifest_for


def wesnoth_manifest_for(items: list[dict], inventory: dict) -> dict:
    # Only correct the inherited Pixelorama-specific provenance label.
    # Original encoder algorithms, parameters and archive structures remain unchanged.
    manifest = _ORIGINAL_MANIFEST_FOR(items, inventory)
    manifest["paired_source_order"] = (
        "thea_shadow_fix, dune_firetrooper_cleanup, ant_sprite_distinction; frozen path order; "
        "first-parent version then Git child version"
    )
    return manifest


def configure_sources() -> None:
    baseline.REPOSITORY = SOURCE_REPO
    baseline.REPOSITORY_URL = SOURCE_URL
    baseline.LICENSE = SOURCE_LICENSE
    sparse.manifest_for = wesnoth_manifest_for


def first_parent(commit_sha: str) -> str:
    # Unlike Pixelorama's original commits, Godot's art redraw is a merge.
    # Its first-parent diff is fixed in the preregistered protocol.
    endpoint = f"https://api.github.com/repos/{SOURCE_REPO}/commits/{commit_sha}"
    info = json.loads(baseline.request_bytes(endpoint, 500_000))
    if info.get("sha") != commit_sha or len(info.get("parents", [])) != 1:
        raise ValueError("Missing expected historical source commit/first parent")
    parent = info["parents"][0]["sha"]
    if len(parent) != 40:
        raise ValueError("Invalid source Git first parent SHA")
    return parent


def source_license(out: Path) -> None:
    ref = SOURCE_GROUPS[0][1]
    url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{ref}/COPYING"
    blob = baseline.request_bytes(url, 150_000)
    if b"GNU GENERAL PUBLIC LICENSE" not in blob or b"Version 2" not in blob:
        raise ValueError("Unverified historical original source root GPLv2 license")
    (out / "THIRD_PARTY_WESNOTH_ROOT_COPYING").write_bytes(blob)
    credits_url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{ref}/copyrights.csv"
    credits = baseline.request_bytes(credits_url, 180_000)
    if len(credits) < 1000:
        raise ValueError("Wesnoth original contributor credits missing")
    (out / "THIRD_PARTY_WESNOTH_COPYRIGHTS.csv").write_bytes(credits)
    (out / "SOURCE_ATTRIBUTION.txt").write_text(
        "Historical original Git PNG revisions: wesnoth/wesnoth\\n"
        "https://github.com/wesnoth/wesnoth\\n"
        "Root GPL-2.0-or-later text and contributor credits archived; "
        "verify specific sprite author rights.\\n"
        "Full upstream Git paths, commits, first parents and original PNG SHA256 "
        "are preserved in real_git_source_inventory.json.\\n"
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
    assert [len(g[2]) for g in SOURCE_GROUPS] == [14, 4, 3]
    assert sum(len(g[2]) for g in SOURCE_GROUPS) == EXPECTED_PAIRS
    assert len({path for _, _, group in SOURCE_GROUPS for path in group}) == EXPECTED_PAIRS
    assert len({sha for _, sha, _ in SOURCE_GROUPS}) == 3
    assert len(sparse.ALL) == 14 and len(sparse.NEW) == 2
    sparse.self_test()
    print("EXP87 SELFTEST PASS: 21 fixed Git PNG paths; same 14 codecs")


def execute(out: Path) -> None:
    out.mkdir(exist_ok=True, parents=True)
    source_license(out)
    items, inventory = discover(out)
    cases = [("all_21_pairs", items)]
    for group, _, paths in SOURCE_GROUPS:
        selected = [item for item in items if item["meta"]["commit_label"] == group]
        assert len(selected) == 2 * len(paths)
        cases.append((group, selected))
    results = []
    for name, selected in cases:
        row = sparse.evaluate(selected, inventory, name, out)
        results.append(row)
        print("EXP87", name, "changed fraction",
              row["changed_rgba_pixel_statistics"]["changed_pixel_fraction"],
              "sparse", row["best_sparse_bytes"], row["best_sparse_codec"],
              "reference", row["best_reference_bytes"],
              row["best_of_12_strong_references"],
              "win", row["sparse_smaller_than_best_reference"], flush=True)
    changed_pairs = inventory["pixel_changed_pairs"]
    source_qualified = changed_pairs >= 2
    aggregate_win = results[0]["sparse_smaller_than_best_reference"]
    group_wins = sum(int(x["sparse_smaller_than_best_reference"]) for x in results[1:])
    transfer_win = bool(source_qualified and aggregate_win and group_wins >= 2)
    decision = ("TRANSFER_WIN_UNDER_LIMITED_TWO_COMMIT_RULE" if transfer_win else
                "SOURCE_INCONCLUSIVE" if not source_qualified else
                "PREREGISTERED_TRANSFER_GATE_FAILED")
    payload = {
        "experiment": 87, "protocol": PROTOCOL,
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
    assert len(results) == 4
    print("EXP87 COMPLETE:", decision, "; 56 persisted full exact archives", flush=True)


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
