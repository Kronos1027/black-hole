"""
BLKH Exp92 — preregistered sixth-source original Endless Sky Git sprite edits and frozen ±8 temporal displacement.

Original source exact file paths and commits frozen in
BHUH_EXP92_TEMPORAL_SHIFT8_ENDLESS_SKY_PROTOCOL.md before accessing source image bytes.
Keep the unchanged 14 Exp84 original baselines and add one preregistered ±8 exact RGBA temporal codec.
The original frozen 11 PNG edit pairs were selected by upstream Git filename metadata before source pixels were read.
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
import experiment_92_motion_shift_patch as motion
import time

SOURCE_REPO = "endless-sky/endless-sky"
SOURCE_URL = "https://github.com/endless-sky/endless-sky"
SOURCE_LICENSE = "Endless Sky copyright images/* CC-BY-SA-4.0 subject to individual file exceptions"
SOURCE_GROUPS = [
    ("astral_crop_first8", "6985a40773ede0f175805fde399e8cc3fea1e5e7", [
        "images/ship/astral cetacean/astral-00.png",
        "images/ship/astral cetacean/astral-01.png",
        "images/ship/astral cetacean/astral-02.png",
        "images/ship/astral cetacean/astral-03.png",
        "images/ship/astral cetacean/astral-04.png",
        "images/ship/astral cetacean/astral-05.png",
        "images/ship/astral cetacean/astral-06.png",
        "images/ship/astral cetacean/astral-07.png",
    ]),
    ("horizon_effect_fix", "57051117483c502f389240060b20c1a83167ae91", [
        "images/effect/horizon grappler+0.png",
        "images/effect/horizon grappler+1.png",
    ]),
    ("vesvi_planet_tweak", "b85cc0a066639ebe78a64d25f2b27b427ecb637f", [
        "images/planet/vesvi.png",
    ]),
]
EXPECTED_PAIRS = 11
PROTOCOL = "research/dedup/BHUH_EXP92_TEMPORAL_SHIFT8_ENDLESS_SKY_PROTOCOL.md"
_ORIGINAL_MANIFEST_FOR = baseline.manifest_for


def endless_sky_manifest_for(items: list[dict], inventory: dict) -> dict:
    # Only correct the inherited Pixelorama-specific provenance label.
    # Original encoder algorithms, parameters and archive structures remain unchanged.
    manifest = _ORIGINAL_MANIFEST_FOR(items, inventory)
    manifest["paired_source_order"] = (
        "astral_crop_first8, horizon_effect_fix, vesvi_planet_tweak; frozen path order; "
        "first-parent version then Git child version"
    )
    return manifest


def configure_sources() -> None:
    baseline.REPOSITORY = SOURCE_REPO
    baseline.REPOSITORY_URL = SOURCE_URL
    baseline.LICENSE = SOURCE_LICENSE
    sparse.manifest_for = endless_sky_manifest_for


def first_parent(commit_sha: str) -> str:
    # Frozen original Endless Sky Git commits must each have one first parent.
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
    root = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{ref}/"
    license_text = baseline.request_bytes(root + "license.txt", 150_000)
    copyright_text = baseline.request_bytes(root + "copyright", 250_000)
    credits = baseline.request_bytes(root + "credits.txt", 250_000)
    if b"GNU GENERAL PUBLIC LICENSE" not in license_text or b"CC-BY-SA-4.0" not in copyright_text:
        raise ValueError("Unverified original Endless Sky license and artwork copyright")
    (out / "THIRD_PARTY_ENDLESS_SKY_license.txt").write_bytes(license_text)
    (out / "THIRD_PARTY_ENDLESS_SKY_copyright").write_bytes(copyright_text)
    (out / "THIRD_PARTY_ENDLESS_SKY_credits.txt").write_bytes(credits)
    (out / "SOURCE_ATTRIBUTION.txt").write_text(
        "Original Endless Sky Git PNG artwork revisions, "
        "https://github.com/endless-sky/endless-sky\\n"
        "Original copyright files identify many images as CC BY-SA 4.0 "
        "with possible individual exceptions; check every chosen image before "
        "redistributing artwork.\\n"
        "Upstream rights, Git paths, revisions and pixel/full PNG SHA256 are "
        "recorded in the separately retained CI artifact; no source PNG "
        "artwork is committed to the BLKH Git repository.\\n"
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
        "independent_source_repo_distinct_from_previous_five": True,
    }
    if len(items) != 2 * EXPECTED_PAIRS:
        raise AssertionError("Frozen source set incomplete; do not replace source paths")
    (out / "real_git_source_inventory.json").write_text(json.dumps(inventory, indent=2))
    return items, inventory


def self_test() -> None:
    configure_sources()
    assert [len(g[2]) for g in SOURCE_GROUPS] == [8, 2, 1]
    assert sum(len(g[2]) for g in SOURCE_GROUPS) == EXPECTED_PAIRS
    assert len({path for _, _, group in SOURCE_GROUPS for path in group}) == EXPECTED_PAIRS
    assert len({sha for _, sha, _ in SOURCE_GROUPS}) == 3
    assert len(sparse.ALL) == 14 and len(sparse.NEW) == 2
    sparse.self_test()
    motion.self_test()
    print("EXP92 SELFTEST PASS: 11 fixed Git PNG paths; 14 unchanged old codecs plus new exact motion method")


def execute(out: Path) -> None:
    out.mkdir(exist_ok=True, parents=True)
    source_license(out)
    items, inventory = discover(out)
    cases = [("all_11_pairs", items)]
    for group, _, paths in SOURCE_GROUPS:
        selected = [item for item in items if item["meta"]["commit_label"] == group]
        assert len(selected) == 2 * len(paths)
        cases.append((group, selected))
    results = []
    for name, selected in cases:
        row = sparse.evaluate(selected, inventory, name, out)
        manifest = row["shared_source_manifest"]
        start = time.perf_counter_ns()
        archive, detail = motion.encode(selected, manifest)
        encode_sec = (time.perf_counter_ns()-start)/1e9
        saved_path = out / name / (motion.MODE + ".bl92m")
        saved_path.write_bytes(archive)
        start = time.perf_counter_ns()
        saved = saved_path.read_bytes()
        decoded, decoded_meta = motion.decode(saved)
        decode_sec = (time.perf_counter_ns()-start)/1e9
        if (decoded_meta != {**manifest, "codec": motion.MODE}
                or any(not np.array_equal(x["rgba"], y) for x,y in zip(selected,decoded))):
            raise AssertionError("New complete motion archive is not original source RGBA exact")
        row["methods"][motion.MODE] = {
            "filename":saved_path.name, "bytes":len(saved), "sha256":baseline.sha(saved),
            "encode_seconds":encode_sec,"decode_seconds":decode_sec,**detail,
        }
        row["new_motion_codec"] = motion.MODE
        row["new_motion_bytes"] = len(saved)
        row["new_motion_strictly_smaller_than_all_old_methods"] = (
            len(saved) < min(row["methods"][x]["bytes"] for x in sparse.ALL))
        row["new_motion_strict_win_vs_best_12_references"] = (
            len(saved) < row["best_reference_bytes"])
        row["new_motion_strict_win_vs_two_previous_sparse"] = (
            len(saved) < row["best_sparse_bytes"])
        results.append(row)
        print("EXP92",name,
              "fraction original changed",row["changed_rgba_pixel_statistics"]["changed_pixel_fraction"],
              "motion",len(saved),"shifted pairs",detail["nonzero_motion_pairs"],
              "motion mismatch fraction",detail["motion_after_pixel_difference_fraction"],
              "prior sparse",row["best_sparse_bytes"],
              "best of 12 refs",row["best_reference_bytes"],
              "new motion strictly best",row["new_motion_strictly_smaller_than_all_old_methods"],
              flush=True)
    changed_pairs = inventory["pixel_changed_pairs"]
    qualified = changed_pairs >= 2
    whole_win = results[0]["new_motion_strictly_smaller_than_all_old_methods"]
    group_wins = sum(int(x["new_motion_strict_win_vs_best_12_references"]) for x in results[1:])
    transfer = bool(qualified and whole_win and group_wins >= 2)
    decision = ("SIXTH_SOURCE_FROZEN_MOTION_GATE_PASSED" if transfer else
                "SOURCE_INCONCLUSIVE" if not qualified else
                "PREREGISTERED_NEW_TEMPORAL_MOTION_GATE_FAILED")
    payload = {
        "experiment": 92, "protocol": PROTOCOL,
        "source_repo": SOURCE_REPO, "source_license": SOURCE_LICENSE,
        "source_revisions": inventory,
        "total_git_status_modified_png_pairs": EXPECTED_PAIRS,
        "source_images": len(items),
        "candidate_codecs": list(sparse.NEW) + [motion.MODE],
        "reference_codecs": list((sparse.BINARY,) + sparse.OLD),
        "all_codecs": list(sparse.ALL) + [motion.MODE],
        "software": {"python":platform.python_version(),"numpy":np.__version__,
                     "pillow":PIL_VERSION,"zstandard":zstd.__version__},
        "source_qualified":qualified,
        "new_motion_aggregate_win":bool(whole_win),
        "new_motion_group_wins_against_reference":group_wins,
        "new_motion_preregistered_transfer_win":transfer,
        "decision":decision,"results":results,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    assert len(results) == 4
    print("EXP92 COMPLETE:", decision, "; 60 persisted full exact archives", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/blkh_exp92")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        execute(Path(args.out))


if __name__ == "__main__":
    main()
