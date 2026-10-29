"""
BLKH Exp85: actual authored Mindustry source PNG revisions, new independent project.

STRICTLY reuse Exp83P + Exp84's 14 codecs without alterations, including
bsdiff4 and original historical PNG bytes, with unmodified full RGBA.
Source files frozen before reading real PNG pixels or compression numbers.
Mindustry original repo GPLv3. Follow pinned protocol and preserve
original contributor attribution; no game sprite artwork in BLKH Git repo.

Protocol: research/dedup/BHUH_EXP85_MINDUSTRY_GIT_SPRITES_PROTOCOL.md
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import platform
import urllib.request

import numpy as np
from PIL import __version__ as PIL_VERSION
import zstandard as zstd

import experiment_83_pixelorama_git_history as baseline
import experiment_84_sparse_pixel_patches as sparse

SOURCE_REPO = "Anuken/Mindustry"
SOURCE_URL = "https://github.com/Anuken/Mindustry"
SOURCE_LICENSE = "GNU GPL-3.0 (project-root; individual asset attributions retained)"
SOURCE_GROUPS = [
  [
    "defense",
    "bae93299c7721a35431daa33e3af8b1f6c596c8d",
    [
      "core/assets-raw/sprites/blocks/defense/large-shield-projector-team.png",
      "core/assets-raw/sprites/blocks/defense/large-shield-projector.png"
    ]
  ],
  [
    "texture_cleanup",
    "916a00b6ff8f1bc16427ef430df9223b3855d908",
    [
      "core/assets-raw/sprites/blocks/campaign/launchpod.png",
      "core/assets-raw/sprites/blocks/distribution/phase-conveyor.png",
      "core/assets-raw/sprites/blocks/drills/blast-drill-rotator.png",
      "core/assets-raw/sprites/blocks/drills/oil-extractor-rotator.png",
      "core/assets-raw/sprites/blocks/drills/water-extractor-rotator.png",
      "core/assets-raw/sprites/blocks/environment/pine.png",
      "core/assets-raw/sprites/blocks/environment/snow-pine.png",
      "core/assets-raw/sprites/blocks/environment/spore-pine.png",
      "core/assets-raw/sprites/blocks/liquid/phase-conduit.png",
      "core/assets-raw/sprites/blocks/power/power-source.png",
      "core/assets-raw/sprites/blocks/power/power-void.png",
      "core/assets-raw/sprites/blocks/production/pulverizer.png",
      "core/assets-raw/sprites/blocks/storage/vault.png",
      "core/assets-raw/sprites/blocks/turrets/swarmer.png",
      "core/assets-raw/sprites/units/eclipse.png"
    ]
  ],
  [
    "walls",
    "0aaa77a5eae7c3e3474bc869d6b622253f729575",
    [
      "core/assets-raw/sprites/blocks/walls/scrap-wall-gigantic.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-huge1.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-huge2.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-huge3.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-large1.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-large2.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-large3.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall-large4.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall1.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall2.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall3.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall4.png",
      "core/assets-raw/sprites/blocks/walls/scrap-wall5.png"
    ]
  ]
]
EXPECTED_PAIR_COUNT = 30


def configure_sources() -> None:
    # Existing encoders and decoders are NOT modified. These source metadata
    # globals only select the independent original video-game repository.
    baseline.REPOSITORY = SOURCE_REPO
    baseline.REPOSITORY_URL = SOURCE_URL
    baseline.LICENSE = SOURCE_LICENSE


def source_license(root:Path) -> None:
    url = ("https://raw.githubusercontent.com/Anuken/Mindustry/"
           + SOURCE_GROUPS[0][1] + "/LICENSE")
    blob = baseline.request_bytes(url, 150_000)
    if b"GNU GENERAL PUBLIC LICENSE" not in blob or b"Version 3" not in blob:
        raise ValueError("Unexpected/unverified original source license")
    (root/"THIRD_PARTY_MINDUSTRY_GPLv3_LICENSE.txt").write_bytes(blob)
    (root/"ORIGINAL_ART_ASSET_ATTRIBUTION.txt").write_text(
        "Original game sprite PNGs from Anuken/Mindustry\n"
        "https://github.com/Anuken/Mindustry\n"
        "Project repository LICENSE: GPLv3; check individual source attribution\n"
        "Each PNG original and historical parent/child commit hash is recorded "
        "in real_git_source_inventory.json\n"
        "BLKH program source is NOT implied to have the same license as the images.\n")


def discover(root:Path) -> tuple[list[dict],dict]:
    configure_sources()
    src = root/"sources"
    src.mkdir(parents=True,exist_ok=True)
    images=[]
    provenance={}
    for label,sha,paths in SOURCE_GROUPS:
        parent=baseline.get_parent(sha)
        pairs=[]
        actually_changed=0
        for idx,path in enumerate(paths):
            before_after=[]
            for role,ref in (("old",parent),("new",sha)):
                original,rgba=baseline.source_png(ref,path)
                fn=src/f"{label}_{idx:02}_{role}.png"
                fn.write_bytes(original)
                meta={
                    "source_repository":SOURCE_REPO,
                    "source_original_rights":SOURCE_LICENSE,
                    "commit_label":label,
                    "commit_sha":sha,
                    "parent_sha":parent,
                    "file_path":path,
                    "pair_index":idx,
                    "role":role,
                    "filename":f"{label}:{idx}:{role}:{path}",
                    "source_png_sha256":baseline.sha(original),
                    "pixel_rgba_sha256":baseline.sha(rgba.tobytes()),
                    "width":int(rgba.shape[1]),
                    "height":int(rgba.shape[0]),
                    "alpha_zero_pixels":int(np.count_nonzero(rgba[:,:,3]==0)),
                }
                before_after.append({"rgba":rgba,"raw":original,"meta":meta})
            identical=(before_after[0]["rgba"].shape==before_after[1]["rgba"].shape
                       and np.array_equal(before_after[0]["rgba"],before_after[1]["rgba"]))
            actually_changed+=int(not identical)
            pairs.append({"path":path,"decoded_rgba_identical":identical})
            images.extend(before_after)
        provenance[label]={
            "commit":sha,"first_parent":parent,
            "png_pairs_modified_in_git":len(paths),
            "png_pairs_really_changed_rgba":actually_changed,
            "paths":pairs,
        }
    inventory={
        "source_repository":SOURCE_REPO,
        "license":SOURCE_LICENSE,
        "source_revisions":provenance,
        "revision_groups":provenance,
        "source_png_count":len(images),
        "real_file_pair_count":len(images)//2,
        "pixel_changed_pairs":sum(x["png_pairs_really_changed_rgba"] for x in provenance.values()),
        "external_repository_independent_of_Pixelorama":True,
    }
    (root/"real_git_source_inventory.json").write_text(json.dumps(inventory,indent=2))
    if len(images)!=EXPECTED_PAIR_COUNT*2 or len(SOURCE_GROUPS)!=3:
        raise AssertionError("Preregistered 30 asset pairs were not observed")
    return images,inventory


def self_test() -> None:
    configure_sources()
    assert len(SOURCE_GROUPS)==3
    assert [len(group[2]) for group in SOURCE_GROUPS]==[2,15,13]
    assert sum(len(group[2]) for group in SOURCE_GROUPS)==EXPECTED_PAIR_COUNT
    assert len({name for _,_,files in SOURCE_GROUPS for name in files})==30
    sparse.self_test()
    print("EXP85 SELFTEST PASS: exact previous 14 codecs, fixed 30 genuine history paths")


def execute(root:Path) -> None:
    root.mkdir(parents=True,exist_ok=True)
    source_license(root)
    originals,inventory=discover(root)
    if len(originals)!=2*EXPECTED_PAIR_COUNT:
        raise AssertionError("Unexpected external sprite-pair source count")
    cases=[("all_30_pairs",originals)]
    for label,commit,paths in SOURCE_GROUPS:
        selected=[item for item in originals if item["meta"]["commit_label"]==label]
        if len(selected)!=2*len(paths):
            raise ValueError("Git group source images missing")
        cases.append((label,selected))
    records=[]
    for name,group in cases:
        row=sparse.evaluate(group,inventory,name,root)
        records.append(row)
        print("EXP85",name,
              "RGBAs changed",sum(
                  (x["rgba"].shape!=y["rgba"].shape or
                   not np.array_equal(x["rgba"],y["rgba"]))
                  for x,y in zip(group[::2],group[1::2])),
              "of",len(group)//2,
              "changed pixel fraction",round(
                  row["changed_rgba_pixel_statistics"]["changed_pixel_fraction"],5),
              "sparse",row["best_sparse_codec"],row["best_sparse_bytes"],
              "best baseline",row["best_of_12_strong_references"],
              row["best_reference_bytes"],
              "won",row["sparse_smaller_than_best_reference"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":85,
            "protocol":"research/dedup/BHUH_EXP85_MINDUSTRY_GIT_SPRITES_PROTOCOL.md",
            "source_repo":SOURCE_REPO,
            "source_license":SOURCE_LICENSE,
            "source_revisions":inventory,
            "total_git_status_modified_png_pairs":EXPECTED_PAIR_COUNT,
            "source_images":2*EXPECTED_PAIR_COUNT,
            "source_is_new_project_relative_to_Pixelorama":True,
            "candidate_codecs":list(sparse.NEW),
            "reference_codecs":list((sparse.BINARY,)+sparse.OLD),
            "tested_original_Git_source_under_GPL3":True,
            "software":{"python":platform.python_version(),
                        "numpy":np.__version__,"pillow":PIL_VERSION,
                        "zstandard":zstd.__version__},
            "results":records,
        },indent=2))
    if len(records)!=4:raise AssertionError("Missing natural source group")
    wins=sum(int(r["sparse_smaller_than_best_reference"]) for r in records[1:])
    criterion=(records[0]["sparse_smaller_than_best_reference"] and wins>=2)
    print("EXP85 COMPLETE: 30 genuine Git history pairs, 56 independently reversible full archives, transfer win",criterion,"wins per three commits",wins,flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp85")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        execute(Path(a.out))


if __name__=="__main__":
    main()
