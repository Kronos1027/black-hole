"""
BLKH Experiment 91: native git PACK vs bsdiff4 vs Zstandard on exact PNG bytes.

Retrospective Wesnoth+SuperTux fixed Git PNG revisions, *stronger*
original PNG FILE-byte reversibility (including ancillary metadata).
Not a comparable pixel-only sparse compressor benchmark.
Preregistered protocol BHUH_EXP91_NATIVE_GIT_PNG_BYTES_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import time

import bsdiff4
import numpy as np
from PIL import __version__ as PIL_VERSION
import zstandard as zstd

import experiment_83_pixelorama_git_history as legacy
import experiment_89_metadata_envelope as previous

METHODS = (
    "pnglist_zstd9",
    "pngpair_bsdiff_zstd9",
    "git_pack_level9_window50_depth50",
)
MAGIC = b"BL91P"
HEADER = struct.Struct("<5sBBHI")
U32 = struct.Struct("<I")
PAIR = struct.Struct("<II")
METADATA_MEMBER_BYTES = 52  # 20-byte Git blob SHA-1 + 32-byte raw PNG SHA256
MAX_FILES = 58
MAX_ONE_PNG = 5_000_000
MAX_RAW = 150_000_000
MAX_ARCHIVE = 35_000_000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def oid_bytes(raw: bytes) -> bytes:
    header = b"blob " + str(len(raw)).encode("ascii") + b"\x00"
    return hashlib.sha1(header + raw).digest()


def gitrun(args: list[str], *, stdin: bytes = b"") -> bytes:
    proc = subprocess.run(["git"] + args, input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=90, check=False)
    if proc.returncode:
        raise RuntimeError("Git tool failure: " + " ".join(args[:5])
                           + "; " + proc.stderr.decode("utf-8", errors="replace")[-1400:])
    return proc.stdout


def git_version() -> str:
    return gitrun(["--version"]).decode().strip()


def init_repository(where: Path) -> None:
    gitrun(["init", "-q", "--object-format=sha1", str(where)])


def encode_git_pack(raw_pngs: list[bytes]) -> bytes:
    with tempfile.TemporaryDirectory(prefix="bl91-encode-") as temp:
        repo = Path(temp) / "objects"
        init_repository(repo)
        object_ids = []
        for source in raw_pngs:
            observed = gitrun(["-C", str(repo), "hash-object", "-w", "--stdin"],
                              stdin=source).strip().decode("ascii")
            expected = oid_bytes(source).hex()
            if observed != expected:
                raise AssertionError("Actual Git blob OID not reproducible")
            object_ids.append(observed)
        request = ("\n".join(object_ids) + "\n").encode("ascii")
        packed = gitrun(
            ["-C", str(repo), "pack-objects", "--stdout", "--window=50",
             "--depth=50", "--compression=9", "--delta-base-offset"],
            stdin=request,
        )
    if not packed.startswith(b"PACK") or not packed:
        raise AssertionError("Native Git pack-objects omitted PACK bytes")
    return packed


def decode_git_pack(packed: bytes, expected_oids: list[bytes]) -> list[bytes]:
    if not packed.startswith(b"PACK"):
        raise ValueError("Invalid native packfile magic")
    # Crucially: fresh isolated repo, NO access to the temporary encoder object database.
    with tempfile.TemporaryDirectory(prefix="bl91-decode-") as temp:
        repo = Path(temp) / "verify"
        init_repository(repo)
        gitrun(["-C", str(repo), "index-pack", "--stdin"], stdin=packed)
        return [gitrun(["-C", str(repo), "cat-file", "blob", oid.hex()])
                for oid in expected_oids]


def zcompress(raw: bytes) -> bytes:
    return zstd.ZstdCompressor(level=9).compress(raw)


def zdecompress(raw: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(raw,max_output_size=MAX_RAW)


def pack_framed(method: str, inputs: list[bytes]) -> bytes:
    if method not in METHODS or not inputs or len(inputs)%2 or len(inputs)>MAX_FILES:
        raise ValueError("Invalid PNG-source byte codec method/count")
    if any(not img.startswith(b"\x89PNG\r\n\x1a\n") or
           not 8 < len(img) <= MAX_ONE_PNG for img in inputs):
        raise ValueError("Invalid or oversized original PNG file-byte source")
    if method == METHODS[0]:
        plain = b"".join(U32.pack(len(img)) + img for img in inputs)
        payload = zcompress(plain)
    elif method == METHODS[1]:
        joined = bytearray()
        for old, new in zip(inputs[::2],inputs[1::2]):
            delta = bsdiff4.diff(old,new)
            joined.extend(PAIR.pack(len(old),len(delta)))
            joined.extend(old)
            joined.extend(delta)
        payload = zcompress(bytes(joined))
    else:
        payload = encode_git_pack(inputs)
    manifest = b"".join(oid_bytes(img) + hashlib.sha256(img).digest()
                        for img in inputs)
    result = HEADER.pack(MAGIC, 1, METHODS.index(method), len(inputs),len(payload)) + manifest + payload
    if len(result) > MAX_ARCHIVE:
        raise ValueError("Entire compressed PNG source archive exceeds bound")
    return result


def decode_framed(blob: bytes) -> tuple[list[bytes], str]:
    if len(blob) < HEADER.size+METADATA_MEMBER_BYTES*2+1 or len(blob)>MAX_ARCHIVE:
        raise ValueError("Truncated/oversized source archive")
    magic, version, method_id, count, content_length = HEADER.unpack_from(blob)
    if magic != MAGIC or version != 1 or method_id >= len(METHODS):
        raise ValueError("Unknown fixed format/method")
    if not 2 <= count <= MAX_FILES or count % 2:
        raise ValueError("Unexpected exact historical file count")
    begin = HEADER.size
    end = begin + count*METADATA_MEMBER_BYTES
    if content_length < 1 or len(blob) != end + content_length:
        raise ValueError("Mismatched source manifest/payload lengths")
    oids = [blob[begin+i*52:begin+i*52+20] for i in range(count)]
    hashes = [blob[begin+i*52+20:begin+(i+1)*52] for i in range(count)]
    method = METHODS[method_id]
    data = blob[end:]
    if method == METHODS[2]:
        files = decode_git_pack(data,oids)
    else:
        decoded = zdecompress(data)
        files = []
        cursor = 0
        if method == METHODS[0]:
            for _ in range(count):
                if len(decoded)-cursor < U32.size:
                    raise ValueError("Incomplete original PNG member length")
                n, = U32.unpack_from(decoded,cursor)
                cursor += U32.size
                if not 8 < n <= MAX_ONE_PNG or n > len(decoded)-cursor:
                    raise ValueError("Bad complete old/new original PNG file length")
                files.append(decoded[cursor:cursor+n])
                cursor += n
        else:
            for _ in range(count//2):
                if len(decoded)-cursor < PAIR.size:
                    raise ValueError("Truncated old PNG source+patch lengths")
                old_n,patch_n = PAIR.unpack_from(decoded,cursor)
                cursor += PAIR.size
                if (not 8 < old_n <= MAX_ONE_PNG or patch_n<1 or patch_n>MAX_ONE_PNG*2
                        or old_n+patch_n>len(decoded)-cursor):
                    raise ValueError("Bad original PNG bsdiff source/patch length")
                old = decoded[cursor:cursor+old_n]
                cursor += old_n
                patch = decoded[cursor:cursor+patch_n]
                cursor += patch_n
                files.extend((old,bsdiff4.patch(old,patch)))
        if cursor != len(decoded):
            raise ValueError("Trailing unclaimed source PNG bytes")
    if len(files) != count:
        raise ValueError("Native file-byte decoder omitted source PNGs")
    for raw, oid, original_sha in zip(files,oids,hashes):
        if (not raw.startswith(b"\x89PNG\r\n\x1a\n") or
                not 8<len(raw)<=MAX_ONE_PNG or oid_bytes(raw)!=oid or
                hashlib.sha256(raw).digest()!=original_sha):
            raise ValueError("Original PNG bytes or Git object SHA mismatch")
    return files,method


def self_test() -> None:
    # True file-byte PNG originals, differing ancillary chunk bytes but same decoded RGBA.
    from io import BytesIO
    from PIL import Image, PngImagePlugin
    a = np.zeros((23,31,4),dtype=np.uint8)
    a[2:7,4:9]=[127,28,93,255]
    files = []
    for i in range(4):
        meta = PngImagePlugin.PngInfo()
        meta.add_text("original-source-note", "item "+str(i))
        with BytesIO() as out:
            Image.fromarray(a, "RGBA").save(out,"PNG",pnginfo=meta)
            files.append(out.getvalue())
    for method in METHODS:
        stored = pack_framed(method,files)
        recreated, decoder_name = decode_framed(stored)
        assert decoder_name == method
        assert recreated == files
        for bad in (stored[:-1],stored[:-13], stored[:12]):
            try: decode_framed(bad)
            except (ValueError, RuntimeError, zstd.ZstdError, OSError): pass
            else: raise AssertionError("Incomplete source archive accepted")
    print("EXP91 SELFTEST PASS: three byte-exact original PNG codecs including metadata-only edits and native isolated Git packfile", flush=True)


def build(root: Path) -> None:
    root.mkdir(exist_ok=True,parents=True)
    all_reports = []
    count = 0
    for label, exp, aggregate, expected_pairs, expected_images in previous.DATASETS:
        target = root/label
        target.mkdir(exist_ok=True,parents=True)
        exp.source_license(target)
        originals, inventory = exp.discover(target)
        assert len(originals) == expected_images and len(originals)//2==expected_pairs
        cases = [(aggregate,originals)]
        for name, _, paths in exp.SOURCE_GROUPS:
            items = [v for v in originals if v["meta"]["commit_label"] == name]
            assert len(items) == 2*len(paths)
            cases.append((name,items))
        assert len(cases)==4
        reports = []
        for case, sources in cases:
            filelist = [entry["raw"] for entry in sources]
            folder = target/case
            folder.mkdir(exist_ok=True,parents=True)
            modes = {}
            for method in METHODS:
                began=time.perf_counter_ns()
                stream=pack_framed(method,filelist)
                encode_sec=(time.perf_counter_ns()-began)/1e9
                filename = method + ".bl91p"
                (folder/filename).write_bytes(stream)
                began=time.perf_counter_ns()
                checked = (folder/filename).read_bytes()
                recovered,name = decode_framed(checked)
                decode_sec=(time.perf_counter_ns()-began)/1e9
                assert name==method and recovered==filelist
                # Strongest fidelity: raw PNG file bytes, not just pixel reconstruction.
                for original,source in zip(recovered,sources):
                    assert sha(original)==source["meta"]["source_png_sha256"]
                    rgba=legacy.decode_rgba(original,source["rgba"].shape)
                    assert np.array_equal(rgba,source["rgba"])
                modes[method] = {
                    "filename": filename, "bytes":len(checked),"sha256":sha(checked),
                    "encode_seconds":encode_sec,"decode_seconds":decode_sec,
                    "exact_original_png_files":True,
                }
                count+=1
            winner=min((v["bytes"],key) for key,v in modes.items())
            row={
                "group":case,"original_git_png_pairs":len(sources)//2,
                "original_png_raw_bytes":sum(len(x) for x in filelist),
                "distinct_git_blob_oids":len({oid_bytes(x) for x in filelist}),
                "methods":modes,
                "smallest_source_bytes_method":winner[1],
                "smallest_source_bytes_size":winner[0],
            }
            reports.append(row)
            print("EXP91",label,case,"original",row["original_png_raw_bytes"],
                  "native_git",modes[METHODS[2]]["bytes"],
                  "zstd9",modes[METHODS[0]]["bytes"],
                  "bsdiff_zstd9",modes[METHODS[1]]["bytes"],
                  "winner",winner,flush=True)
        all_reports.append({
            "dataset": label,"source_pair_count":expected_pairs,
            "original_git_inventory":inventory,"groups":reports,
        })
        (root/"results.json").write_text(json.dumps({
            "experiment":91,
            "protocol":"research/dedup/BHUH_EXP91_NATIVE_GIT_PNG_BYTES_PROTOCOL.md",
            "previously_seen_source_sets":True,
            "semantic_contract":"identical original PNG FILE bytes including all source PNG metadata",
            "methods":list(METHODS),
            "software":{"python":platform.python_version(),
                        "numpy":np.__version__,"Pillow":PIL_VERSION,
                        "zstandard":zstd.__version__,"git":git_version()},
            "datasets":all_reports,
        },indent=2))
    assert len(all_reports)==2 and count==24
    print("EXP91 COMPLETE: 24 genuine exact original PNG-byte standalone archives; independent Git/source audit still required",flush=True)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--out",default="/tmp/blkh_exp91")
    args=parser.parse_args()
    self_test() if args.self_test else build(Path(args.out))


if __name__ == "__main__":
    main()
