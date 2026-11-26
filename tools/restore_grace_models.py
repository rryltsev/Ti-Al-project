"""Recover the verified Ti-Al GRACE SavedModels and training checkpoints."""
import argparse
import csv
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "research" / "grace-final"
INDEX = ROOT / "docs" / "grace-final-models.csv"
CARD = ROOT / "research/tial_fm/runs/grace-2l-preparation-20261107-01/package/model-card.json"
BASE_NAME = "checkpoint.data-00000-of-00001"
BLOCK = 1024 * 1024


def sha256(path):
    output = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(BLOCK), b""):
            output.update(chunk)
    return output.hexdigest()


def download(url, dest, size, checksum):
    if dest.exists():
        if dest.stat().st_size == size and sha256(dest) == checksum:
            return
        raise ValueError(f"Existing archive has a different checksum: {dest}")
    partial = dest.with_suffix(".partial")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            if sha256(partial) != checksum:
                raise ValueError(f"Downloaded archive checksum mismatch: {partial}")
            partial.replace(dest)
            return
        if offset > size:
            raise ValueError(f"Partial download is larger than the pinned archive: {partial}")
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = offset and response.status == 206
                if response.status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected range returned by the file server")
                with partial.open("ab" if resumed else "wb") as output:
                    shutil.copyfileobj(response, output, BLOCK)
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            if attempt == 3:
                raise
            print(f"Download interrupted, resuming ({error})", file=sys.stderr)
            time.sleep(2 ** attempt)
        if partial.exists() and partial.stat().st_size == size and sha256(partial) == checksum:
            partial.replace(dest)
            return
    raise ValueError(f"Archive size or checksum mismatch: {partial}")


def base_file(args, card):
    entry = next(v for k, v in card["files"].items() if k.endswith("/" + BASE_NAME))
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if args.download_base:
        path = cache / "GRACE-2L-OMAT-medium-base-fp64-checkpoint.tar.gz"
        download(card["source_url"], path, card["archive_bytes"], card["archive_sha256"])
    else:
        path = args.base.resolve()
        if path.is_dir():
            path /= BASE_NAME
    if path.stat().st_size == entry["bytes"]:
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Base checkpoint checksum mismatch: {path}")
        return path
    target = cache / BASE_NAME
    recovery = json.loads((MODELS / "base-recovery.json").read_text(encoding="utf-8"))
    if path.stat().st_size == recovery["bytes"]:
        if sha256(path) != recovery["sha256"] or recovery["member_sha256"] != entry["sha256"]:
            raise ValueError(f"Release recovery archive checksum mismatch: {path}")
        with zipfile.ZipFile(path) as source, source.open(recovery["member"]) as data:
            write_verified(target, entry["bytes"], entry["sha256"], lambda output: shutil.copyfileobj(data, output, BLOCK))
        return target
    if path.stat().st_size != card["archive_bytes"] or sha256(path) != card["archive_sha256"]:
        raise ValueError(f"Base input does not match the pinned file or archive: {path}")
    if target.exists() and target.stat().st_size == entry["bytes"] and sha256(target) == entry["sha256"]:
        return target
    with tarfile.open(path, "r:gz") as source:
        members = [m for m in source if m.isfile() and m.name.split("/")[-1] == BASE_NAME and m.size == entry["bytes"]]
        if len(members) != 1:
            raise ValueError("Could not identify the base checkpoint in its archive")
        with source.extractfile(members[0]) as data:
            write_verified(target, entry["bytes"], entry["sha256"], lambda output: shutil.copyfileobj(data, output, BLOCK))
    return target


def write_verified(target, size, expected, fill):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size or sha256(target) != expected:
            raise ValueError(f"Existing file differs from its manifest: {target}")
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            fill(output)
        except BaseException:
            output.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        if temp.stat().st_size != size or sha256(temp) != expected:
            raise ValueError(f"Reconstructed file differs from its UMT SHA-256: {target}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def copy_bytes(stream, output, count):
    while count:
        data = stream.read(min(count, BLOCK))
        if not data:
            raise EOFError("Truncated input")
        output.write(data)
        count -= len(data)


def unshuffle(packed):
    length = len(packed)
    data = bytearray(length)
    start = 0
    for lane in range(8):
        stop = start + (length + 7 - lane) // 8
        data[lane::8] = packed[start:stop]
        start = stop
    return data


def decode(segment, base, capsule, weights, output):
    size = segment["bytes"]
    if "member" in segment:
        with capsule.open(segment["member"]) as data:
            copy_bytes(data, output, size)
        return
    ref = segment.get("base_offset")
    if ref is not None:
        base.seek(ref)
    if segment.get("same_as_base"):
        copy_bytes(base, output, size)
        return
    with weights.open(segment["id"]) as delta:
        while size:
            length = min(size, BLOCK)
            packed = delta.read(length)
            if len(packed) != length:
                raise EOFError("Truncated weight difference")
            data = unshuffle(packed)
            if ref is not None:
                original = base.read(length)
                if len(original) != length:
                    raise EOFError("Truncated base checkpoint")
                data = (int.from_bytes(data, "little") ^ int.from_bytes(original, "little")).to_bytes(length, "little")
            output.write(data)
            size -= length


def restore(case, catalog, base, output):
    with zipfile.ZipFile(MODELS / f"{case}.zip") as capsule, zipfile.ZipFile(MODELS / "weights" / f"grace-weights-{case}.zip") as weights:
        layout = json.loads(capsule.read("layout.json"))
        if layout["case"] != case or layout["chunk_bytes"] != BLOCK:
            raise ValueError(f"Invalid layout: {case}")
        expected = {row["remote_path"]: (int(row["bytes"]), row["sha256"]) for row in catalog}
        actual = {f["path"]: (f["bytes"], f["sha256"]) for f in layout["files"]}
        if actual != expected:
            raise ValueError(f"Layout and UMT inventory disagree: {case}")
        prior = {}
        segments = {seg["id"]: seg for seg in layout["segments"]}
        for entry in layout["files"]:
            target = (output / entry["path"]).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Invalid model path: {entry['path']}")

            def fill(stream):
                if "member" in entry:
                    with capsule.open(entry["member"]) as data:
                        copy_bytes(data, stream, entry["bytes"])
                    return
                for identifier in entry["segments"]:
                    segment = segments[identifier]
                    if identifier in prior:
                        earlier, offset = prior[identifier]
                        with earlier.open("rb") as data:
                            data.seek(offset)
                            copy_bytes(data, stream, segment["bytes"])
                    else:
                        decode(segment, base, capsule, weights, stream)

            write_verified(target, entry["bytes"], entry["sha256"], fill)
            offset = 0
            for identifier in entry.get("segments", []):
                prior[identifier] = (target, offset)
                offset += segments[identifier]["bytes"]
        print(f"{case}: verified {len(layout['files'])} original files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list", action="store_true")
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--case", action="append", metavar="NAME")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--base", type=Path, help="Base data file, its directory, original tar.gz, or the project's recovery ZIP")
    source.add_argument("--download-base", action="store_true", help="Fetch and verify the pinned Hugging Face archive")
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/grace-base", help="Cache directory for base files")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/grace-models", help="Root of the reconstructed UMT model tree")
    args = parser.parse_args()
    cases = {}
    with INDEX.open(encoding="utf-8", newline="") as table:
        for row in csv.DictReader(table):
            cases.setdefault(row["case"], []).append(row)
    if args.list:
        print("\n".join(sorted(cases)))
        return
    selected = sorted(cases) if args.all else list(dict.fromkeys(args.case))
    for case in selected:
        if case not in cases:
            parser.error(f"Unknown case {case}. Use --list")
    if not (args.base or args.download_base):
        parser.error("Choose --base or --download-base")
    card = json.loads(CARD.read_text(encoding="utf-8-sig"))
    with base_file(args, card).open("rb") as base:
        for case in selected:
            restore(case, cases[case], base, args.output.resolve())


if __name__ == "__main__":
    main()
