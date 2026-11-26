"""List, download and verify selected archives from the Ti–Al project release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fnmatch
import hashlib
from http.client import IncompleteRead
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "release-assets.json"
BASE = "https://github.com/rryltsev/Ti-Al-project/releases/download"
BLOCK = 1024 * 1024


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, record):
    return path.is_file() and path.stat().st_size == record["bytes"] and hash_file(path) == record["sha256"]


def local_path(folder, name):
    if not name or name in {".", ".."} or any(ch in name for ch in "/\\:"):
        raise ValueError(f"Unexpected release filename {name!r}")
    return folder / name


def fetch_object(row, folder, tag):
    target = local_path(folder, row["name"])
    if target.exists():
        if matches(target, row):
            return target
        raise ValueError(f"Existing file differs from the catalog: {target}")
    partial = target.with_name(target.name + ".part")
    url = f"{BASE}/{quote(row.get('tag', tag), safe='')}/{quote(target.name, safe='')}"
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        if matches(partial, row):
            break
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= row["bytes"]:
            raise ValueError(f"Invalid partial download; move it aside and retry: {partial}")
        headers = {"User-Agent": "Ti-Al-project-archive/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                append = response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError(f"Unexpected response range for {target.name}")
                with partial.open("ab" if append else "wb") as stream:
                    shutil.copyfileobj(response, stream, BLOCK)
        except (URLError, OSError, IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code in {403, 404}:
                raise
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if not matches(partial, row):
        raise ValueError(f"Incomplete download or SHA-256 mismatch: {partial}")
    os.replace(partial, target)
    return target


@contextmanager
def archive_lock(folder, name):
    locks = folder / ".parts" / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (name + ".lock")).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            while True:
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except PermissionError:
                    time.sleep(0.5)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Closing the file releases the OS lock, including on process exit.
        yield


def assemble(row, folder, tag):
    target = local_path(folder, row["name"])
    if "parts" not in row:
        return fetch_object(row, folder, tag)
    if target.exists():
        if matches(target, row):
            return target
        raise ValueError(f"Existing file differs from the catalog: {target}")
    cache = folder / ".parts" / target.name

    def fetch(part):
        return fetch_object(part, cache, row.get("tag", tag))

    with ThreadPoolExecutor(max_workers=3) as workers:
        parts = list(workers.map(fetch, row["parts"]))
    output = tempfile.NamedTemporaryFile(dir=folder, prefix=target.name, suffix=".assembling", delete=False)
    assembling = Path(output.name)
    try:
        with output:
            for part in parts:
                with part.open("rb") as src:
                    shutil.copyfileobj(src, output, BLOCK)
        if not matches(assembling, row):
            raise ValueError(f"Assembled file differs from the catalog: {target.name}")
        os.replace(assembling, target)
    finally:
        assembling.unlink(missing_ok=True)
    for part in parts:
        part.unlink()
    if not any(cache.iterdir()):
        cache.rmdir()
    return target


def download(row, folder, tag):
    target = local_path(folder, row["name"])
    if matches(target, row):
        return target
    with archive_lock(folder, target.name):
        return assemble(row, folder, tag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="show filenames, sizes and groups")
    mode.add_argument("--get", action="store_true", help="fetch selected archives")
    mode.add_argument("--verify", action="store_true", help="check local archives without network access")
    parser.add_argument("--group", action="append", help="group to select (repeatable)")
    parser.add_argument("--name", action="append", help="filename or glob (repeatable)")
    parser.add_argument("--all", action="store_true", help="select the complete release")
    parser.add_argument("--dest", type=Path, default=ROOT / "downloads")
    args = parser.parse_args()
    if (args.get or args.verify) and not (args.name or args.group or args.all):
        parser.error("Select archives with --name, --group or --all.")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    rows = [r for r in catalog["assets"] if (not args.group or r["group"] in args.group)
            and (not args.name or any(fnmatch.fnmatchcase(r["name"], p) for p in args.name))]
    if not rows:
        parser.error("No matching archives. Use --list to view the catalog.")
    for row in rows:
        if args.get:
            print(download(row, args.dest, catalog["tag"]), flush=True)
        elif args.verify:
            if not matches(local_path(args.dest, row["name"]), row):
                raise SystemExit(f"Missing or incorrect: {row['name']}")
            print("OK", row["name"])
        else:
            print(f"{row['group']:18} {row['bytes'] / 2**20:9.2f} MiB  {row['name']}")


if __name__ == "__main__":
    main()
