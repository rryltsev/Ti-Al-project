"""Restore numerical research inputs or the dated manuscript from a clone."""
import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def hash_file(path):
    with path.open("rb") as stream:
        return sha(stream)


def read_index(name):
    with (ROOT / "docs" / name).open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def destination(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid member path {name}")
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path leaves destination: {name}")
    return target


def restore(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as out:
        try:
            shutil.copyfileobj(source, out)
        except BaseException:
            out.close()
            target.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--data", metavar="ARCHIVE_PREFIX")
    group.add_argument("--edition", metavar="NEW_DIRECTORY", type=Path)
    options = parser.parse_args()
    if options.edition:
        folder = options.edition.resolve()
        if folder.exists():
            parser.error("Choose a new output directory for the edition")
        edition = read_index("manuscript-2026-11-03.csv")
        for row in edition:
            file = destination(ROOT, row["repository_path"])
            if hash_file(file) != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {file}")
        for row in edition:
            with destination(ROOT, row["repository_path"]).open("rb") as source:
                restore(destination(folder, row["edition_path"]), source)
        print(f"Restored {len(edition)} edition files in {folder}")
        return
    inventory = read_index("research-data-archives.csv")
    if options.list:
        for row in inventory:
            print(f"{Path(row['repository_path']).name}  {row['members']} files")
    else:
        rows = [r for r in inventory if Path(r["repository_path"]).name.startswith(options.data)]
        if not rows:
            parser.error("Unknown archive prefix. Use --list to see available names.")
        for row in rows:
            path = destination(ROOT, row["repository_path"])
            if hash_file(path) != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    target = destination(ROOT, name)
                    # Existing originals are common when restoring on the source machine.
                    if target.exists():
                        with archive.open(name) as source:
                            if hash_file(target) != sha(source):
                                raise ValueError(f"Existing file differs: {target}")
                    else:
                        with archive.open(name) as source:
                            restore(target, source)
            print(f"Restored {path.name}")


if __name__ == "__main__":
    main()
