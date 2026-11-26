"""Restore original window trajectories from the downloaded UMT result archives."""
import argparse
import csv
import hashlib
from pathlib import Path
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(row):
    archive = ROOT / row["archive"]
    folder = ROOT / row["series"]
    if sha256(archive) != row["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if not member.isfile() or member.name == "inputs.sha256":
                continue
            if Path(member.name).name != member.name or ":" in member.name or "\\" in member.name:
                raise ValueError(f"Unexpected window filename: {member.name}")
            path = folder / member.name
            data = stream.extractfile(member).read()
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(f"Existing file differs from archive: {path}")
            else:
                with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
                    tmp.write(data)
                try:
                    Path(tmp.name).replace(path)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", nargs="+", choices=["all", "solvus_1000", "solvus_temperatures",
                                                     "solvus_closure", "solvus_cage2304"])
    args = parser.parse_args()
    with (ROOT / "series_archives.csv").open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if "all" in args.study or row["series"].split("/")[0] in args.study]
    missing = [row["archive"] for row in rows if not (ROOT / row["archive"]).is_file()]
    if missing:
        parser.exit(1, f"Missing {len(missing)} archives, starting with {missing[0]}. "
                       "Extract the corresponding phase data ZIP into the repository root first.\n")
    count = sum(restore(row) for row in rows)
    print(f"Verified {len(rows)} archives and {count} files in the window series.")


if __name__ == "__main__":
    main()
