"""Restore final Adam optimizer states from the matching GRACE checkpoints."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

from restore_grace_models import ROOT, copy_bytes, sha256, write_verified


CATALOG = ROOT / "docs/grace-optimizers.csv"
PACKAGES = ROOT / "research/grace-final/optimizers"


def restore(row, models, destination):
    package = PACKAGES / row["package"]
    weights = models / row["weights"]
    if sha256(package) != row["package_sha256"]:
        raise ValueError(f"Optimizer package checksum mismatch: {package}")
    if not weights.is_file() or sha256(weights) != row["weights_sha256"]:
        raise ValueError(f"Restore the matching final GRACE checkpoint first: {weights}")

    target = destination / row["case"]
    with zipfile.ZipFile(package) as archive, weights.open("rb") as reference:
        recipe = json.loads(archive.read("recipe.json"))

        def restore_data(output):
            for segment in recipe:
                if "offset" in segment:
                    reference.seek(segment["offset"])
                    copy_bytes(reference, output, segment["bytes"])
                else:
                    with archive.open(segment["literal"]) as fragment:
                        copy_bytes(fragment, output, segment["bytes"])

        def restore_index(output):
            with archive.open("checkpoint.index") as original:
                shutil.copyfileobj(original, output)

        write_verified(target / "checkpoint.data-00000-of-00001", int(row["bytes"]), row["sha256"], restore_data)
        write_verified(target / "checkpoint.index", int(row["index_bytes"]), row["index_sha256"], restore_index)
    print(f"Verified {row['case']}: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list", action="store_true", help="List the available runs")
    choice.add_argument("--all", action="store_true", help="Restore all available optimizer states")
    choice.add_argument("--case", action="append", metavar="NAME", help="Restore a run, repeat for several")
    parser.add_argument("--models", type=Path, default=ROOT / "tmp/grace-models", help="Output tree of restore_grace_models.py")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/grace-optimizers")
    args = parser.parse_args()
    with CATALOG.open(encoding="utf-8", newline="") as table:
        rows = {r["case"]: r for r in csv.DictReader(table)}
    if args.list:
        print("\n".join(sorted(rows)))
        return
    selected = sorted(rows) if args.all else list(dict.fromkeys(args.case))
    unknown = sorted(set(selected) - rows.keys())
    if unknown:
        parser.error(f"Unknown cases: {', '.join(unknown)}")
    for case in selected:
        restore(rows[case], args.models.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
