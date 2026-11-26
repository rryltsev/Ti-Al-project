"""Collect independently computed native ATAT ECI tables without rerunning a fit."""
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    out = Path(__file__).resolve().parent / "tables"
    out.mkdir(exist_ok=True)
    runs = out.parents[1] / "runs"
    sources = ("atat-verify-static-02", "atat-verify-vref-02", "atat-verify-volumes-02")
    thermal, static, checks, hashes = [], [], [], {}
    for run in sources:
        parent = runs / run / "outputs"
        archive = parent / "fits.tar.gz"
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        expected = (parent / "verified.sha256").read_text().splitlines()
        if f"{sha}  {archive.name}" not in expected:
            raise ValueError(f"Native outputs were not verified for {run}")
        hashes[run] = sha
        with tarfile.open(archive) as t:
            for item in t:
                if not item.name.endswith("/validation.json"):
                    continue
                key = item.name.rsplit("/", 1)[0]
                val = json.load(t.extractfile(item))
                ratio = val["volume_ratio"]
                with io.TextIOWrapper(t.extractfile(key + "/eci_native_order.csv")) as f:
                    rows = list(csv.DictReader(f))
                if len(rows) != 20 or [r["cluster"] for r in rows] != val["names_native"]:
                    raise ValueError(f"Native cluster order mismatch in {key}")
                dest = static if run == sources[0] else thermal
                dest.extend(dict(volume_ratio=ratio, **r) for r in rows)
                checks.append(dict(mode="static-only" if run == sources[0] else "shared_stability_mask",
                    volume_ratio=ratio, stable_train=val["n_train"], stable_control=val["n_control"],
                    max_ECI_disagreement_eV=val["native_vs_svd_max_abs_ECI_eV"],
                    max_native_prediction_disagreement_eV_atom=max(val["corrdump_vs_XJ_eV_atom"].values()),
                    control_RMSE_static_eV_atom=val["control_rmse_static_eV_atom"]))
    if len(static) != 20 or len(thermal) != 180 or len(checks) != 10:
        raise ValueError("Missing results from native fits")
    for name, rows in [("eci_thermal_grid.csv", thermal), ("eci_static_unmasked.csv", static),
                       ("native_validation.csv", checks)]:
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda row: row["volume_ratio"]))
    (out / "input_archives.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
