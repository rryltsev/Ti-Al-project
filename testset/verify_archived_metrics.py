"""Check Test50 hashes and reproduce the published raw MAE/RMSE offline."""

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
MONITOR = ROOT / "monitor"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(path):
    entries = read(path)
    for name, expected in entries.items():
        actual = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return len(entries)


def main():
    geometries = verify_manifest(ROOT / "TiAl-test50-v1" / "SHA256SUMS.json")
    predictions = verify_manifest(MONITOR / "prediction-manifest.json")
    bundle = read(MONITOR / "data" / "latest.json")
    report = read(MONITOR / "results" / "n49-f94a64dc4d95bca4" / "metrics.json")
    records = bundle["records"]
    ids = [row["metadata"]["structure_id"] for row in records]
    if ids != report["common_ids"] or len(set(ids)) != 49 or "TiAl-test-025" in ids:
        raise ValueError("Unexpected common subset")
    for xc, model in report["models"].items():
        for cache, metric_set in [(model.get("prediction_cache", xc), "metrics"),
                                  ("pretrained", "pretrained_metrics")]:
            energy_errors, force_errors = [], []
            for sid, row in zip(ids, records):
                reference = row["labels"][xc]
                pred = read(MONITOR / "predictions" / cache / (sid + ".json"))
                if row["symbols"] != pred["symbols"] or row["metadata"]["poscar_sha256"] != pred["poscar_sha256"]:
                    raise ValueError(f"Geometry mismatch: {xc} {sid}")
                energy_errors.append((pred["energy_eV"] - reference["energy_eV"]) * 1000 / len(row["symbols"]))
                for a, b in zip(pred["forces_eV_A"], reference["forces_eV_A"], strict=True):
                    force_errors.extend(x - y for x, y in zip(a, b, strict=True))
            for group, errs in [("energy_meV_atom", energy_errors), ("force_eV_A", force_errors)]:
                target = report[metric_set][xc]["all"][group]
                for key, value in [("mae", fmean(map(abs, errs))), ("rmse", math.sqrt(fmean(v * v for v in errs)))]:
                    if not math.isclose(value, target[key], rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"{xc}/{cache} {group} {key}: {value} vs {target[key]}")
    print(f"Verified {geometries} Test50 files, {predictions} predictions and 24 metrics on {len(ids)} common structures.")


if __name__ == "__main__":
    main()
