"""Re-run measured slow S3 windows with independent random and D019-like starts."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("selection", type=Path, help="CSV with model,T_K,c0,start_dependence from the S3 review")
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    source = Path(__file__).resolve().parent / "inputs" / "atat_s3_5120"
    by_key = {}
    with (source / "windows.tsv").open() as f:
        for r in csv.reader(f, delimiter="\t"):
            by_key[r[0], int(r[1]), float(r[5])] = r
    records = list(csv.DictReader(args.selection.open(newline="")))
    rows, files, prior, isotherms = [], set(), set(), set()
    for rec in records:
        key = rec["model"], int(rec["T_K"]), float(rec["c0"])
        if key[:2] in isotherms:
            raise ValueError(f"Only one S3 repeat centre per isotherm: {key}")
        diagnostics = source.parents[2] / "thermo" / "results" / "atat_s3" / f"{key[0]}_{key[1]}" / "window_correlations.csv"
        with diagnostics.open(newline="") as f:
            matched = [r for r in csv.DictReader(f) if float(r["c0"]) == key[2] and
                       r["model"] == key[0] and int(float(r["T_K"])) == key[1] and int(r["N"]) == 5120]
        if len(matched) != 1:
            raise ValueError(f"Missing or duplicate diagnostic record for {key}")
        rec["tau_order"] = matched[0]["tau_int_order_frames"]
        rec["tau_c"] = matched[0]["tau_int_count_frames"]
        rec["diagnostics_sha256"] = hashlib.sha256(diagnostics.read_bytes()).hexdigest()
        if max(float(rec["tau_order"]), float(rec["tau_c"])) < 200 and rec["start_dependence"] != "true":
            raise ValueError("Slow window lacks qualifying diagnostics")
        isotherms.add(key[:2])
        base = by_key[key]
        prior.add(int(base[7]))
        for mode, offset in [("random", 200000000), ("ordered", 250000000)]:
            r = base.copy()
            r[7:11] = [str(int(base[7]) + offset), "50000", "400000", mode]
            rows.append(r)
        files.add(base[11])
    seeds = {int(row[7]) for row in rows}
    if not rows or len(seeds) != len(rows) or seeds & prior:
        raise ValueError("Missing or duplicated repeats")
    args.output.mkdir(exist_ok=False, parents=True)
    for name in sorted(files):
        shutil.copyfile(source / name, args.output / name)
    shutil.copyfile(args.selection, args.output / "selection.csv")
    with (args.output / "windows.tsv").open("w", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)
    metadata = json.loads((source / "fit_source.json").read_text())
    metadata.update(n_windows=len(rows), starts=["random", "ordered"], equil=50000, production=400000,
                    selection=records, window_rule="one measured slow centre per isotherm",
                    baseline_sha256=hashlib.sha256((source/"windows.tsv").read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    (args.output/"fit_source.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
