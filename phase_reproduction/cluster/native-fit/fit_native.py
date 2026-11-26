"""Fit all stable training labels with ATAT lsfit and read energies with corrdump."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np


def read_labels(path):
    data = {}
    with tarfile.open(path) as tar:
        for member in tar:
            p = Path(member.name)
            if not member.isfile() or p.name not in ("label.json", "vibrations.csv"):
                continue
            data[p.parent.name, p.name] = tar.extractfile(member).read().decode()
    return data


def run_tool(args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, text=True, timeout=120).stdout


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--ratio", type=float, required=True)
    cli.add_argument("--lsfit", required=True)
    cli.add_argument("--corrdump", required=True)
    cli.add_argument("--static-only", action="store_true")
    args = cli.parse_args()
    # Use Unix line endings in ephemeral native working copies. ATAT has
    # custom whitespace parsing when reading lists of clusters and cells.
    for name in ("lat.in", "clusters.out", "selected.out"):
        src = Path(name)
        src.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
    design = list(csv.DictReader(Path("design.csv").open()))
    names = json.loads(Path("design.json").read_text())["native_names"]
    x = np.array([[float(r[n]) for n in names] for r in design])
    labels = read_labels(args.archive)
    use, train, targets, temp = [], [], [], None
    for i, r in enumerate(design):
        ident = f"{r['id']}_v{args.ratio:.2f}"
        label = json.loads(labels[ident, "label.json"])
        if label["source_cell_sha256"] != r["cell_sha256"] or label["fmax_eV_A"] > 1e-5:
            raise ValueError(f"Source cell or relaxation mismatch in {ident}")
        if not args.static_only and label["stable_mesh"] is False:
            continue
        y = [label["static_eV_atom"]]
        if not args.static_only:
            if label["stable_mesh"] is not True:
                raise ValueError(f"Missing phonon stability in {ident}")
            table = list(csv.DictReader(io.StringIO(labels[ident, "vibrations.csv"])))
            t = [float(v["temperature_K"]) for v in table]
            if temp is not None and temp != t:
                raise ValueError("Mismatch in temperature grid")
            temp = t
            y.extend(float(v["Fvib_eV_atom"]) for v in table)
        targets.append(y)
        use.append(i)
        train.append(r["split"] == "train")
    train = np.asarray(train)
    xu, y = x[use], np.array(targets)
    if np.linalg.matrix_rank(xu[train]) != len(names) or sum(train) <= len(names) or np.all(train):
        raise ValueError("Need an overdetermined training set and held-out labels")
    np.savetxt("x.dat", xu[train], fmt="%.17g")
    # ATAT 3.52 lsfit prints five fixed decimals in its original source.
    # Scaling the right-hand side retains sub-meV precision after division.
    unit_scale = 1e8
    np.savetxt("y_eV_times_1e8.dat", y[train] * unit_scale, fmt="%.17g")
    raw = run_tool([args.lsfit, "-x=x.dat", "-y=y_eV_times_1e8.dat"])
    Path("lsfit.out").write_text(raw)
    coef = np.fromstring(raw, sep=" ").reshape(y.shape[1], len(names)).T / unit_scale
    svd = np.linalg.lstsq(xu[train], y[train], rcond=None)[0]
    diff = float(np.max(np.abs(svd-coef)))
    if not np.isfinite(diff) or diff > 1e-8:
        raise ValueError(f"Native and SVD coefficients differ by {diff}")
    with Path("eci_native_order.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "J_static_eV_per_cluster", *[f"Jvib_{int(t)}K" for t in temp or []]])
        w.writerows([n, *row] for n, row in zip(names, coef))
    slices = {"static": coef[:, 0]}
    if temp and 1000. in temp:
        slices["total_1000K"] = coef[:, 0] + coef[:, 1+temp.index(1000.)]
    errors = {}
    for name, j in slices.items():
        eci = name + ".eci.out"
        # Native multiplicities use the two-site parent cell. Use J/2 to
        # return eV/atom for arbitrary supercell sizes.
        np.savetxt(eci, j / 2, fmt="%.17g")
        raw = run_tool([args.corrdump, "-c", "-l=lat.in", "-s=selected.out", "-eci="+eci, "-sig=14"])
        Path(name+".predicted.out").write_text(raw)
        predicted = np.fromstring(raw, sep=" ")
        if predicted.shape != (len(x),):
            raise ValueError("Native prediction count mismatch")
        err = float(np.max(abs(predicted - x @ j)))
        if not np.isfinite(err) or err > 1e-9:
            raise ValueError(f"Native corrdump energies differ from XJ by {err}")
        errors[name] = err
    resid = xu @ coef - y
    report = dict(volume_ratio=args.ratio, n_train=int(sum(train)), n_control=int(sum(~train)),
                  native_vs_svd_max_abs_ECI_eV=diff, corrdump_vs_XJ_eV_atom=errors,
                  control_rmse_static_eV_atom=float(np.mean(resid[~train, 0]**2)**.5),
                  names_native=names, targets=["static", *(temp or [])], unit_scale=unit_scale)
    Path("validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
