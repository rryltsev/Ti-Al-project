"""Prepare controlled, fixed-volume 1000 K VCSGC substitutions using archived inputs."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
MC = HERE.parents[1] / "mc"
ORBIT_NAMES = ("constant", "point", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
               "t112", "t114", "t117", "t123", "t125", "t136", "t222_1",
               "t222_2", "t226", "t247", "t334")
SOURCES = {
    ("paper", "static"): ("section5120", "static_S1", "static_S1.txt"),
    ("paper", "HA"): ("section5120", "HA_fig2", "HA_fig2.txt"),
    ("masked", "static"): ("atat_static_section", "atat_static", "atat_static_1000.txt"),
    ("masked", "HA"): ("atat_ha_section", "atat_HA", "atat_HA_1000.txt"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def endpoints():
    values, metadata = {}, {}
    with tarfile.open(MC / "runs/paper_reconstruction_inputs.tar.gz") as archive:
        for (base, model), (run, label, filename) in SOURCES.items():
            folder = MC / "runs" / run
            receipts = json.loads((folder / "inputs.json").read_text())["sha256"]
            data = (archive.extractfile(run + "/" + filename).read() if base == "paper" else
                    (MC / "inputs" / run / filename).read_bytes().replace(b"\r\n", b"\n"))
            windows = (folder / "windows.tsv").read_bytes().replace(b"\r\n", b"\n")
            if digest(data) != receipts[filename] or digest(windows) != receipts["windows.tsv"]:
                raise ValueError(f"Historical input hash disagrees for {run}/{filename}")
            grid = list(map(float, data.split()))
            if len(grid) != 21 or grid[0] != 17.288329847774033:
                raise ValueError("Unexpected reference volume or coefficient count")
            rows = [r for r in csv.reader(io.StringIO(windows.decode()), delimiter="\t") if r[0] == label]
            if (len(rows) != 35 or [float(r[5]) for r in rows] != [i/100 for i in range(2, 37)] or
                    any(r[1:5] != ["1000", "16", "16", "10"] or
                        r[8:11] != ["10000", "40000", "random"] for r in rows) or len({r[6] for r in rows}) != 1):
                raise ValueError(f"Unexpected control parameters in {run}")
            combined = json.loads((folder / "retrieved.json").read_text())
            with gzip.open(folder / "samples.csv.gz", "rt") as stream:
                fields = next(csv.reader(stream))
            values[base, model] = grid[1:]
            metadata[base + "_" + model] = dict(directory=run, label=label, file=filename,
                coefficient_sha256=receipts[filename], windows_sha256=receipts["windows.tsv"],
                delta_mu0_eV=float(rows[0][6]), samples_sha256=combined["sha256_combined"],
                job=combined["job_id"], fields=fields)
    return values, metadata


def make_input(plan, dest, values, metadata):
    files = {}
    windows = []
    records = []
    specs = plan["models"]
    if not specs:
        raise ValueError("Select at least one intervention")
    names = set()
    for n, spec in enumerate(specs):
        name, base, model = (spec[k] for k in ("name", "base", "model"))
        other = {"paper": "masked", "masked": "paper"}[base]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name) or name in names:
            raise ValueError(f"Duplicate or invalid model name: {name}")
        names.add(name)
        base_static, target_static = values[base, "static"], values[other, "static"]
        baseline, target = values[base, model], values[other, model]
        new = list(baseline)
        used = set()
        for component, orbit_list in spec["replace"].items():
            if component not in ("static", "vib", "effective") or (model == "static" and component == "vib"):
                raise ValueError(f"Invalid component {component} in {model}")
            for orbit in orbit_list:
                i = ORBIT_NAMES.index(orbit)
                if i < 2:
                    raise ValueError("Affine ECI do not change fixed-volume coexistence compositions")
                channels = {("static", i), ("vib", i)} if component == "effective" else {(component, i)}
                if used & channels:
                    raise ValueError(f"Overlapping substitutions of {orbit}")
                used.update(channels)
                if component == "effective":
                    new[i] = target[i]
                elif component == "static":
                    new[i] += target_static[i] - base_static[i]
                else:
                    new[i] += (target[i]-target_static[i]) - (baseline[i]-base_static[i])
        if new == baseline:
            raise ValueError(f"Unchanged intervention: {name}")
        fname = name + ".txt"
        files[fname] = " ".join(format(x, ".17g") for x in (17.288329847774033, *new))+"\n"
        mu = metadata[base+"_"+model]["delta_mu0_eV"]
        for center in range(2, 37):
            seed = plan["seed_offset"] + 100*n + center
            if not 0 < seed < 2**63:
                raise ValueError("Invalid seed")
            windows.append((name, 1000, 16, 16, 10, center/100, mu, seed, 10000, 40000,
                            "random", fname, 0.0, "orbit-sums"))
        records.append(dict(**spec, source_other=other, J_eV=new,
            delta_J_eV=[x-y for x,y in zip(new,baseline)], delta_mu0_eV=mu, coefficient_sha256=digest(files[fname].encode())))
    files["windows.tsv"] = "".join("\t".join(map(str, r))+"\n" for r in windows)
    dest.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (dest/name).write_bytes(data.encode("ascii"))
    result = dict(orbit_names=ORBIT_NAMES, multiplicity="coefficients per cluster; sums count clusters once",
        baseline_inputs=metadata, variants=records, seed_offset=plan["seed_offset"],
        n_windows=len(windows), dimensionless_kappa=100, stride=10, N=5120,
        windows_sha256=digest(files["windows.tsv"].encode()))
    (dest/"metadata.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, help="write the four historical endpoint receipts")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.plan is None) != (args.output is None) or not (args.plan or args.audit):
        parser.error("provide --audit FILE and/or --plan JSON --output NEW_DIRECTORY")
    values, metadata = endpoints()
    if args.audit:
        args.audit.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf8")
    if args.plan:
        make_input(json.loads(args.plan.read_text(encoding="utf8")), args.output, values, metadata)
