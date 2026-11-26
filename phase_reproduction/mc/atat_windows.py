"""Prepare VCSGC windows from a new Ti-Al ATAT/MTP fit at tabulated temperatures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"atomistic"))
from hcp_orbits import ORBIT_NAMES, VREF, build_orbits, features, sublattices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("grid", type=Path)
    p.add_argument("output", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--models", nargs="+", choices=["static","HA","QHA"])
    mode.add_argument("--s3", action="store_true", help="seven upper-temperature 5120-site S3 sections at 0.5 at.% spacing")
    p.add_argument("--temperatures", nargs="+", type=int, default=[1000])
    p.add_argument("--size", type=int, choices=[768,5120], default=768)
    p.add_argument("--spacing", type=int, choices=[1,2], default=2, help="window spacing in at.%%")
    args = p.parse_args()
    if args.s3:
        args.size = 5120
        sections = [(1600,"static"),(1625,"static"),(1650,"static"),
                    (1200,"HA"),(1225,"HA"),(1200,"QHA"),(1225,"QHA")]
        half_percent = range(4,73)
    else:
        sections = [(t,m) for t in args.temperatures for m in args.models]
        half_percent = range(4,73,2*args.spacing)
    orbit = build_orbits((4,4,3))
    d019 = 2*(sublattices((4,4,3))==0)-1
    ends = np.array([features(d019*0-1,orbit),features(d019,orbit)])
    args.output.mkdir(parents=True,exist_ok=False)
    rows = []
    with np.load(args.grid,allow_pickle=False) as fit:
        if tuple(fit["orbit_names"].astype(str)) != ORBIT_NAMES:
            raise ValueError("Unexpected source fit orbit order")
        vol = fit["volume_A3_atom"]
        if np.any(np.diff(vol)<=0):
            raise ValueError("unordered volume grid")
        ref = np.flatnonzero(np.isclose(vol,VREF,rtol=0,atol=1e-6))
        if len(ref)!=1:
            raise ValueError("missing Vref in fit")
        for t,model in sections:
            coef = fit["J_static"].copy()
            if model != "static":
                ti = np.flatnonzero(fit["temperature_K"]==t)
                if len(ti)!=1:
                    raise ValueError(f"missing phonon T={t}")
                coef += fit["J_vib"][:,ti[0],:]
            if model=="QHA" and len(vol)<3:
                raise ValueError("QHA needs a volume grid")
            vs,js = (vol,coef) if model=="QHA" else (vol[ref],coef[ref])
            if js.shape!=(len(vs),20) or not np.isfinite(js).all():
                raise ValueError("invalid coefficients")
            # Approximate tilt only; Eq. S5 uses the actual window composition.
            energies = (ends@js.T).min(axis=1)
            mu = float(4*(energies[1]-energies[0]))
            name = f"atat_{model}_{t}.txt"
            np.savetxt(args.output/name,np.column_stack([vs,js]),fmt="%.17g")
            for h in half_percent:
                dims = (8,8,6) if args.size==768 else (16,16,10)
                seed = 100000*t+1000*["static","HA","QHA"].index(model)+args.size
                seed += 500000000+h if args.s3 else h//2
                rows.append((f"atat_{model}",t,*dims,h/200,mu,seed,10000,40000,
                             "random",name,.045 if model=="QHA" else 0.))
    (args.output/"windows.tsv").write_text("".join("\t".join(map(str,row))+"\n" for row in rows),encoding="ascii")
    metadata = {"source":str(args.grid.resolve()),"sha256":hashlib.sha256(args.grid.read_bytes()).hexdigest(),
                "qha_measure":"flat dv at zero pressure, symmetric ±0.045 A³/atom once per sweep, natural cubic interpolation",
                "N":args.size,"n_windows":len(rows)}
    if args.s3:
        assert len(rows) == len(set(r[7] for r in rows)) == 483
        metadata.update(sections=sections, c0_range=[.02,.36], c0_step=.005, kappa_eV=100,
                        protocol="5120 sites, one random start per window, 10000 equilibration + 40000 production sweeps, stride 10",
                        c0_provenance="S3 states 0.005 spacing for static 1600 and 1650 K. Applied this spacing at other temperatures and used the main window range.")
    (args.output/"fit_source.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
