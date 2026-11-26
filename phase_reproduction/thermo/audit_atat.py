"""Compare the native ATAT cluster representatives to the finite-lattice D019 MFA basis."""
import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"atomistic"))
from hcp_orbits import ORBIT_NAMES, MULTIPLICITIES, build_orbits, sublattices


def audit(path, output):
    if path.is_dir():
        read=lambda n: (path/n).read_text()
    else:
        with tarfile.open(path,"r:gz") as tar:
            files={Path(m.name).name: tar.extractfile(m).read().decode()
                   for m in tar if m.isfile() and Path(m.name).name in ("clusters.out","lat.in")}
        read=files.__getitem__
    lines=read("lat.in").splitlines()
    axes=np.loadtxt(io.StringIO("\n".join(lines[:3])))
    cell=np.loadtxt(io.StringIO("\n".join(lines[3:6])))
    if not np.allclose(cell,np.eye(3)) or "Ti,Al" not in lines[6]:
        raise ValueError("This audit uses the v16 normalized HCP basis and Ti,Al spin convention")
    # Parse counted records; native clusters.out may use blank lines between records.
    entries=iter(filter(None,(s.strip() for s in read("clusters.out").splitlines())))
    records=[]
    for mul in entries:
        mult=float(mul)
        _diam=float(next(entries))
        count=int(next(entries))
        coords=np.array([[float(x) for x in next(entries).split()[:3]] for _ in range(count)])
        distances=sorted(np.linalg.norm((coords[k]-coords[l])@axes)
                         for k in range(count) for l in range(k))
        records.append((mult,coords,distances))
    shells=sorted(set(round(e[2][0],8) for e in records if len(e[1])==2))[:7]
    if len(shells)!=7:
        raise ValueError("Missing HCP pair shells")
    shape=(4,4,4)
    labels=sublattices(shape)
    orbit_arrays=build_orbits(shape)
    checked={}
    for index,(mult,pos,distances) in enumerate(records):
        length=len(pos)
        if length>=4:
            continue
        if distances:
            shell=[]
            for d in distances:
                close=np.flatnonzero(np.isclose(shells,d,atol=1e-7,rtol=1e-7))
                if len(close)!=1:
                    break
                shell.append(int(close[0])+1)
            if len(shell)!=len(distances):
                continue
            name=("p" if length==2 else "t")+"".join(map(str,sorted(shell)))
        else:
            name="constant" if length==0 else "point"
        if name=="t222":
            base=int(round(2*pos[0,2]))%2
            offset=pos[:,:2].mean(axis=0)-(1-base)/3
            name+="_1" if np.allclose(offset,np.rint(offset),atol=1e-7) else "_2"
        if name not in ORBIT_NAMES:
            continue
        if name in checked:
            raise ValueError(f"Ambiguous assignment for {name}")
        idx=ORBIT_NAMES.index(name)
        if abs(mult/2-MULTIPLICITIES[idx])>1e-9:
            raise ValueError(f"Unexpected ATAT multiplicity {name}: {mult}")
        if length:
            base=np.rint(2*pos[:,2]).astype(int)%2
            ij=pos[:,:2]-base[:,None]/3
            if np.max(abs(ij-np.rint(ij)))>1e-6:
                raise ValueError("Coordinate is off the HCP parent")
            parity=(np.rint(ij).astype(int)+base[:,None])%2
            colors=2*parity[:,0]+parity[:,1]
            counts=np.bincount(colors,minlength=4)
            direct=np.bincount(counts,minlength=length+1)/4
            # Each infinite-lattice representative generates four translationally
            # equivalent choices for the preferred D019 color.
            site_counts=(labels[orbit_arrays[idx]]==0).sum(axis=1)
            explicit=np.bincount(site_counts,minlength=length+1)/len(site_counts)
            np.testing.assert_allclose(direct,explicit,atol=1e-14,rtol=0)
        else:
            direct=np.ones(1)
        checked[name]={"native_index_0based":index,"multiplicity_per_primitive":mult,
                       "probability_by_preferred_site_count":direct.tolist()}
    if set(checked)!=set(ORBIT_NAMES):
        raise ValueError(f"Missing native orbits: {set(ORBIT_NAMES)-set(checked)}")
    hashes={name:hashlib.sha256(read(name).encode()).hexdigest() for name in ("lat.in","clusters.out")}
    result=dict(source_text_sha256=hashes,shell_radii=shells,matched=checked)
    if path.is_dir() and (path/"design.csv").is_file():
        native=sorted(ORBIT_NAMES,key=lambda name:checked[name]["native_index_0based"])
        largest=0.
        correlation_rows=[]
        for row in csv.DictReader((path/"design.csv").open(encoding="utf8")):
            c=int(row["N_Al"])/int(row["N"])
            source=path/"selected"/row["id"]
            if hashlib.sha256((source/"str.out").read_bytes()).hexdigest()!=row["str_sha256"]:
                raise ValueError("Broken str.out provenance")
            rho=np.loadtxt(source/"corr.out")
            expected=np.array([float(row[s]) for s in native])
            observed=rho*np.array([checked[s]["multiplicity_per_primitive"]/2 for s in native])
            largest=max(largest,float(np.max(abs(expected-observed))))
            np.testing.assert_allclose(rho[native.index("point")],2*c-1,atol=1e-12)
            correlation_rows.append((row["split"],expected))
        if largest>1e-10:
            raise ValueError("Per-atom normalization mismatch")
        train=np.array([v for split,v in correlation_rows if split=="train"])
        result.update(selected=len(correlation_rows),train=len(train),rank=int(np.linalg.matrix_rank(train)),
                      max_abs_weighted_corr_error=largest)
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive_or_directory",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    audit(args.archive_or_directory,args.output)
