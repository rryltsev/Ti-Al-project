"""Select a frozen Ti-rich design from native ATAT genstr and corrdump output."""
import argparse
import csv
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hcp_orbits import (MULTIPLICITIES, ORBIT_NAMES, STATIC_S1, VOLUME_RATIOS,
                        build_orbits, features, features_for_parent_sites, hcp_geometry, sublattices)
from mtp_cfg import Frame, write_cfg


AXES = np.array([[1.,0,0], [.5,3**.5/2,0], [0,0,1.6]])
DISTANCES = np.sqrt([.25*1.6**2+1/3, 1., .25*1.6**2+4/3, 1.6**2,
                     .25*1.6**2+7/3, 3., 1+1.6**2])


def orbit_name(record):
    lines = record.strip().splitlines()
    order = int(lines[2])
    if order == 0:
        return "constant"
    if order == 1:
        return "point"
    sites = np.array([r.split()[:3] for r in lines[3:3+order]], dtype=float)
    lengths = [np.linalg.norm((p-q) @ AXES) for p,q in combinations(sites, 2)]
    shells = sorted(int(np.argmin(abs(DISTANCES - d)))+1 for d in lengths)
    if any(np.min(abs(DISTANCES - d)) > 1e-6 for d in lengths):
        return None
    name = ("p" if order == 2 else "t") + "".join(map(str, shells))
    if name == "t222":
        b = int(round(2*sites[0,2])) % 2
        center = sites[:,:2].mean(axis=0) - (1-b)/3
        name += "_1" if np.allclose(center, np.rint(center), atol=1e-8) else "_2"
    return name


def geometry(raw):
    lines = [s.split() for s in raw.strip().splitlines() if s.strip()]
    if len(lines[0]) == 6:  # ATAT writes axis lengths/angles by default in some builds.
        if not np.allclose(np.array(lines[0],float), [1,1,1.6,90,90,60], atol=1e-8):
            raise ValueError("Unexpected coordinate axes")
        start=1
    else:
        if not np.allclose(np.array(lines[:3],float), AXES, atol=1e-8):
            raise ValueError("Unexpected coordinate axes")
        start=3
    raw_matrix = np.array(lines[start:start+3], float)
    matrix = np.rint(raw_matrix).astype(int)
    if not np.allclose(matrix, raw_matrix, atol=1e-8):
        raise ValueError("Noninteger supercell")
    if np.linalg.det(matrix) < 0:
        matrix[[0,1]] = matrix[[1,0]]
    rows = lines[start+3:]
    frac = np.array([r[:3] for r in rows], float)
    b = np.rint(2*frac[:,2]).astype(int) % 2
    indices = frac - b[:,None]*[1/3,1/3,.5]
    integer = np.rint(indices).astype(int)
    if not np.allclose(indices, integer, atol=1e-8):
        raise ValueError("Fail mapping native genstr output to ideal HCP sites")
    typ = np.array([{"Ti":0,"Al":1}[r[3]] for r in rows])
    prim, _ = hcp_geometry((1,1,1))
    return Frame(matrix@prim, frac@prim, typ), matrix, np.column_stack([integer,b])


def choose_design(x, compositions, n_native, fixed_train, fixed_control, eligible):
    unique = []
    seen = set()
    seen.update(tuple(np.round(x[i],10)) for i in fixed_train+fixed_control)
    if len(seen) != len(fixed_train)+len(fixed_control):
        raise ValueError("Duplicate fixed correlations")
    for i,(vec,c) in enumerate(zip(x[:n_native],compositions[:n_native])):
        key = tuple(np.round(vec, 10))
        if eligible[i] and 0 <= c <= .35 and key not in seen:
            unique.append(i)
            seen.add(key)
    unique = np.array(unique, int)
    # Ensure D019 and HCP-Ti are present in the selected native set.
    d = features(np.where(sublattices((2,2,1))==0,1,-1), build_orbits((2,2,1)))
    hits = unique[np.max(abs(x[unique]-d),axis=1) < 1e-8]
    if len(hits) != 1:
        raise ValueError("D019 not recovered from genstr")
    train = [int(unique[compositions[unique]==0][0]), int(hits[0]), *fixed_train]
    control = list(fixed_control)
    # Reserve seven geometrically distinct holdouts a priori, spanning composition.
    # Seed breaks degeneracy among different decorations of a given rational composition.
    rng = np.random.default_rng(20261124)
    for target in ([.10,1/6,.20,.25,.30] if fixed_control else [.10,.125,1/6,.20,.25,.30,1/3]):
        available = np.setdiff1d(unique, train+control)
        dist = abs(compositions[available]-target)
        control.append(int(rng.choice(available[np.isclose(dist,dist.min())])))
    xnorm = x / np.maximum(np.sqrt(np.mean(x[unique]**2,axis=0)),1e-8)
    while len(train) < 39:
        available = np.setdiff1d(unique, train+control)
        a = xnorm[train]
        inverse = np.linalg.inv(a.T@a + 1e-6*np.eye(x.shape[1]))
        leverage = np.einsum("ij,jk,ik->i",xnorm[available],inverse,xnorm[available])
        train.append(int(available[np.argmax(leverage)]))
    rank = np.linalg.matrix_rank(x[train])
    if rank != len(ORBIT_NAMES):
        raise ValueError(f"Only {rank}/20 independent combinations; increase the genstr atom limit")
    return train,control,unique


def repeats(cell, n):
    heights = 1 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    candidates = np.array(list(product(range(1,17), repeat=3)))
    counts = np.prod(candidates, axis=1)*n
    mask = (counts >= 504) & (counts <= 1728) & ((candidates*heights).min(axis=1)*min(VOLUME_RATIOS)**(1/3)>19.2)
    rep = candidates[mask]
    if len(rep)==0:
        return None
    extents = rep*heights
    score = np.abs(np.log(counts[mask]/768)) + np.log(extents.max(axis=1)/extents.min(axis=1))
    return rep[np.argmin(score)].tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fixed", type=Path, help="six named native MC-SQS cells, each with str.out and full corr.out")
    args = p.parse_args()
    with tarfile.open(args.archive,"r:gz") as arc:
        contents = {name:arc.extractfile(name).read() for name in
                    ["lat.in","clusters.out","sym.out","enum.out","correlations.out","binaries.sha256","job.sh"]}
    raw = [s.strip()+"\n" for s in re.split(r"(?m)^end\s*$",contents["enum.out"].decode()) if s.strip()]
    rho = np.loadtxt(contents["correlations.out"].decode().splitlines(),ndmin=2)
    blocks = re.split(r"\n\s*\n",contents["clusters.out"].decode().strip())
    names = [orbit_name(s) for s in blocks]
    columns = []
    for name,expected in zip(ORBIT_NAMES,MULTIPLICITIES):
        if names.count(name)!=1:
            raise ValueError(f"Expected a single cluster for {name}, got {names.count(name)}")
        idx = names.index(name)
        if int(blocks[idx].splitlines()[0]) != int(2*expected):
            raise ValueError(f"ATAT multiplicity differs for {name}")
        columns.append(idx)
    if rho.shape != (len(raw),len(names)):
        raise ValueError(f"Mismatch in structure/correlation output sizes: {len(raw)}, {rho.shape}")
    n_native = len(raw)
    origin = [f"genstr:{i+1}" for i in range(n_native)]
    fixed_train, fixed_control = [], []
    if args.fixed:
        for al, split in [(17,"train"), (20,"train"), (22,"control")]:
            for state in ["random","preferred"]:
                key = f"{split}{al}_{state}"
                path=args.fixed/key
                text = (path/"str.out").read_text().strip()+"\n"
                frame, _, _ = geometry(text)
                if len(frame.types)!=64 or sum(frame.types)!=al:
                    raise ValueError(f"Invalid composition in {key}")
                (fixed_train if split=="train" else fixed_control).append(len(raw))
                raw.append(text)
                origin.append(f"mcsqs:{key}")
                rho=np.vstack([rho,np.loadtxt(path/"corr.out",ndmin=2)])
    comps = np.array([len(re.findall(r"\sAl\s*$",s,re.M))/len(re.findall(r"\s(?:Al|Ti)\s*$",s,re.M)) for s in raw])
    x = rho[:,columns]*MULTIPLICITIES
    if not np.allclose(x[:,1], 2*comps-1,atol=1e-9):
        raise ValueError("Native point correlation species convention differs from S1")
    # Filter by geometric feasibility of the force supercell before choosing
    # training structures. Never use MTP energies or frequencies for selection.
    repeat_by_matrix = {}
    eligible = np.zeros(n_native,dtype=bool)
    for i in np.flatnonzero(comps[:n_native] <= .35):
        frame, matrix, _ = geometry(raw[i])
        signature = tuple(matrix.ravel())
        if signature not in repeat_by_matrix:
            repeat_by_matrix[signature] = repeats(frame.cell,len(frame.types))
        eligible[i] = repeat_by_matrix[signature] is not None
    train,control,unique = choose_design(x,comps,n_native,fixed_train,fixed_control,eligible)
    dest = args.output
    dest.mkdir(parents=True, exist_ok=False)
    (dest/"cells").mkdir()
    (dest/"native").mkdir()
    native_order = sorted(columns)
    for name in ["lat.in","sym.out","binaries.sha256","job.sh"]:
        (dest/name).write_bytes(contents[name])
    (dest/"native"/"clusters.all.out").write_bytes(contents["clusters.out"])
    (dest/"clusters.out").write_bytes(("\n\n".join(blocks[i] for i in native_order)+"\n\n").encode("ascii"))
    native_names = [names[i] for i in native_order]
    error = 0.
    with (dest/"pool.csv").open("w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["native_id", "c_Al", "unique_Ti_rich", *ORBIT_NAMES])
        keep=set(unique)|set(fixed_train+fixed_control)
        for i in range(len(raw)):
            w.writerow([origin[i], comps[i], int(i in keep), *x[i]])
    with (dest/"design.csv").open("w",newline="") as f, (dest/"jobs.csv").open("w",newline="") as g:
        w,j = csv.writer(f),csv.writer(g)
        w.writerow(["id","split","cell","cell_sha256","shape","N","N_Al","D019_preferred",
                    "native_id","str_sha256",*ORBIT_NAMES])
        j.writerow(["id","ratio","cell","repeat_a","repeat_b","repeat_c","mesh","label_subdir"])
        sequence=[]
        for split, ids in [("train",train),("control",control)]:
            for seq,src in enumerate(ids):
                ident=f"{split}_{seq:02d}"
                path=dest/"selected"/ident
                path.mkdir(parents=True)
                text = raw[src]
                (path/"str.out").write_bytes(text.encode())
                sequence.append(text+"end\n")
                with (path/"corr.out").open("w",newline="\n") as stream:
                    np.savetxt(stream,rho[src,native_order][None,:],fmt="%.14g")
                frame, matrix, sites = geometry(text)
                independent = features_for_parent_sites(matrix, sites, 2*frame.types-1)
                error = max(error,float(np.max(abs(independent-x[src]))))
                if error>1e-8:
                    raise ValueError(f"corrdump/Python S1 feature mismatch at {ident}: {error}")
                rel=f"cells/{ident}.cfg"
                write_cfg(dest/rel,[frame])
                n=len(frame.types)
                w.writerow([ident,split,rel,hashlib.sha256((dest/rel).read_bytes()).hexdigest(),
                            json.dumps(matrix.tolist()),n,int(sum(frame.types)),int(src==train[1] or origin[src].endswith("_preferred")),
                            origin[src],hashlib.sha256(text.encode()).hexdigest(),*x[src]])
                if n==64:
                    # S1 specifies 2x2x3 on a 4x4x2 chemical HCP cell in parent axes.
                    if not np.array_equal(matrix, np.diag([4,4,2])):
                        raise ValueError("64-site source matrix is not the S1 repeat orientation")
                    rep = [2,2,3]
                else:
                    rep = repeat_by_matrix[tuple(matrix.ravel())]
                for ratio in VOLUME_RATIOS:
                    j.writerow([ident,ratio,rel,*rep,6 if n==64 else 16,f"{ident}_v{ratio:.2f}"])
    (dest/"native"/"selected.out").write_bytes("".join(sequence).encode("ascii"))
    with (dest/"native"/"S1_round_eV_atom.eci").open("w",newline="\n") as stream:
        np.savetxt(stream, STATIC_S1[[ORBIT_NAMES.index(n) for n in native_names]]/2,fmt="%.15g")
    metadata=dict(source="ATAT genstr and corrdump 3.52", source_archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  genstr_candidates=n_native,n_fixed_mcsqs=len(fixed_train)+len(fixed_control),
                  genstr_Ti_rich=int(sum(comps[:n_native]<=.35)),
                  genstr_with_compatible_force_supercell=int(eligible.sum()),
                  distinct_Ti_rich_features=len(keep),max_Al_fraction=.35,n_train=len(train),n_control=len(control),
                  rank=int(np.linalg.matrix_rank(x[train])),condition_number=float(np.linalg.cond(x[train])),
                  maximum_independent_feature_error=error,native_names=native_names, paper_names=ORBIT_NAMES,
                  native_rho_to_s1="rho times clusters.out multiplicity divided by two; Ti=-1",
                  fit_use="new independent ATAT design; source authors' exact decoration IDs unavailable")
    (dest/"design.json").write_text(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
