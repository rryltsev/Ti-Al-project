"""Cross-check structures, chemical correlations and common raw MTP labels of two Ti-Al designs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE/"atomistic/atat"))
from select_cells import geometry, orbit_name
from hcp_orbits import MULTIPLICITIES, ORBIT_NAMES, VREF, features_for_parent_sites, hcp_geometry

OFFSET = np.array([2,2,3])  # integer units of 1/6 of one primitive lattice coordinate


def csv_in(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def csv_out(path,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        wr = csv.DictWriter(f,fieldnames=rows[0])
        wr.writeheader()
        wr.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_poscar(path):
    lines = path.read_text().splitlines()
    scale = float(lines[1])
    assert scale>0 and lines[7].lower().startswith("d"),path
    cell = scale*np.array([s.split() for s in lines[2:5]],float)
    kinds = np.repeat(lines[5].split(),[int(s) for s in lines[6].split()])
    assert set(kinds)<=set(("Al","Ti"))
    frac = np.array([s.split()[:3] for s in lines[8:8+len(kinds)]],float)
    return cell,frac,np.where(kinds=="Al",1,-1)


def key(x):
    return tuple(np.round(x,8))


def space_group():
    metric = np.array([[2,1],[1,2]])
    ops = []
    for vals in product([-1,0,1],repeat=4):
        a = np.reshape(vals,(2,2))
        if not np.array_equal(a@metric@a.T,metric):
            continue
        for z in (-1,1):
            rot = np.zeros((3,3),int)
            rot[:2,:2] = a
            rot[2,2] = z
            for s in (np.zeros(3,int),OFFSET):
                if set(map(tuple, np.array([s,OFFSET@rot+s])%6))=={(0,0,0),tuple(OFFSET)}:
                    ops.append((rot,s))
    assert len(ops)==24
    return ops


class Crystal:
    def __init__(self,matrix,sites,spins):
        self.matrix = matrix
        self.sites = sites
        self.spins = spins
        self.n = len(spins)
        self.det = abs(round(np.linalg.det(matrix)))
        self.adj = np.rint(np.linalg.inv(matrix)*self.det).astype(int)
        self.coords = 6*sites[:,:3] + sites[:,3:4]*OFFSET
        self.occ = {self.address(v):s for v,s in zip(self.coords,spins)}
        assert self.n==2*self.det==len(self.occ)

    def address(self,coord):
        b = int((coord[2]%6)//3)
        xyz = coord-b*OFFSET
        if np.any(xyz%6):
            raise ValueError("Invalid HCP site")
        return (*((xyz//6 @ self.adj)%self.det),b)

    def equivalent_to(self,tgt,ops):
        # Prove equality on the infinite lattice for each origin and point operation.
        # Every rotated source period must be a period of the target decoration;
        # one source supercell then fixes the field everywhere.
        for op,(rot,origin) in enumerate(ops):
            periods = 6*self.matrix@rot
            if not all(tgt.occ[tgt.address(v+d)]==s for d in periods
                       for v,s in zip(tgt.coords,tgt.spins)):
                continue
            source = self.coords@rot + origin
            for extra in tgt.coords[tgt.sites[:,3]==0]:
                if all(tgt.occ[tgt.address(v+extra)]==s for v,s in zip(source,self.spins)):
                    return op,extra.tolist()
        return None


def raw_from_atat(raw):
    frame,matrix,sites = geometry(raw)
    return Crystal(matrix,sites,2*frame.types-1)


def group_columns(src):
    pub = json.loads((src/"models/geometry.json").read_text(encoding="utf8"))["orbits"]
    blocks = re.split(r"\n\s*\n",(src/"models/static/clusters.out").read_text().strip())
    names = [orbit_name(b) for b in blocks]
    local_blocks = re.split(r"\n\s*\n",(BASE/"atomistic/atat/output/clusters.out").read_text().strip())
    local_names = [orbit_name(b) for b in local_blocks]
    assert set(names)==set(local_names)==set(ORBIT_NAMES) and len(names)==len(pub)==20
    index = [names.index(name) for name in ORBIT_NAMES]
    columns = []
    for i,name in enumerate(ORBIT_NAMES):
        k = index[i]
        pm = int(blocks[k].splitlines()[0])
        lm = int(local_blocks[local_names.index(name)].splitlines()[0])
        assert pm==lm==pub[k]["native_multiplicity_per_primitive_cell"]==2*MULTIPLICITIES[i]
        assert pub[k]["index"]==k
        columns.append(dict(working_name=name,author_name=pub[k]["name"],
                            author_column_index=k,working_S1_index=i,
                            working_native_file_index=local_names.index(name),
                            multiplicity_per_atom=MULTIPLICITIES[i]))
    return index,columns


def run(src,out):
    out.mkdir(parents=True,exist_ok=True)
    source_rows = csv_in(src/"structures/index.csv")
    decos = [s.strip() for s in re.split(r"(?m)^end\s*$",(src/"structures/structures.out").read_text()) if s.strip()]
    static = np.load(src/"models/static/volume_temperature_coefficients.npz")
    thermal = np.load(src/"models/HA_QHA/volume_temperature_coefficients.npz")
    idx,columns = group_columns(src)
    csv_out(out/"orbit_mapping.csv",columns)
    reference_x = static["X"][:,idx]
    assert np.array_equal(reference_x,thermal["X"][:,idx]) and len(decos)==len(source_rows)==len(reference_x)==46
    own = BASE/"atomistic/atat/output"
    own_rows = csv_in(own/"design.csv")
    own_by_x = defaultdict(list)
    own_crystals = {}
    for row in own_rows:
        ident = row["id"]
        fn = own/"selected"/ident/"str.out"
        assert digest(fn)==row["str_sha256"]
        crystal = raw_from_atat(fn.read_text())
        vec = features_for_parent_sites(crystal.matrix,crystal.sites,crystal.spins)
        assert max(abs(vec-np.array([float(row[n]) for n in ORBIT_NAMES])))<2e-10
        own_crystals[ident] = crystal
        own_by_x[key(vec)].append(row)
    pool = defaultdict(list)
    for row in csv_in(own/"pool.csv"):
        pool[key([float(row[n]) for n in ORBIT_NAMES])].append(row["native_id"])
    with tarfile.open(BASE/"cluster/runs/atat-enum-12-01/outputs/enum.tar.gz") as arc:
        enum_stream = arc.extractfile("enum.out").read().decode()
    enum = [raw for raw in re.split(r"(?m)^end\s*$",enum_stream) if raw.strip()]
    assert len(enum)==5777
    def pool_crystal(native_id):
        method,number = native_id.split(":")
        if method=="genstr":
            return raw_from_atat(enum[int(number)-1])
        assert method=="mcsqs"
        return raw_from_atat((BASE/"atomistic/atat_large/cells64"/number/"str.out").read_text())
    vol = static["volumes_A3_atom"]
    it0 = np.argmin(abs(vol-VREF))
    assert abs(vol[it0]-VREF)<2e-9 and np.array_equal(vol,thermal["volumes_A3_atom"])
    temps = thermal["temperatures_K"]
    label_rows = csv_in(BASE/"atat_comparison/MTP_labels_414.csv")
    labels = {(r["id"],round(float(r["V_A3_per_atom"]),6)):r for r in label_rows}
    own_temps = [int(re.fullmatch(r"Fvib_(\d+)K_eV_atom",k)[1]) for k in label_rows[0] if k.startswith("Fvib_")]
    ops = space_group()
    site_comparison, shared, label_comparison = [],[],[]
    parent0 = parse_poscar(src/source_rows[0]["parent"])[0]
    mat0 = np.array([l.split() for l in decos[0].splitlines()[3:6]],float)
    primitive = np.linalg.solve(mat0,parent0)
    own_primitive = hcp_geometry((1,1,1))[0]
    for i,(row,raw) in enumerate(zip(source_rows,decos)):
        assert int(row["index"])==i
        fit = row["role"]=="fit"
        assert static["training"][i]==fit
        parent, relaxed = src/row["parent"], src/row["relaxed_Vref"]
        assert digest(parent)==row["parent_sha256"] and digest(relaxed)==row["relaxed_sha256"]
        c = raw_from_atat(raw)
        pa,pf,ps = parse_poscar(parent)
        ra,rf,rs = parse_poscar(relaxed)
        assert c.n==int(row["atoms"])==len(ps) and np.count_nonzero(ps==1)==int(row["Al_atoms"]) and sorted(ps)==sorted(rs)
        matrix = np.rint(pa@np.linalg.inv(primitive)).astype(int)
        frac = pf@pa@np.linalg.inv(primitive)
        b = np.rint(2*frac[:,2]).astype(int)%2
        int_pos = frac-b[:,None]*(OFFSET/6)
        site_error = np.max(abs(np.rint(int_pos)-int_pos))
        assert site_error<2e-7 and np.max(abs(pa-matrix@primitive))<2e-7
        poscar = Crystal(matrix,np.column_stack((np.rint(int_pos).astype(int),b)),ps)
        assert np.array_equal(poscar.matrix,c.matrix) or c.equivalent_to(poscar,ops) is not None
        assert all(poscar.occ[poscar.address(v)]==s for v,s in zip(c.coords,c.spins))
        x = features_for_parent_sites(c.matrix,c.sites,c.spins)
        xerr = max(abs(x-reference_x[i]))
        assert xerr<1e-8
        hits,certified = own_by_x[key(x)],[]
        for candidate in hits:
            ident = candidate["id"]
            proof = c.equivalent_to(own_crystals[ident],ops)
            if proof is None:
                continue
            certified.append(ident)
            shared.append(dict(author_index=i,author=row["configuration"],author_split=row["role"],
                               ours=ident,our_split=candidate["split"],author_atoms=c.n,
                               our_atoms=own_crystals[ident].n,rotation_index=proof[0],shift_sixths=proof[1]))
            for k,v in enumerate(vol):
                lr = labels[ident,round(v,6)]
                sa = bool(static["stable"][k,i])
                so = lr["stable_mesh"].lower()=="true"
                ea = static["E_static_eV_atom"][k,i]
                eo = float(lr["E_MTP_eV_atom"])
                for t in own_temps:
                    q, = np.flatnonzero(temps==t)
                    vib_a = thermal["Y_eV_atom"][k,i,q]-ea
                    both = sa and so
                    vib_o = float(lr[f"Fvib_{t}K_eV_atom"]) if so else None
                    if both:
                        assert np.isfinite(vib_a) and np.isfinite(vib_o)
                    label_comparison.append(dict(author_index=i,ours=ident,V_over_Vref=v/VREF,T_K=t,
                        author_stable=sa,our_stable=so,author_static_eV_atom=ea,our_static_eV_atom=eo,
                        static_ours_minus_author_meV_atom=1000*(eo-ea) if np.isfinite(ea) else "",
                        Fvib_ours_minus_author_meV_atom=1000*(vib_o-vib_a) if both else ""))
        delta_static_index = static["E_static_eV_atom"][it0,i]-float(row["static_target_eV_atom"])
        assert abs(delta_static_index)<1e-12
        ids_in_pool = pool[key(x)]
        verified_pool = [candidate for candidate in ids_in_pool if c.equivalent_to(pool_crystal(candidate),ops) is not None]
        # Reused structures can be delivered in a separately oriented cell.
        cell_change = max(abs(ra-pa).flat)
        site_comparison.append(dict(author_index=i,name=row["configuration"],role=row["role"],atoms=c.n,
            Al_atoms=int(row["Al_atoms"]),parent_A3_atom=abs(np.linalg.det(pa))/c.n,
            relaxed_A3_atom=abs(np.linalg.det(ra))/c.n,reused=int(row["equivalent_cell_reused"]),
            parent_vs_relaxed_matrix_max_abs_A=cell_change,parent_site_map_max_abs_primitive=site_error,
            geometry_vs_author_X_max_abs=xerr,raw_static_vs_index_eV_atom=delta_static_index,
            same_working_features=" ".join(z["id"] for z in hits),verified_decorations=" ".join(certified),
            same_features_in_enumerated_pool=" ".join(ids_in_pool),
            verified_decorations_in_enumerated_pool=" ".join(verified_pool),parent_sha256=digest(parent),
            relaxed_sha256=digest(relaxed)))

    csv_out(out/"cells.csv",site_comparison)
    csv_out(out/"shared_decorations.csv",shared)
    csv_out(out/"shared_direct_labels.csv",label_comparison)
    def max_error(name):
        return max(abs(float(r[name])) for r in site_comparison)
    summary = dict(source_structures=len(source_rows),author_training=int(static["training"].sum()),
        own_training=sum(r["split"]=="train" for r in own_rows),
        author_features_rank=int(np.linalg.matrix_rank(reference_x[static["training"]])),
        author_training_condition=float(np.linalg.cond(reference_x[static["training"]])),
        unique_author_feature_vectors=len(set(map(key,reference_x))),
        source_cells_with_same_20_features_as_working=sum(bool(r["same_working_features"]) for r in site_comparison),
        source_cells_with_same_20_features_in_enumerated_pool=sum(bool(r["same_features_in_enumerated_pool"]) for r in site_comparison),
        source_cells_with_verified_decorations_in_enumerated_pool=sum(bool(r["verified_decorations_in_enumerated_pool"]) for r in site_comparison),
        exact_shared_periodic_decorations=len(shared),
        shared_split_pairs=dict(Counter(f'{r["author_split"]}/{r["our_split"]}' for r in shared)),
        source_training_counts_by_volume=static["stable"][:,static["training"]].sum(1).tolist(),
        source_stable_all_points=int(static["stable"].sum()),
        independently_checked_X_entries=int(reference_x.size),
        largest_feature_error=max_error("geometry_vs_author_X_max_abs"),
        max_parent_site_map_error_primitive=max_error("parent_site_map_max_abs_primitive"),
        source_parent_max_abs_Vref_error=max(abs(r["parent_A3_atom"]-VREF) for r in site_comparison),
        relaxed_max_abs_Vref_error=max(abs(r["relaxed_A3_atom"]-VREF) for r in site_comparison),
        author_a_c_A=[primitive[0,0],primitive[2,2]],working_a_c_A=[own_primitive[0,0],own_primitive[2,2]],
        author_N_hist=dict(sorted(Counter(r["atoms"] for r in site_comparison).items())),
        working_N_hist=dict(sorted(Counter(int(r["N"]) for r in own_rows).items())))
    for label,sub in [("all",label_comparison),("vref1000",[r for r in label_comparison if abs(r["V_over_Vref"]-1)<1e-8 and r["T_K"]==1000])]:
        for col in ("static_ours_minus_author_meV_atom","Fvib_ours_minus_author_meV_atom"):
            vals = np.array([r[col] for r in sub if r[col]!=""])
            summary[f"{label}_{col}"] = dict(rmse=float(np.mean(vals**2)**.5),max_abs=float(max(abs(vals))))
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("author_archive",type=Path)
    arg.add_argument("--out",type=Path,default=HERE)
    args = arg.parse_args()
    run(args.author_archive,args.out)
