"""Compare two tangent solvers on exactly the same saved VCSGC means."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0,str(BASE/"thermo"))
from vcsgc import integrate


def table(p):
    with p.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def compare(package,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((package/"manifest.json").read_text(encoding="utf8"))
    for name,sha in manifest["files_sha256"].items():
        raw = (package/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:
            raise ValueError("Unexpected bytes in "+name)
    potential = hashlib.sha256((BASE/"inputs/fitted.mtp").read_bytes()).hexdigest()
    assert potential==manifest["MTP_sha256"]
    modspec = importlib.util.spec_from_file_location("author_phase_contacts",
                         package/"phase_check/reproduce_C46_phase_contacts.py")
    mod = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(mod)
    groups = defaultdict(list)
    points = table(package/"phase_check/C46_phase_boundary_points.csv")
    ref = {(r["energetic_model"],int(r["T_K"]),int(r["MC_sites"])):r for r in points
           if r["statistics"]=="MC"}
    for r in table(package/"phase_check/MC_window_means.csv"):
        kind = "HA" if r["model"]=="HA_C40" else r["model"]
        key = ("author",kind,int(r["T_K"]),int(r["N"]))
        groups[key].append([float(r["mean_c_Al"]),float(r["mean_delta_mu_eV"])])
    local = BASE/"thermo/results"
    for path in sorted(local.glob("atat*/**/window_means.csv")):
        for r in table(path):
            kind = r["model"].removeprefix("atat_")
            key = ("ours",kind,int(float(r["T_K"])),int(r["N"]))
            groups[key].append([float(r["mean_c"]),float(r["mean_mu_eV"])])
    rows=[]
    for (origin,kind,t,n),values in sorted(groups.items()):
        c,mu = np.array(values).T
        first = mod.tangent(c,mu)
        second = [s for s in integrate(c,mu,.005)[2] if not s["window_edge"]]
        pa,pb = (100*first["c_alpha"],100*first["c_alpha2"]) if first["resolved"] else ("","")
        if origin=="author":
            published = ref[kind,t,n]
            assert (published["status"]=="resolved")==first["resolved"]
            if first["resolved"]:
                assert abs(float(published["c_alpha_Al_at_percent"])-pa)<1e-8
                assert abs(float(published["c_alpha2_Al_at_percent"])-pb)<1e-8
        # Author returns the widest hull gap. Keep every interval our solver finds;
        # choose its widest to compare the numerical implementations only.
        widest = max(second,key=lambda r:r["alpha2_at_pct"]-r["alpha_at_pct"],default=None)
        oa,ob = (widest["alpha_at_pct"],widest["alpha2_at_pct"]) if widest else ("","")
        both = first["resolved"] and widest is not None
        rows.append(dict(source=origin,model=kind,T_K=t,N=n,windows=len(values),
            author_resolved=first["resolved"],our_num_interior_intervals=len(second),
            author_alpha=pa,author_alpha2=pb,our_widest_alpha=oa,our_widest_alpha2=ob,
            delta_alpha=oa-pa if both else "",delta_alpha2=ob-pb if both else "",
            our_all_intervals=" | ".join(f"{v['alpha_at_pct']:.8f},{v['alpha2_at_pct']:.8f}" for v in second)))
    with (out/"mc_same_windows.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summary=dict(verified_manifest_files=len(manifest["files_sha256"]),MTP_sha256=potential)
    for origin in ("author","ours"):
        rr=[r for r in rows if r["source"]==origin]
        summary[origin]=dict(isotherms=len(rr),windows=sum(r["windows"] for r in rr),
             resolved_author=sum(r["author_resolved"] for r in rr),
             resolved_ours=sum(r["our_num_interior_intervals"]>0 for r in rr),
             status_mismatches=sum(r["author_resolved"]!=(r["our_num_interior_intervals"]>0) for r in rr),
             multiple_intervals=sum(r["our_num_interior_intervals"]>1 for r in rr),
             max_widest_difference_at_pct=max(abs(d) for r in rr for d in
                 (r["delta_alpha"],r["delta_alpha2"]) if d!=""))
    (out/"checks.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("package",type=Path)
    p.add_argument("--out",type=Path,default=HERE/"mc")
    a=p.parse_args()
    compare(a.package,a.out)
