"""Compare S1 and Fig. 2 with new CPU MTP ECI on the same per-cluster spin basis."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"atomistic"))
from hcp_orbits import ORBIT_NAMES, MULTIPLICITIES, build_orbits, features, sublattices
from fit_eci import Labels


def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write(p, data):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def rounded_s1(si):
    w="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with ZipFile(si) as z:
        table=ET.fromstring(z.read("word/document.xml")).find(".//"+w+"tbl")
    return ["".join(t.text or "" for t in r.findall(w+"tc")[2].iter(w+"t"))
            for r in table.findall(w+"tr")[1:]]


def defect_probes(s1, red, masked, vib):
    """Canonical D019 antisite pairs and two solutes in HCP Ti; energy per defect pair."""
    shape=(8,8,6)
    orbits=build_orbits(shape)
    ordered=np.where(sublattices(shape)==0,1,-1)
    n=len(ordered)
    assert ordered[0]==1
    neighbors=[]
    for p in orbits[2:9]:
        ids=set(p[np.any(p==0,axis=1)].ravel())-{0}
        neighbors.append(sorted(ids))
    far=next(i for i in range(n) if ordered[i]==-1 and all(i not in shell for shell in neighbors))
    # Each transfer is at fixed Al count. The far pair has no cluster containing both changes.
    probe=[]
    for mode in ["D019_antisite_swap", "HCP_Ti_AlAl_minus_far"]:
        if mode=="D019_antisite_swap":
            ref=ordered
        else:
            ref=np.full(n,-1)
            ref[[0,far]]=1
        x0=features(ref,orbits)
        for shell,sites in [("p1",neighbors[0]),("p2",neighbors[1]),("beyond_p7",[far])]:
            variations=[]
            for site in sites:
                if mode=="D019_antisite_swap":
                    if ordered[site]!=-1:
                        continue
                    changed=ref.copy()
                    changed[[0,site]]*=-1
                else:
                    changed=np.full(n,-1)
                    changed[[0,site]]=1
                dy=n*(features(changed,orbits)-x0)
                assert np.allclose(dy[:2],0.,atol=1e-12)
                variations.append(dy)
            if not np.allclose(variations,variations[0],atol=1e-9):
                raise ValueError(f"Multiple symmetry environments for {mode}/{shell}")
            dy=variations[0][2:]
            row=dict(contrast=mode,separation=shell,source_site=0,second_site=sites[0],n_equivalent_checked=len(variations))
            for label,coef in [("paper_static",s1),("new_static",masked),("paper_vib",red),("new_vib",vib),
                               ("paper_total",s1+red),("new_total",masked+vib)]:
                row[label+"_eV_defect_pair"]=float(dy@coef[2:])
            probe.append(row)
    return probe


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pd",type=Path,help="folder with the two Ti-Al v16 Word documents")
    args=cli.parse_args()
    branch=ROOT/"atomistic/atat"
    source=json.loads((ROOT/"thermo/results/coefficients_HA1000.json").read_text(encoding="utf8"))
    for name,sha in source["sources_sha256"].items():
        if name.endswith(".docx"):
            assert hashlib.sha256((args.pd/name).read_bytes()).hexdigest()==sha
    s1_text=rounded_s1(next(args.pd.glob("*v16*доп*.docx")))
    s1=np.array(list(map(float,s1_text)))
    assert np.array_equal(s1,np.asarray(source["static_S1_eV"])) and tuple(source["orbit_names"])==ORBIT_NAMES
    # The PDF/figure has no red columns for the empty or point orbit.
    red=np.array(source["fig2_vib_pixel_eV"])
    red[:2]=np.nan
    paper_half_ulp=np.array([.5*10**Decimal(value).as_tuple().exponent for value in s1_text])
    full=np.load(branch/"static_fit/static_grid.npz")["J_static"][0]
    thermal=np.load(branch/"ha_fit/grid.npz")
    masked=thermal["J_static"][0]
    exact_T=int(np.flatnonzero(thermal["temperature_K"]==1000).item())
    vib=thermal["J_vib"][0,exact_T]
    assert thermal["temperature_K"][exact_T]==1000
    names=ORBIT_NAMES
    paper_total=s1+red
    new_total=masked+vib
    pixel=np.array([np.nan,np.nan]+[source["eV_per_pixel"]["pairs"]]*7+[source["eV_per_pixel"]["triplets"]]*11)
    blue=np.array(source["fig2_static_pixel_eV"])
    blue[:2]=np.nan

    matched=rows(branch/"output/design.csv")
    x=np.array([[float(r[n]) for n in names] for r in matched])
    labels=Labels(ROOT/"cluster/runs/atat-static-02/outputs/labels.tar.gz")
    y=np.array([json.loads(labels.read(r["id"]+"_v1.00","label.json"))["static_eV_atom"] for r in matched])
    train=np.array([r["split"]=="train" for r in matched])
    ids=np.array([r["id"] for r in matched])
    excluded=json.loads((branch/"ha_fit/validation.json").read_text())["volumes"][0]["excluded_unstable"]
    def refit(exclude):
        mask=train & ~np.isin(ids,exclude)
        c,_,rank,_=np.linalg.lstsq(x[mask],y[mask],rcond=None)
        assert rank==20
        return c
    assert np.allclose(refit([]),full,rtol=0,atol=5e-13)
    assert np.allclose(refit(excluded),masked,rtol=0,atol=5e-9)
    individual=[refit([ident])-full for ident in excluded]

    ideal=features(np.where(sublattices((2,2,1))==0,1,-1),build_orbits((2,2,1)))
    # Independent occupations at c=1/4 in the thermodynamic-limit Bernoulli alloy.
    random=MULTIPLICITIES * (-.5)**np.array([0,1]+[2]*7+[3]*11)
    variants={"random_ideal25_minus_D019":random-ideal}
    for count in (17,20,22):
        a=[i for i,r in enumerate(matched) if r["native_id"].endswith(f"{count}_random")]
        b=[i for i,r in enumerate(matched) if r["native_id"].endswith(f"{count}_preferred")]
        assert len(a)==len(b)==1
        variants[f"random_minus_preferred_{count}Al64"]=x[a[0]]-x[b[0]]
    comps=[]
    order=[]
    mill=lambda x: None if not np.isfinite(x) else float(x*1000)
    for i,n in enumerate(names):
        comps.append(dict(orbit=n,multiplicity_atom=MULTIPLICITIES[i],paper_S1_printed_eV=s1_text[i],
                          paper_S1_half_last_digit_meV=mill(paper_half_ulp[i]),
                          paper_Fig2_static_digitized_meV=mill(blue[i]),paper_Fig2_one_pixel_meV=mill(pixel[i]),
                          paper_static_meV=mill(s1[i]),new_static39_meV=mill(full[i]),
                          new_static37_meV=mill(masked[i]),delta_static39_minus_paper_meV=mill(full[i]-s1[i]),
                          delta_static37_minus_paper_meV=mill(masked[i]-s1[i]),delta_mask37_minus39_meV=mill(masked[i]-full[i]),
                          **{f"static_shift_omit_{j}_only_meV":mill(c[i]) for j,c in zip(excluded,individual)},
                          paper_vib_digitized_meV=mill(red[i]),new_vib_meV=mill(vib[i]),delta_vib_meV=mill(vib[i]-red[i]),
                          paper_total_S1_plus_Fig2_meV=mill(paper_total[i]),new_total_meV=mill(new_total[i]),
                          delta_total_meV=mill(new_total[i]-paper_total[i])))
        for descr,df in variants.items():
            assert np.allclose(df[:2],0,atol=1e-14)
            order.append(dict(contrast=descr,orbit=n,delta_feature=df[i],
                         paper_static_meV_atom=mill(s1[i]*df[i]),new_static39_meV_atom=mill(full[i]*df[i]),
                         new_static37_meV_atom=mill(masked[i]*df[i]),
                         paper_vib_meV_atom=0. if i<2 else mill(red[i]*df[i]),
                         new_vib_meV_atom=mill(vib[i]*df[i]),paper_total_meV_atom=0. if i<2 else mill(paper_total[i]*df[i]),
                         new_total_meV_atom=mill(new_total[i]*df[i])))
    write(HERE/"coefficients_1000K_vref.csv",comps)
    write(HERE/"order_contributions.csv",order)
    write(HERE/"defect_probes.csv",defect_probes(s1,red,masked,vib))
    rms=lambda val: float(np.sqrt(np.mean(val**2))*1000.)
    rmse={name:rms((x@c-y)[~train]) for name,c in [("article_S1_rounded",s1),("new39",full),("new37",masked)]}
    bands={"p1_p7":slice(2,9),"triplets":slice(9,20),"interacting":slice(2,20)}
    summary=dict(unit_ECI="meV/cluster",unit_order="meV/atom",temperature_K=1000,
                 removed=excluded,article_vib_absent_orbits=list(names[:2]),new_static_holdout_RMSE_meV_atom=rmse,
                 static_holdout_mask_prediction_change_rms_meV_atom=rms((x@(masked-full))[~train]),metrics={})
    for label,sl in bands.items():
        s=s1[sl];v=red[sl];total=paper_total[sl]
        summary["metrics"][label]=dict(static39_diff_rms=rms(full[sl]-s),static37_diff_rms=rms(masked[sl]-s),
                      mask_shift_rms=rms(masked[sl]-full[sl]),vib_diff_rms=rms(vib[sl]-v),total_diff_rms=rms(new_total[sl]-total),
                      static_same_sign=int(np.sum(s*masked[sl]>0)),vib_same_sign=int(np.sum(v*vib[sl]>0)),
                      total_same_sign=int(np.sum(total*new_total[sl]>0)),n=len(s))
    summary["ordering"]={}
    direct_rows=rows(branch/"ha_fit/ordered_random_eV_atom.csv")
    for label,vec in variants.items():
        group={}
        for prefix,terms in [("paper_S1",s1),("new39",full),("new37",masked),("paper_vib_Fig2",red),
                             ("new_vib",vib),("paper_total",paper_total),("new_total",new_total)]:
            products=vec[2:]*terms[2:]*1000
            group[prefix]=dict(pairs=products[:7].sum(),triplets=products[7:].sum(),sum=products.sum())
        if label.startswith("random_minus_preferred_"):
            number=label.split("_")[-1].split("Al")[0]
            group["direct_MTP"]={r["observable"]:1000*float(r["MTP_random_minus_preferred"])
                     for r in direct_rows if r["N_Al_per_64"]==number and r["observable"] in ("static","Fvib_1000K","E_plus_Fvib_1000K")}
        summary["ordering"][label]=group
    (HERE/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
