"""Assess changes in fitted 1000 K MFA contacts when replacing C46 training data."""
import argparse
from pathlib import Path
import sys

import numpy as np

from compare import HERE,PROJECT,ORBIT_NAMES,VREF,load_npz,rows,save_csv
sys.path.insert(0,str(PROJECT/"eci_boundary"))
from fit_sensitivity import alpha2,audit_tangent,contact,derivative,sublattice_weights


def run(src,out):
    s=load_npz(src/"models/static/volume_temperature_coefficients.npz")
    h=load_npz(src/"models/HA_QHA/volume_temperature_coefficients.npz")
    vi,=np.flatnonzero(np.isclose(s["volumes_A3_atom"],VREF,rtol=0,atol=1e-8))
    ti,=np.flatnonzero(s["temperatures_K"]==1000)
    cells=rows(src/"structures/index.csv")
    own=rows(PROJECT/"atomistic/atat/output/design.csv")
    lab={r["id"]:r for r in rows(PROJECT/"atat_comparison/MTP_labels_414.csv") if abs(float(r["V_over_Vref"])-1)<1e-8}
    y_a=np.column_stack([s["E_static_eV_atom"][vi],h["Y_eV_atom"][vi,:,ti]])
    y_o=np.array([[float(lab[r["id"]]["E_MTP_eV_atom"]),
                   float(lab[r["id"]]["E_MTP_eV_atom"])+float(lab[r["id"]]["Fvib_1000K_eV_atom"] or "nan")] for r in own])
    x_a,x_o=s["X"],np.array([[float(r[k]) for k in ORBIT_NAMES] for r in own])
    m_a=s["training"]&s["stable"][vi]
    m_o=np.array([r["split"]=="train" and lab[r["id"]]["stable_mesh"].lower()=="true" for r in own])
    ids_a=[r["configuration"] for r in cells];ids_o=[r["id"] for r in own]
    same=np.max(abs(x_a[:,None]-x_o[None,:]),axis=2)<1e-8
    if np.any(same.sum(axis=0)>1) or np.any(same.sum(axis=1)>1):
        raise ValueError("Nonunique matched correlation rows")
    w=sublattice_weights()
    points=[]
    for x,y,mask,matches,alt,tag in [(x_a,y_a,m_a,same,y_o,"author"),(x_o,y_o,m_o,same.T,y_a,"ours")]:
        ys=y.copy()
        for i,j in np.argwhere(matches):
            if not np.isfinite(alt[j]).all():
                raise ValueError("Common structure has incomplete alternative labels")
            ys[i]=alt[j]
        js=np.linalg.lstsq(x[mask],y[mask],rcond=None)[0]
        adjusted=np.linalg.lstsq(x[mask],ys[mask],rcond=None)[0]
        for k,method in enumerate(["static","HA"]):
            def solve(j):
                j=j.copy();j[:2]=0.
                result=contact(w,j,[.085,.83,.002]);audit_tangent(w,j,result)
                return np.array([100*result[0][0],alpha2(result[0])])
            ab=solve(js[:,k]);ba=solve(adjusted[:,k])
            points.append(dict(dataset=tag,model=method,fit=int(mask.sum()),replaced_training_labels=int((mask & matches.any(axis=1)).sum()),
                alpha_before=ab[0],alpha_after=ba[0],alpha2_before=ab[1],alpha2_after=ba[1],
                shift_alpha2_at_pct=ba[1]-ab[1],max_abs_shift_J_nonaffine_meV=1000*np.max(abs(js[2:,k]-adjusted[2:,k]))))
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/"replace_shared_labels.csv",points)
    # One row per unique training correlation. The contribution of a shared
    # row is the sum of its two opposite-signed score vectors at fixed X.
    groups={}
    use_x=np.vstack([x_a[m_a],x_o[m_o]])
    use_y=np.vstack([y_a[m_a],y_o[m_o]])
    a_count=int(m_a.sum())
    sign=np.r_[-np.ones(a_count),np.ones(m_o.sum())]
    ord_a=np.flatnonzero(m_a);ord_o=np.flatnonzero(m_o)
    for k,i in enumerate(ord_a):
        match,=np.where(same[i])
        group=(ids_a[i],ids_o[match[0]] if match.size else "")
        groups.setdefault(group,[]).append(k)
    for k,i in enumerate(ord_o,a_count):
        match,=np.where(same[:,i])
        group=(ids_a[match[0]] if match.size else "",ids_o[i])
        groups.setdefault(group,[]).append(k)
    nodes,qw=np.polynomial.legendre.leggauss(64)
    accum=np.zeros((len(use_x),2,3))
    endpoints=[]
    for side,mask in [("author",sign<0),("ours",sign>0)]:
        j=np.linalg.lstsq(use_x[mask],use_y[mask],rcond=None)[0]
        ends=[]
        for ind in range(2):
            jp=j[:,ind].copy();jp[:2]=0.
            ends.append([alpha2(contact(w,jp,[.085,.83,.002])[0]),
                         1000*j[ORBIT_NAMES.index("t334"),ind],1000*j[ORBIT_NAMES.index("t123"),ind]])
        endpoints.append(np.array(ends))
    for node,wt in zip(nodes,qw):
        lmb=.5*(node+1)
        weights=np.where(sign>0,lmb,1-lmb)
        ax=np.sqrt(weights)[:,None]*use_x
        ay=np.sqrt(weights)[:,None]*use_y
        jj=np.linalg.lstsq(ax,ay,rcond=None)[0]
        cov=np.linalg.solve(ax.T@ax,use_x.T)
        for ind in range(2):
            jp=jj[:,ind].copy();jp[:2]=0.
            p,jac,*_=contact(w,jp,[.085,.83,.002])
            grad=derivative(w,p,jac)
            dj=cov*(sign*(use_y[:,ind]-use_x@jj[:,ind]))
            da=grad@dj
            accum[:,ind]+=wt*.5*np.stack([da,1000*dj[ORBIT_NAMES.index("t334")],1000*dj[ORBIT_NAMES.index("t123")]],axis=1)
    error=np.max(abs(accum.sum(axis=0)-(endpoints[1]-endpoints[0])))
    if error>2e-6:raise ValueError(f"Training pathway closure failure: {error}")
    contributions=[]
    for (aid,oid),sel in groups.items():
        switch=("shared_training" if len(sel)==2 else "leave_author_training" if sel[0]<a_count else "enter_our_training")
        sums=accum[sel].sum(axis=0)
        for j,model in enumerate(["static","HA"]):
            contributions.append(dict(author_structure=aid,our_structure=oid,training_change=switch,model=model,
                alpha2_shift_at_pct=sums[j,0],t334_J_shift_meV=sums[j,1],t123_J_shift_meV=sums[j,2]))
    save_csv(out/"training_change_path.csv",contributions)
    print("Shared-label replacements:",points)
    print("Max closure error:",error)
    for model in ("static","HA"):
        pool=[r for r in contributions if r["model"]==model]
        print(model,"largest training-row terms:")
        for r in sorted(pool,key=lambda r:abs(r["alpha2_shift_at_pct"]),reverse=True)[:7]: print(r)
        print("By change:",{s:sum(r["alpha2_shift_at_pct"] for r in pool if r["training_change"]==s) for s in ("shared_training","leave_author_training","enter_our_training")})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("author_directory",type=Path)
    ap.add_argument("--out",type=Path,default=HERE)
    ar=ap.parse_args()
    run(ar.author_directory,ar.out)
