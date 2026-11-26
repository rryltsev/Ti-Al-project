"""Attribute 1000 K Ti–Al MFA tangent shifts to S1-order CE interactions."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"thermo"))
from mfa import Isotherm, KB, ORBIT_NAMES, lower_hull_edges, sublattice_weights
from hcp_orbits import MULTIPLICITIES, VREF

KBT=1000*KB
NAMES=ORBIT_NAMES[2:]
DEGREES=np.array([2]*7+[3]*11)


def derivative(poly,du,dv):
    result=np.zeros_like(poly)
    result[:,:-1,:]+=du*poly[:,1:,:]*np.arange(1,poly.shape[1])[None,:,None]
    result[:,:,:-1]+=dv*poly[:,:,1:]*np.arange(1,poly.shape[2])[None,None,:]
    return result


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


class Tangents:
    def __init__(self):
        weights=sublattice_weights()
        self.weights=weights
        phi=weights[2:]
        dc=derivative(phi,2,2)
        dm=derivative(phi,1.5,-.5)
        self.basis=np.stack([phi,dc,dm,derivative(dc,2,2),derivative(dc,1.5,-.5),derivative(dm,1.5,-.5)])

    def state(self,j,c,m):
        q,r=c+.75*m,c-.25*m
        if not (0<q<1 and 0<r<1):
            raise ValueError("Invalid sublattice filling")
        u,v=2*q-1,2*r-1
        polys=np.einsum("kaij,i,j->ka",self.basis,u**np.arange(4),v**np.arange(4))
        h=lambda p:p*np.log(p)+(1-p)*np.log1p(-p)
        lq,lr=np.log(q/(1-q)),np.log(r/(1-r))
        sq,sr=1/(q*(1-q)),1/(r*(1-r))
        ent=KBT/4*np.array([h(q)+3*h(r),lq+3*lr,.75*(lq-lr),sq+3*sr,
                            .75*(sq-sr),.5625*sq+.1875*sr])
        f=polys@j+ent
        return f,polys

    def solve(self,j,initial,check=True):
        # Variables (c_alpha,q_alpha2,r_alpha2) keep both probabilities bounded.
        def equations(x):
            a,q,r=x
            b,m=(q+3*r)/4,q-r
            fa,_=self.state(j,a,0); fb,_=self.state(j,b,m)
            return np.array([fa[1]-fb[1], (fa[1]*(b-a)-(fb[0]-fa[0]))/.12, fb[2]])
        refine=lambda x:least_squares(equations,x,bounds=(1e-11,1-1e-11),ftol=1e-14,xtol=1e-14,gtol=1e-14)
        def distinct(sol):
            a,q,r=sol.x
            b=(q+3*r)/4
            return sol.success and np.max(abs(sol.fun))<1e-9 and a+.005<b<.35 and q-r>.05
        sol=refine(initial)
        grid=np.linspace(0,.35,701)
        feasible=Isotherm(self.weights,np.r_[0.,0.,j],1000)
        energy=None
        if not distinct(sol):
            energy,q_grid=feasible.equilibrium(grid)
            gaps=[(l,r) for l,r in lower_hull_edges(grid,energy) if grid[r]-grid[l]>.005]
            if len(gaps)!=1 or gaps[0][0]==0 or gaps[0][1]==len(grid)-1:
                raise ValueError(f"No unique interior coexistence on the global hull: {gaps}")
            left,right=gaps[0]
            q=q_grid[right]
            sol=refine(np.clip([grid[left],q,(4*grid[right]-q)/3],1e-9,1-1e-9))
        a,q,r=sol.x; b,m=(q+3*r)/4,q-r
        if (not sol.success or np.max(np.abs(equations(sol.x)))>1e-9 or b-a<.005 or m<.05 or b>.35):
            raise ArithmeticError(f"Lost distinct common tangent: {sol.x}, residuals {sol.fun}")
        fa,pa=self.state(j,a,0); fb,pb=self.state(j,b,m)
        states=[(a,0,fa,pa),(b,m,fb,pb)]
        width=b-a
        # d(mu_coex)/dJ from differentiating the equal grand potentials.
        dmu=(pb[0]-pa[0])/width
        random=lambda c: MULTIPLICITIES[2:]*(2*c-1)**DEGREES
        random_prime=lambda c: 2*MULTIPLICITIES[2:]*DEGREES*(2*c-1)**(DEGREES-1)
        random_slope=(random(b)-random(a))/width
        deriv=[]; background=[]; curv=[]
        for c,order,f,p in states:
            k=f[3]-f[4]**2/f[5]
            if k<=0 or f[5]<=0:
                raise ValueError("Unstable chemical or sublattice curvature at a contact")
            curv.append(k)
            dm_dc=-f[4]/f[5]
            deriv.append((dmu-p[1]-p[2]*dm_dc)/k)
            background.append((random_slope-random_prime(c))/k)
        if check:
            if energy is None:
                energy,_=feasible.equilibrium(grid)
            margin=np.min(energy-(fa[0]+fa[1]*(grid-a)))
            endpoint_energy,_=feasible.equilibrium(np.array([a,b]))
            if margin< -2e-8 or max(abs(endpoint_energy-[fa[0],fb[0]]))>2e-8:
                raise ValueError(f"Failed global supporting-line check: margin {margin:.3g}")
        return {"x":sol.x,"c":np.array([a,b]),"q":q,"r":r,"m":m,"derivatives":np.array(deriv),
                "background":np.array(background),"curvature":curv,
                "order_phi":pb[0]-random(b),"tangent_mu_gauge":float(fa[1])}


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    sources={"paper":ROOT/"thermo/results/coefficients_HA1000.json",
             "native":ROOT/"atomistic/atat/ha_fit/grid.npz",
             "native_contacts":ROOT/"thermo/results/atat_vref_1000/grid_mfa_contacts.csv",
             "paper_contacts":ROOT/"thermo/results/mfa_contacts.csv"}
    archive=json.loads(sources["paper"].read_text(encoding="utf8"))
    if tuple(archive["orbit_names"])!=ORBIT_NAMES:
        raise ValueError("Invalid published orbit order")
    static0=np.array(archive["static_S1_eV"])[2:]
    vib0=np.array(archive["fig2_vib_pixel_eV"])[2:]
    with np.load(sources["native"]) as native:
        if tuple(native["orbit_names"])!=ORBIT_NAMES:
            raise ValueError("Invalid native orbit order")
        vi,=np.flatnonzero(np.isclose(native["volume_A3_atom"],VREF))
        ti,=np.flatnonzero(native["temperature_K"]==1000)
        static1=native["J_static"][vi,2:]
        vib1=native["J_vib"][vi,ti,2:]
    model=Tangents()
    states={}
    jr={"static_paper":static0,"static_native":static1,
        "HA_00":static0+vib0,"HA_10":static1+vib0,"HA_01":static0+vib1,"HA_11":static1+vib1}
    for key,j in jr.items():
        states[key]=model.solve(j,[.08,.76,.025])
    rows=[]
    for key,s in states.items():
        rows.append(dict(state=key,alpha_at_pct=100*s["c"][0],alpha2_at_pct=100*s["c"][1],
                         m_alpha=0.,m_alpha2=s["m"],q_alpha2=s["q"],r_alpha2=s["r"],
                         curvature_alpha_eV=s["curvature"][0],curvature_alpha2_eV=s["curvature"][1],
                         mu_eV_affine_gauge=s["tangent_mu_gauge"]))
    write_csv(out/"states.csv",rows)
    benchmarks=[("paper_contacts","model","static_S1","static_paper"),("paper_contacts","model","HA_Fig2_S1","HA_00"),
                ("native_contacts","model","static","static_native"),("native_contacts","model","HA","HA_11")]
    errors={}
    for filename,kind,label,key in benchmarks:
        with sources[filename].open() as f:
            row,=[r for r in csv.DictReader(f) if r[kind]==label and float(r["T_K"])==1000]
        ref=np.array([float(row[k]) for k in ("alpha_at_pct","alpha2_at_pct")])
        diff=100*states[key]["c"]-ref
        if np.max(np.abs(diff))>1e-4:
            raise ValueError(f"Failed prior independent endpoint comparison {key}: {diff}")
        errors[key]=diff.tolist()
    contrib=[]; sensitivity=[]; checks={}; class_sums=[]; factorial=[]
    ds=static1-static0; dv=vib1-vib0
    for mode,start,finish,delta_v in [("static","static_paper","static_native",np.zeros_like(dv)),
                                      ("HA","HA_00","HA_11",dv)]:
        j0,j1=jr[start],jr[finish]; total=j1-j0
        base,end=states[start],states[finish]
        delta_c=100*(end["c"]-base["c"])
        def integrate(nodes):
            x,w=leggauss(nodes)
            d=np.zeros((2,18)); bg=np.zeros_like(d)
            for lam,weight in zip((x+1)/2,w/2):
                s=model.solve(j0+lam*total,(1-lam)*base["x"]+lam*end["x"])
                d+=weight*s["derivatives"]; bg+=weight*s["background"]
            return 100*d,100*bg
        d8,bg8=integrate(8); d,bg=integrate(16)
        coord=d*total; random=bg*total; excess=coord-random
        for i,name in enumerate(NAMES):
            mixed0=j0.copy(); mixed0[i]=j1[i]
            mixed1=j1.copy(); mixed1[i]=j0[i]
            plus=100*(model.solve(mixed0,base["x"])["c"]-base["c"])
            minus=100*(end["c"]-model.solve(mixed1,end["x"])["c"])
            for k,branch in enumerate(("alpha","alpha2")):
                contrib.append(dict(mode=mode,branch=branch,orbit=name,J_paper_eV=j0[i],J_native_eV=j1[i],
                    delta_static_meV=1000*ds[i],delta_vib_meV=1000*delta_v[i],
                    path_total_at_pct=coord[k,i],path_static_at_pct=d[k,i]*ds[i],
                    path_vib_at_pct=d[k,i]*delta_v[i],random_background_at_pct=random[k,i],
                    order_excess_at_pct=excess[k,i],swap_from_paper_at_pct=plus[k],
                    swap_from_native_at_pct=minus[k],paper_order_energy_delta_meV=1000*total[i]*base["order_phi"][i],
                    path_derivative_at_pct_per_eV=d[k,i]))
        # Same mean-field sublattice power distribution gives indistinguishable
        # orbit directions up to multiplicity. This is an exact restriction of MFA.
        classes={}
        for i in range(18):
            norm=model.weights[i+2]/MULTIPLICITIES[i+2]
            fingerprint=tuple(norm.ravel())
            classes.setdefault(fingerprint,[]).append(i)
        for indices in classes.values():
            for k,branch in enumerate(("alpha","alpha2")):
                class_sums.append(dict(mode=mode,branch=branch,orbits=" ".join(NAMES[a] for a in indices),
                    effective_delta_J_meV=1000*sum(MULTIPLICITIES[a+2]*total[a] for a in indices),
                    path_total_at_pct=coord[k,indices].sum(),path_static_at_pct=(d*ds)[k,indices].sum(),
                    path_vib_at_pct=(d*delta_v)[k,indices].sum()))
        err=np.max(abs(coord-d8*total))
        closure=coord.sum(axis=1)-delta_c
        if err>1e-6 or np.max(abs(closure))>1e-6:
            raise ValueError(f"Nonconverged path integral: {mode}, error {err}, closure {closure}")
        checks[mode]={"actual_endpoint_shift_at_pct":delta_c.tolist(),"closure_error_at_pct":closure.tolist(),
                      "quad8_vs_16_max_orbit_at_pct":float(err),
                      "random_background_at_pct":random.sum(axis=1).tolist(),"order_excess_at_pct":excess.sum(axis=1).tolist()}
        for k,branch in enumerate(("alpha","alpha2")):
            series=[r for r in contrib if r["mode"]==mode and r["branch"]==branch]
            forward=np.array([r["swap_from_paper_at_pct"] for r in series])
            reverse=np.array([r["swap_from_native_at_pct"] for r in series])
            for method,values in (("swap_forward",forward),("swap_reverse",reverse)):
                orders=np.argsort(-abs(values))[:3]
                sensitivity.append(dict(mode=mode,branch=branch,method=method,
                    spearman_abs_vs_path=spearmanr(abs(coord[k]),abs(values)).statistic,
                    top3_orbits=" ".join(NAMES[a] for a in orders),
                    orbit_max_difference_at_pct=np.max(abs(values-coord[k])),
                    isolated_sum_at_pct=values.sum(),actual_at_pct=delta_c[k]))
            for sector,ii in (("pairs",slice(0,7)),("triplets",slice(7,None))):
                factorial.append(dict(mode=mode,branch=branch,quantity=sector,
                    shift_at_pct=coord[k,ii].sum(),static_at_pct=(d*ds)[k,ii].sum(),vib_at_pct=(d*delta_v)[k,ii].sum()))
    write_csv(out/"orbit_attribution.csv",contrib)
    write_csv(out/"mfa_orbit_classes.csv",class_sums)
    write_csv(out/"swap_rank_checks.csv",sensitivity)
    write_csv(out/"pair_triplet_sums.csv",factorial)
    effect=lambda hi,lo:100*(states[hi]["c"]-states[lo]["c"])
    facts={key:value.tolist() for key,value in {
        "static_on_paper_vib":effect("HA_10","HA_00"),"static_on_native_vib":effect("HA_11","HA_01"),
        "vib_on_paper_static":effect("HA_01","HA_00"),"vib_on_native_static":effect("HA_11","HA_10"),
        "static_vib_interaction":effect("HA_11","HA_01")-effect("HA_10","HA_00"),
        "two_group_shapley_static":.5*(effect("HA_10","HA_00")+effect("HA_11","HA_01")),
        "two_group_shapley_vib":.5*(effect("HA_01","HA_00")+effect("HA_11","HA_10"))}.items()}
    out.joinpath("checks.json").write_text(json.dumps({"T_K":1000,"volume_A3_atom":VREF,"prior_errors_at_pct":errors,
        "path_integrals":checks,"HA_crossed_factorial_at_pct":facts,
        "source_sha256":{key:hashlib.sha256(p.read_bytes()).hexdigest() for key,p in sources.items()}},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
    main(p.parse_args().out)
