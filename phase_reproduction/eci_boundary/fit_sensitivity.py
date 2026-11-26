"""Influence of individual stable training structures on the 1000 K MFA attribution."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import expit, logit, xlogy

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "thermo"))
from mfa import KB, Isotherm, sublattice_weights
from hcp_orbits import ORBIT_NAMES

KBT = 1000 * KB


def powers(p):
    z = 2*p - 1
    return np.array([[1.,z,z*z,z**3], [0.,2.,4*z,6*z*z], [0.,0.,8.,24*z]])


def entropy(p):
    return xlogy(p,p) + xlogy(1-p,1-p)


def thermodynamics(coef, q, r):
    # Derivatives in independent preferred (q) and other (r) site probabilities.
    vals = powers(q) @ coef @ powers(r).T
    pq, pr = .25*KBT, .75*KBT
    f = vals[0,0] + pq*entropy(q) + pr*entropy(r)
    df = np.array([vals[1,0] + pq*logit(q), vals[0,1] + pr*logit(r)])
    h = np.array([[vals[2,0] + pq/(q*(1-q)), vals[1,1]],
                  [vals[1,1], vals[0,2] + pr/(r*(1-r))]])
    return f, df, h


def equations(coef, p):
    a,q,r = p
    b = .25*q+.75*r
    fa, dfa, ha = thermodynamics(coef, a, a)
    fb, dfb, hb = thermodynamics(coef, q, r)
    mu, curvature = dfa.sum(), ha.sum()
    slopes = dfb - np.array([.25,.75])*mu
    residual = np.r_[slopes, fb-fa-mu*(b-a)]
    jac = np.array([[-.25*curvature, *hb[0]], [-.75*curvature, *hb[1]],
                    [-(b-a)*curvature, *slopes]])
    return residual, jac, fa, mu


def contact(w, j, seed):
    coef = np.einsum("i,ijk->jk", j, w)
    def fun(z):
        p = np.clip(expit(z),1.e-12,1-1.e-12)
        res,jac,*_ = equations(coef,p)
        return res, jac*(p*(1-p))
    for guess in [seed, [.07,.88,.0005], [.12,.76,.004], [.045,.95,.00008]]:
        sol = root(fun,logit(guess),jac=True,tol=1.e-10)
        p = expit(sol.x)
        if np.any(p<1e-12) or np.any(p>1-1e-12):
            continue
        res,jac,fa,mu = equations(coef,p)
        a,q,r = p
        b = .25*q+.75*r
        if np.max(abs(res))<2.e-9 and b-a>.015 and q-r>.1 and 0<a<b<.35:
            return p,jac,fa,mu
    raise ValueError("No robust Ti-rich ordered contact found")


def audit_tangent(w, j, result):
    p,_,fa,mu = result
    c = np.arange(351)[:,None]/1000
    q = c + np.linspace(0,1,81)[None,:]*(np.minimum(1,4*c)-c)
    values = Isotherm(w,j,1000).homogeneous(c,q)
    if np.min(values - (fa+mu*(c-p[0]))) < -2e-8:
        raise ValueError("Tangent crosses a lower phase on the scanned surface")


def alpha2(p):
    return 25*p[1]+75*p[2]


def derivative(w, p, jac):
    a,q,r = p
    b = .25*q+.75*r
    ma = powers(a) @ w @ powers(a).T
    mb = powers(q) @ w @ powers(r).T
    dmu = ma[:,1,0]+ma[:,0,1]
    direct = np.array([mb[:,1,0]-.25*dmu, mb[:,0,1]-.75*dmu,
                       mb[:,0,0]-ma[:,0,0]-(b-a)*dmu])
    implicit = -np.linalg.solve(jac,direct)
    return 25*implicit[1]+75*implicit[2]


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def save(rows,path):
    with path.open("w",encoding="utf8",newline="") as f:
        out = csv.DictWriter(f,fieldnames=rows[0].keys())
        out.writeheader()
        out.writerows(rows)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    inputs = read_csv(BASE/"atomistic/atat/output/design.csv")
    energies = {r["id"]:r for r in read_csv(BASE/"atat_comparison/MTP_labels_414.csv")
                if abs(float(r["V_over_Vref"])-1)<1e-8 and r["stable_mesh"].lower()=="true"}
    train = [r for r in inputs if r["split"]=="train" and r["id"] in energies]
    x = np.array([[float(r[k]) for k in ORBIT_NAMES] for r in train])
    ys = np.array([float(energies[r["id"]]["E_MTP_eV_atom"]) for r in train])
    yv = np.array([float(energies[r["id"]]["Fvib_1000K_eV_atom"]) for r in train])
    paper = json.loads((BASE/"thermo/results/coefficients_HA1000.json").read_text())
    w = sublattice_weights()
    roots, weights = np.polynomial.legendre.leggauss(8)
    contacts,terms = [],[]
    for model,y,key,expected_target,expected_ref in [
        ("static",ys,"static_S1_eV",22.593217,20.707655),
        ("HA",ys+yv,"effective_eV",21.079083,19.219401)]:
        ref = np.array(paper[key])
        ref[:2] = 0
        initial = contact(w,ref,[.08,.8,.002])
        audit_tangent(w,ref,initial)
        if abs(alpha2(initial[0])-expected_ref)>2.e-4:
            raise ValueError("Reference endpoint differs from the archived isotherm")
        for removed in [-1,*range(len(train))]:
            keep = np.arange(len(train))!=removed
            j,_,rank,svals = np.linalg.lstsq(x[keep],y[keep],rcond=None)
            ident = "none" if removed==-1 else train[removed]["id"]
            row = dict(model=model,omitted=ident,n_train=int(keep.sum()),rank=int(rank),
                       condition_number=svals[0]/svals[-1],alpha_at_pct="",alpha2_at_pct="",
                       shift_vs_paper_reconstruction="",q_alpha2="",r_alpha2="",sum_error_at_pct="",
                       status="rank_deficient")
            if rank==len(ORBIT_NAMES):
                j[:2] = 0
                delta = j-ref
                endpoint = contact(w,j,initial[0])
                audit_tangent(w,j,endpoint)
                p = endpoint[0]
                shift = alpha2(p)-alpha2(initial[0])
                if removed==-1 and abs(alpha2(p)-expected_target)>2.e-4:
                    raise ValueError("Full refit differs from the archived new isotherm")
                integrated = np.zeros(len(ref))
                guess = initial[0]
                for node,weight in zip(roots, weights):
                    guess,jac,_,_ = contact(w,ref+(.5*node+.5)*delta,guess)
                    integrated += (.5*weight)*derivative(w,guess,jac)*delta
                error = sum(integrated)-shift
                if abs(error)>2e-5:
                    raise ValueError(f"Path integral did not close for {model}/{ident}: {error}")
                row.update(alpha_at_pct=100*p[0],alpha2_at_pct=alpha2(p),
                           shift_vs_paper_reconstruction=shift,q_alpha2=p[1],r_alpha2=p[2],
                           sum_error_at_pct=error,status="ok")
                for k in range(2,len(j)):
                    terms.append(dict(model=model,omitted=ident,orbit=ORBIT_NAMES[k],
                                      delta_meV=1000*delta[k],contribution_at_pct=integrated[k]))
            contacts.append(row)
    save(contacts,out/"leave_one_out_contacts.csv")
    save(terms,out/"leave_one_out_contributions.csv")
    print(f"{len(contacts)} refits, {sum(r['status']=='ok' for r in contacts)} rank-sufficient solutions")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"robustness")
    run(p.parse_args().out)
