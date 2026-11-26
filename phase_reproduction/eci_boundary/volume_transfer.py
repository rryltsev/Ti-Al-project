"""Intervene on reference-volume ECI while retaining our measured QHA volume dependence."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from scipy.special import expit, logit

from fit_sensitivity import BASE, HERE, ORBIT_NAMES, powers, thermodynamics, sublattice_weights, save
from mfa import VolumeIsotherm, VREF


def coexistence(w, vol, j, guess):
    # Remove only ONE volume-independent affine term from each column.
    # Volume derivatives of the constant and point interactions remain present.
    js = j.copy()
    js[:,:2] -= js[np.argmin(abs(vol-VREF)),:2].copy()
    spline = CubicSpline(vol, np.einsum("ni,ijk->njk",js,w),axis=0,bc_type="natural")
    def residual(z,diagnostics=False):
        a,q,r = np.clip(expit(z[:3]),1.e-12,1-1.e-12)
        va,vb = z[3:]
        fa,dfa,_ = thermodynamics(spline(va),a,a)
        fb,dfb,_ = thermodynamics(spline(vb),q,r)
        mu = dfa.sum()
        b = (q+3*r)/4
        if diagnostics:
            return a,q,r,va,vb,fa,mu
        dva = powers(a)[0] @ spline(va,1) @ powers(a)[0]
        dvb = powers(q)[0] @ spline(vb,1) @ powers(r)[0]
        return np.r_[dfb-np.array([.25,.75])*mu,fb-fa-mu*(b-a),dva,dvb]
    z0 = np.r_[logit(guess[:3]),guess[3:]]
    sol = root(residual,z0,tol=1.e-10)
    a,q,r,va,vb,fa,mu = residual(sol.x,True)
    b = (q+3*r)/4
    if (max(abs(residual(sol.x)))>2e-8 or not (0<a<b<.35 and q>r+.1)
            or not (vol[0]<min(va,vb) and max(va,vb)<vol[-1])):
        raise ValueError("QHA tangent did not converge to two interior volumes")
    # Independent global check over chemical composition, order and all cubic volume intervals.
    model = VolumeIsotherm(w,vol,js,1000)
    c = np.linspace(0,.35,176)[:,None]
    qp = c + (np.minimum(1,4*c)-c)*np.linspace(0,1,49)[None,:]
    vals = model.homogeneous(c,qp)
    if np.min(vals - (fa+mu*(c-a))) < -2e-8:
        raise ValueError("Tangent crosses the relaxed QHA surface")
    return np.array([a,q,r,va,vb])


def run():
    with np.load(BASE/"atomistic/atat/qha_fit/grid.npz",allow_pickle=False) as f:
        if tuple(f["orbit_names"])!=ORBIT_NAMES:
            raise ValueError("Unexpected basis order")
        i, = np.flatnonzero(f["temperature_K"]==1000)
        v,j = f["volume_A3_atom"], f["J_static"]+f["J_vib"][:,i]
    ref = json.loads((BASE/"thermo/results/coefficients_HA1000.json").read_text())
    iv, = np.flatnonzero(abs(v-VREF)<1e-8)
    delta = j[iv]-ref["effective_eV"]
    delta[:2] = 0.
    anchored = j - delta
    w = sublattice_weights()
    rows = []
    for origin,j0,sign in [("own",j,-1),("paper_anchored",anchored,1)]:
        init = coexistence(w,v,j0,[.094,.87,.002,17.58,17.24])
        baseline = 25*init[1]+75*init[2]
        if origin=="own" and abs(baseline-22.397749)>1.e-4:
            raise ValueError("Full-grid QHA fails previously measured endpoint")
        groups = {"none":[],**{k:[i] for i,k in enumerate(ORBIT_NAMES) if i>=2},
                  "t334+t123":[ORBIT_NAMES.index(k) for k in ("t334","t123")],
                  "pairs":list(range(2,9)),"triplets":list(range(9,20)),"all":list(range(2,20))}
        for label,indices in groups.items():
            jt = j0.copy()
            jt[:,indices] += sign*delta[indices]
            a,q,r,va,vb = coexistence(w,v,jt,init) if indices else init
            rows.append(dict(origin=origin,replaced=label,alpha_at_pct=100*a,
                             alpha2_at_pct=25*q+75*r,shift_alpha2_at_pct=25*q+75*r-baseline,
                             q_alpha2=q,r_alpha2=r,volume_alpha_A3=va,volume_alpha2_A3=vb))
    target = HERE/"robustness"
    target.mkdir(parents=True,exist_ok=True)
    save(rows,target/"qha_reference_volume_swaps.csv")
    for row in rows:
        if row["replaced"] in ("none","t334","t123","t334+t123","all"):
            print(row["origin"],row["replaced"],round(row["alpha2_at_pct"],4),round(row["shift_alpha2_at_pct"],4))


if __name__=="__main__":
    run()
