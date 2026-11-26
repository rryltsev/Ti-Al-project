"""Compare the complete C46 and independent QHA Hamiltonians at 1000 K, P=0."""
import csv
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0,str(BASE/"eci_boundary"))
from fit_sensitivity import ORBIT_NAMES, powers, thermodynamics, sublattice_weights
from mfa import VolumeIsotherm, VREF


def coefficients(path):
    with np.load(path,allow_pickle=False) as f:
        assert tuple(f["orbit_names"])==ORBIT_NAMES
        t, = np.flatnonzero(f["temperature_K"]==1000)
        return f["volume_A3_atom"],f["J_static"]+f["J_vib"][:,t]


def solve(v,w,j,bc,guess=None):
    poly = CubicSpline(v,np.einsum("vi,ijk->vjk",j,w),axis=0,bc_type=bc)
    def equations(z):
        a,q,r=np.clip(expit(z[:3]),1e-12,1-1e-12)
        va,vb=z[3:]
        fa,dfa,_=thermodynamics(poly(va),a,a)
        fb,dfb,_=thermodynamics(poly(vb),q,r)
        mu=dfa.sum()
        da=powers(a)[0] @ poly(va,1) @ powers(a)[0]
        db=powers(q)[0] @ poly(vb,1) @ powers(r)[0]
        return np.r_[dfb-[mu*.25,mu*.75],fb-fa-mu*((q+3*r)*.25-a),da,db]
    if guess is None:
        guess=np.r_[logit([.09,.84,.002]),17.58,17.24]
    z=root(equations,guess,tol=1e-10).x
    a,q,r=expit(z[:3])
    b=(q+3*r)/4
    if max(abs(equations(z)))>2.e-8 or not (0<a<b<.35 and q>r+.1 and
            min(z[3:])>v[0] and max(z[3:])<v[-1]):
        raise ValueError("QHA phase contact is not interior or has not converged")
    # Global support is tested after minimizing over all stationary points in
    # volume on each cubic interval, including both endpoints.
    test=VolumeIsotherm(w,v,j,1000,volume_spline=bc)
    c=np.linspace(0,.35,176)[:,None]
    qp=c+(np.minimum(1,4*c)-c)*np.linspace(0,1,49)
    fa,dfa,_=thermodynamics(poly(z[3]),a,a)
    if np.min(test.homogeneous(c,qp)-fa-dfa.sum()*(c-a)) < -2e-8:
        raise ValueError("Tangent is not globally supporting")
    return z,equations


def alpha2(z):
    return 25*expit(z[1])+75*expit(z[2])


def partials(z,eq,w,dj):
    # Differentiate the five coexistence/zero-pressure equations, treating the
    # 20 entire volume-dependent interaction functions as independent directions.
    eye=np.eye(5)*.00002
    jac=np.column_stack([(eq(z+e)-eq(z-e))/.00004 for e in eye])
    a,q,r=expit(z[:3])
    va,vb=z[3:]
    b=(q+3*r)/4
    aa=powers(a)@w@powers(a).T
    bb=powers(q)@w@powers(r).T
    ja,jb=dj(va),dj(vb)
    dmu=ja*(aa[:,1,0]+aa[:,0,1])
    residual=np.array([jb*bb[:,1,0]-dmu*.25,jb*bb[:,0,1]-dmu*.75,
        jb*bb[:,0,0]-ja*aa[:,0,0]-(b-a)*dmu,dj(va,1)*aa[:,0,0],dj(vb,1)*bb[:,0,0]])
    dz=-np.linalg.solve(jac,residual)
    return 25*q*(1-q)*dz[1] + 75*r*(1-r)*dz[2]


def write(path,rows):
    with path.open("w",encoding="utf8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)


def run():
    v,pub=coefficients(HERE/"eci/author_refit_grid.npz")
    v2,own=coefficients(BASE/"atomistic/atat/qha_fit/grid.npz")
    assert np.allclose(v,v2,atol=5e-12,rtol=0)
    iv,=np.flatnonzero(abs(v-VREF)<1e-8)
    # Uniform affine changes leave coexistence invariant even when volume is free.
    for j in [pub,own]:
        j[:,:2]-=j[iv,:2].copy()
    full=own-pub
    offset=np.broadcast_to(full[iv],full.shape)
    shape=full-offset
    affine=np.zeros_like(full)
    affine[:,:2]=full[:,:2]
    w=sublattice_weights()
    out=HERE/"qha"
    out.mkdir(exist_ok=True)
    swaps,attributions=[],[]
    for bc in ("not-a-knot","natural"):
        spline=lambda x:CubicSpline(v,x,axis=0,bc_type=bc)
        z0,_=solve(v,w,pub,bc)
        z1,_=solve(v,w,own,bc)
        total=alpha2(z1)-alpha2(z0)
        contrib=np.zeros((2,20))
        do,ds=spline(offset),spline(shape)
        z=z0
        nodes,weights=np.polynomial.legendre.leggauss(16)
        for node,weight in zip(nodes,weights):
            z,eq=solve(v,w,pub+(.5+.5*node)*full,bc,z)
            for index,direction in enumerate([do,ds]):
                contrib[index]+=weight*.5*partials(z,eq,w,direction)
        if abs(contrib.sum()-total)>0.0001:
            raise ValueError(f"Attribution does not close: {contrib.sum()-total}")
        for i,k in enumerate(ORBIT_NAMES):
            attributions.append(dict(spline=bc,orbit=k,reference_volume_at_pct=contrib[0,i],
                volume_shape_at_pct=contrib[1,i],full_at_pct=contrib[:,i].sum()))
        both=np.zeros_like(full)
        for k in ("t334","t123"):
            both[:,ORBIT_NAMES.index(k)]=full[:,ORBIT_NAMES.index(k)]
        for origin,jstart,sign in (("C46",pub,1),("ours",own,-1)):
            for label,change in [("none",0),("vref_18",offset),("volume_shape_all",shape),
                                 ("volume_shape_J0_J1",affine),("t334+t123_all_volumes",both),
                                 ("all",full)]:
                z,_=solve(v,w,jstart+sign*change,bc)
                swaps.append(dict(spline=bc,origin=origin,swapped=label,
                    alpha_at_pct=100*expit(z[0]),alpha2_at_pct=alpha2(z),
                    v_alpha_A3_atom=z[3],v_alpha2_A3_atom=z[4]))
        print(bc,alpha2(z0),alpha2(z1),"vref, shape",contrib.sum(axis=1),
              "closure",contrib.sum()-total)
        print(sorted(zip(ORBIT_NAMES,contrib.sum(axis=0)),key=lambda e:-abs(e[1]))[:5])
    write(out/"qha_attribution.csv",attributions)
    write(out/"qha_swaps.csv",swaps)


if __name__=="__main__":
    run()
