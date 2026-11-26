"""D019 Bragg-Williams free energy and equilibrium common tangents."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.interpolate import CubicSpline
from scipy.special import xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atomistic"))
from hcp_orbits import ORBIT_NAMES, VREF, build_orbits, sublattices

KB = 8.617333262145e-5  # eV/K


def sublattice_weights():
    # Large enough for every cluster to have distinct sites under PBC.
    shape = (4, 4, 4)
    labels = sublattices(shape)
    orbits = build_orbits(shape)
    counts = np.zeros((len(orbits), 4, 4))
    counts[0, 0, 0] = .5
    for i, sites in enumerate(orbits[1:], 1):
        nq = (labels[sites] == 0).sum(axis=1)
        for k in range(sites.shape[1] + 1):
            counts[i, k, sites.shape[1] - k] = np.count_nonzero(nq == k)/len(labels)
    return counts


def entropy_term(p):
    p = np.clip(p, 0, 1)
    return xlogy(p, p) + xlogy(1-p, 1-p)


class Isotherm:
    def __init__(self, weights, j, temperature):
        self.kbt = KB*temperature
        self.coef = np.einsum("i,ijk->jk", j, weights)

    def homogeneous(self, c, q):
        r = (4*c-q)/3
        energy = np.polynomial.polynomial.polyval2d(2*q-1, 2*r-1, self.coef)
        return energy + self.kbt*(entropy_term(q) + 3*entropy_term(r))/4

    def equilibrium(self, c):
        """Find the lowest free energy including both ordered and disordered minima."""
        c = np.atleast_1d(c)
        grid = c[:, None] + np.linspace(0, 1, 65)*(np.minimum(1, 4*c)-c)[:, None]
        values = self.homogeneous(c[:, None], grid)
        best = np.argmin(values, axis=1)
        qbest = grid[np.arange(c.size), best]
        fbest = values[np.arange(c.size), best]
        local = (values[:, 1:-1] <= values[:, :-2]) & (values[:, 1:-1] < values[:, 2:])
        rows, cols = np.where(local)
        # Include edge intervals for ordered or disordered minima that lie close
        # enough to an endpoint to be missed by the 65-point order scan.
        rows = np.r_[rows, np.arange(c.size), np.arange(c.size)]
        lo = np.r_[grid[rows[:len(cols)], cols], grid[:, 0], grid[:, -2]]
        hi = np.r_[grid[rows[:len(cols)], cols+2], grid[:, 1], grid[:, -1]]
        for _ in range(49):
            a = (2*lo+hi)/3
            b = (lo+2*hi)/3
            right = self.homogeneous(c[rows], a) > self.homogeneous(c[rows], b)
            lo, hi = np.where(right, a, lo), np.where(right, hi, b)
        q = .5*(lo+hi)
        f = self.homogeneous(c[rows], q)
        for i, qi, fi in zip(rows, q, f):
            if fi < fbest[i]:
                fbest[i], qbest[i] = fi, qi
        return fbest, qbest

    def derivative(self, c):
        dc = 1.e-6
        f, _ = self.equilibrium(np.r_[c-dc, c+dc])
        return (f[len(c):]-f[:len(c)])/(2*dc)


class VolumeIsotherm(Isotherm):
    """Exact minimum of the interpolated volume cubic at each pair (c,q), p=0."""
    def __init__(self, weights, volume, j, temperature, volume_spline="natural"):
        self.kbt = KB*temperature
        self.volume = volume
        self.spline = CubicSpline(volume, np.einsum("vi,ijk->vjk",j,weights), axis=0, bc_type=volume_spline)

    def minimize_volume(self,c,q):
        r = (4*c-q)/3
        s, u = 2*q-1, 2*r-1
        # Compute (s^i * u^j) only for the supported polynomial terms.
        moments = np.stack(np.broadcast_arrays(*[s**i * u**j
                           for i in range(4) for j in range(4-i)]))
        coef = np.stack([self.spline.c[:,:,i,j] for i in range(4) for j in range(4-i)],axis=-1)
        raw = coef @ moments.reshape(len(moments),-1)
        a,b,d,e = raw
        width = np.diff(self.volume)[:,None]
        # Include endpoints and every real internal stationary point of av³+bv²+dv+e.
        z = np.zeros_like(a)
        h = np.broadcast_to(width,z.shape)
        candidates = [z,h]
        eps = 1e-13
        disc = b*b-3*a*d
        root_term = np.sqrt(np.maximum(disc,0))
        for sign in (-1,1):
            x = np.divide(-b+sign*root_term,3*a,out=z.copy(),where=abs(a)>eps)
            candidates.append(np.where((abs(a)>eps)&(disc>=0)&(x>0)&(x<width),x,0))
        x = np.divide(-d,2*b,out=z.copy(),where=abs(b)>eps)
        candidates.append(np.where((abs(a)<=eps)&(abs(b)>eps)&(x>0)&(x<width),x,0))
        emin = np.full(a.shape[1],np.inf)
        vmin = np.zeros_like(emin)
        col = np.arange(a.shape[1])
        for offset in candidates:
            val = ((a*offset+b)*offset+d)*offset+e
            i = val.argmin(axis=0)
            use = val[i,col]<emin
            vmin[use] = (self.volume[i]+offset[i,col])[use]
            emin[use] = val[i,col][use]
        return emin.reshape(s.shape),vmin.reshape(s.shape)

    def homogeneous(self,c,q):
        f,_ = self.minimize_volume(c,q)
        return f+self.kbt*(entropy_term(q)+3*entropy_term((4*c-q)/3))/4


def lower_hull_edges(x, f):
    hull = []
    for k in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2:]
            if (f[b]-f[a])*(x[k]-x[b]) < (f[k]-f[b])*(x[b]-x[a]):
                break
            hull.pop()
        hull.append(k)
    return list(zip(hull, hull[1:]))


def mfa_contacts(model, c, f, min_span=.005):
    contacts = []
    for l, r in lower_hull_edges(c, f):
        if r == l+1 or c[r]-c[l] <= min_span:
            continue
        if l == 0 or r == len(c)-1:
            contacts.append((c[l], c[r], "window_edge", None))
            continue
        grid_pair = c[[l, r]]
        def residual(pair):
            y, _ = model.equilibrium(pair)
            mu = model.derivative(pair)
            return mu - (y[1]-y[0])/(pair[1]-pair[0])
        solution = root(residual, grid_pair, tol=1.e-8)
        x = solution.x
        if (np.max(abs(residual(x))) > 2.e-7 or np.max(abs(x-grid_pair)) > 2*(c[1]-c[0])
                or x[0] >= x[1] or not np.all(np.isfinite(x))):
            slope=(f[r]-f[l])/(c[r]-c[l])
            contacts.append((*grid_pair,"grid_only_refinement_unresolved",slope))
            continue
        y, _ = model.equilibrium(x)
        slope = (y[1]-y[0])/(x[1]-x[0])
        if np.min(f-y[0]-slope*(c-x[0])) < -2e-8:
            raise ValueError("Refined tangent crosses the free energy")
        contacts.append((*x, "interior_tangent", slope))
    return contacts


def run(coeff_file, output):
    coeff = json.loads(coeff_file.read_text(encoding="utf8"))
    if tuple(coeff["orbit_names"]) != ORBIT_NAMES:
        raise ValueError("Incorrect orbital order")
    weights = sublattice_weights()
    c = np.linspace(0, .35, 701)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"mfa_contacts.csv").open("w", encoding="utf8", newline="") as results, (
            output/"mfa_free_energy.csv").open("w", encoding="utf8", newline="") as surfaces:
        writer, curves = csv.writer(results), csv.writer(surfaces)
        writer.writerow(("model", "T_K", "status", "alpha_at_pct", "alpha2_at_pct",
                         "tangent_mu_eV", "paper_alpha_at_pct", "paper_alpha2_at_pct"))
        curves.writerow(("model", "T_K", "c_Al", "q_Al", "r_Al", "f_eV_atom"))
        for kind, temps, j in [("static_S1", range(600, 1801, 100), coeff["static_S1_eV"]),
                               ("HA_Fig2_S1", [1000], coeff["effective_eV"])]:
            for t in temps:
                model = Isotherm(weights, j, t)
                f, q = model.equilibrium(c)
                curves.writerows(zip([kind]*len(c), [t]*len(c), c, q, (4*c-q)/3, f))
                found = mfa_contacts(model, c, f)
                target = ([6.16, 20.80] if kind == "static_S1" else [8.93, 19.42]) if t == 1000 else ["",""]
                for a, b, status, mu in found:
                    writer.writerow((kind,t,status,100*a,100*b,mu,*target))
                if not found:
                    writer.writerow((kind,t,"unresolved","","","",*target))


def run_grid(path, output, temperatures=None, requested_models=None, volume_spline="natural"):
    """Evaluate static, HA, QHA MFA for a machine-readable CE grid."""
    with np.load(path,allow_pickle=False) as data:
        names=tuple(str(k.decode() if isinstance(k,bytes) else k) for k in data["orbit_names"])
        volume,static=np.asarray(data["volume_A3_atom"]),data["J_static"]
        vib=data["J_vib"] if "J_vib" in data else None
        temps=data["temperature_K"] if vib is not None else np.array(temperatures or range(600,1801,100))
    if (names!=ORBIT_NAMES or volume.ndim!=1 or temps.ndim!=1 or len(volume)<1 or
        static.shape!=(len(volume),20) or (vib is not None and (
            vib.shape!=(len(volume),len(temps),20) or not np.isfinite(vib).all())) or
        not all(np.all(np.isfinite(a)) for a in [volume,temps,static]) or
            np.any(np.diff(volume)<=0) or np.any(temps<0) or not np.any(temps>0) or
            len(np.unique(temps))!=len(temps)):
        raise ValueError("Invalid CE coefficient/volume/temperature grid")
    anchors=np.flatnonzero(np.isclose(volume,VREF,rtol=1e-7))
    if len(anchors)!=1:
        raise ValueError("The HA grid must include the specified reference volume")
    iref=anchors[0]
    if temperatures and not set(temperatures)<=set(temps):
        raise ValueError("Requested T is absent from the grid")
    if temperatures and any(t<=0 for t in temperatures):
        raise ValueError("Finite-temperature MFA contacts require T > 0")
    weights=sublattice_weights()
    output.mkdir(parents=True,exist_ok=True)
    c=np.linspace(0,.35,701)
    with (output/"grid_mfa_contacts.csv").open("w",newline="") as output1, (
            output/"grid_mfa_free_energy.csv").open("w",newline="") as output2:
        summary,surfaces=csv.writer(output1),csv.writer(output2)
        summary.writerow(("model","T_K","status","alpha_at_pct","alpha2_at_pct","tangent_mu_eV",
                          "v_alpha","v_alpha2","volume_endpoint"))
        surfaces.writerow(("model","T_K","c_Al","q_Al","r_Al","v_A3_atom","f_eV_atom"))
        for k,t in enumerate(temps):
            if t==0 or (temperatures and t not in temperatures):
                continue
            models={"static":Isotherm(weights,static[iref],t)}
            if vib is not None:
                combined=static+vib[:,k,:]
                models["HA"]=Isotherm(weights,combined[iref],t)
                if len(volume)>=4:
                    models["QHA"]=VolumeIsotherm(weights,volume,combined,t,volume_spline=volume_spline)
            if requested_models and not set(requested_models)<=models.keys():
                raise ValueError("Requested model is absent from the supplied grid")
            for label,model in models.items():
                if requested_models and label not in requested_models:
                    continue
                f,q=model.equilibrium(c)
                v=model.minimize_volume(c,q)[1] if label=="QHA" else np.full_like(c,VREF)
                surfaces.writerows(zip([label]*len(c),[t]*len(c),c,q,(4*c-q)/3,v,f))
                found=mfa_contacts(model,c,f)
                for a,b,status,mu in found:
                    ab=np.array([a,b])
                    if label=="QHA":
                        _,qcontact=model.equilibrium(ab)
                        vs=model.minimize_volume(ab,qcontact)[1]
                    else:
                        vs=np.array([VREF,VREF])
                    clipped=label=="QHA" and (np.isclose(vs,volume[0])|np.isclose(vs,volume[-1])).any()
                    summary.writerow((label,t,status,100*a,100*b,mu,*vs,clipped))
                if not found:
                    summary.writerow((label,t,"unresolved","","","","","",""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("coefficients", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--temperatures", nargs="+", type=float, help="subset for a full NPZ grid")
    p.add_argument("--models", nargs="+", choices=["static","HA","QHA"], help="subset for a fitted NPZ grid")
    p.add_argument("--volume-spline", choices=["natural","not-a-knot"], default="natural",
                   help="QHA cubic spline at the volume endpoints; C46 uses not-a-knot")
    args = p.parse_args()
    if args.coefficients.suffix.lower()==".npz":
        run_grid(args.coefficients,args.output,args.temperatures,args.models,args.volume_spline)
    else:
        run(args.coefficients, args.output)
