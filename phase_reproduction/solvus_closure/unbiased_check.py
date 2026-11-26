"""Independent phase-conditional composition/order/energy from zero-umbrella trajectories."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from thermal_check import ROOT,mbar


def main(a):
    target=json.loads(a.vertex.read_text(encoding="utf8"))
    t,n,mu,cut=target["central"]["T_K"],target["N"],target["central"]["mu_eV"],target["Q_divider"]
    beta=1/(mbar.KB*t)
    fields=defaultdict(list)
    for path in a.manifests:
        with path.open(encoding="utf-8-sig",newline="") as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            if int(row["N"])!=n or float(row["K_eV"])!=0 or float(row["cap"])!=target["cap"] or int(row["axial_free"])!=target["axial_free"]:
                raise ValueError("Control differs in size, umbrella, wall or axial ensemble")
            tr=np.genfromtxt(path.parent/(row["run"]+".csv"),delimiter=",",names=True)
            tr=tr[tr["sweep"]>int(row["burn"])]
            if len(tr)!=int(row["sweeps"])//int(row["stride"]) or not np.allclose(tr["T_K"],float(row["T_K"])):
                raise ValueError("Control trace length/temperature mismatch")
            fields[float(row["T_K"]),float(row["mu_eV"])].append((row["run"],row["start"],tr))
    out=[]
    for (t0,mu0),r in sorted(fields.items()):
        beta0=1/(mbar.KB*t0)
        data=[]; logw=[]; autocorr=[]; chains=[]
        for name,start,tr in r:
            k,q,u=tr["n_Al"],tr["Q"],tr["E_eV"]
            g=max(mbar.tau_int(v) for v in (k,q,u-mu*k,(q>cut).astype(float)))
            # Center all trajectories by the same arbitrary energy constant.
            energy=u-mu*k+7*n
            data.append(np.stack([100*k/n,q,1000*energy/n],axis=1))
            logw.append((beta0-beta)*energy+beta0*(mu-mu0)*k)
            autocorr.append(np.full(len(tr),g))
            chains.append(dict(name=name,start=start,fraction_high_Q=float(np.mean(q>cut)),g=g))
        obs=np.vstack(data); logs=np.concatenate(logw); gs=np.concatenate(autocorr)
        phases=[]
        for sel in (obs[:,1]<=cut,obs[:,1]>cut):
            if not sel.any():
                phases.append(None); continue
            p=np.exp(logs[sel]-logsumexp(logs[sel]))
            x,q,h=p@obs[sel]
            phases.append(dict(x_atpct=float(x),mean_Q=float(q),shifted_energy_meV_atom=float(h),
                               ESS_corrected=float(1/((p*p)@gs[sel])),frames=int(sel.sum())))
        heat=None if None in phases else phases[0]["shifted_energy_meV_atom"]-phases[1]["shifted_energy_meV_atom"]
        out.append(dict(T_K=t0,mu_eV=mu0,chains=chains,phases=phases,latent_meV_atom=heat))
    final=dict(vertex_source=a.vertex.resolve().relative_to(ROOT).as_posix(),target_T=t,target_mu=mu,N=n,Q_divider=cut,
               data=out,note="Conditional means only. Numbers or durations of deliberately chosen starting phases do not supply relative free energies.")
    a.out.write_text(json.dumps(final,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("vertex",type=Path)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
