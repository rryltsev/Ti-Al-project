"""Verify local temperature reweighting and the congruent-root equations exactly."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from thermal_check import mbar, tilted_weights
from vertex_bootstrap import solve
from pooled import Mixture


def main():
    # Two phases have identical mean composition at the prescribed maximum;
    # the disordered phase has a larger concentration variance. Each (k,Q)
    # bin contains two different unbiased energies.
    n, source_t, target_t, source_mu, target_mu=64,1273.15,1244.,3.527,3.551
    k=np.array([12,20,15,17])
    q=np.array([.10,.10,.85,.85])
    idx=np.repeat(np.arange(4),2)
    semigrand=np.repeat([.80,.80,-.60,-.60],2)+np.tile([-.035,.035],4)
    energy=semigrand+target_mu*k[idx]
    degeneracy=np.log(np.repeat([1.,1.,4.,4.],2))+semigrand/(mbar.KB*target_t)
    energy_ref=energy-source_mu*k[idx]
    source=degeneracy-energy_ref/(mbar.KB*source_t)
    model=dict(idx=idx,k=k,q=q,g=np.ones(8),ref=source_mu,energy=energy_ref-energy_ref.mean(),
               log_frame=source-logsumexp(source))
    b,_=tilted_weights(model,source_t,target_t)
    logs=b+(target_mu-source_mu)*k/(mbar.KB*target_t)
    probs=np.exp(logs-logsumexp(logs))
    expected=np.array([.1,.1,.4,.4])
    fit=solve(model,model["log_frame"],n,source_t,.4,(1235.,3.54),[1200.,1280.,3.4,3.65],.04)
    assert np.max(abs(probs-expected))<1e-11 and abs(fit["T_K"]-target_t)<1e-6
    assert abs(fit["mu_eV"]-target_mu)<1e-9 and abs(fit["x_atpct"]-25.)<1e-7
    assert abs(fit["latent_meV_atom"]-1000*1.4/n)<1e-7
    p_ref=np.array([.2,.2,.3,.3])
    p_sources=np.array([[.1,.3,.5,.1],[.3,.4,.1,.2]])
    labels=np.repeat([0,1],[20,30])
    indices=np.concatenate([np.repeat(np.arange(4),np.rint(size*prob).astype(int))
                            for size,prob in zip([20,30],p_sources)])
    u=(-np.log(p_sources/p_ref)+np.array([1.4,-2.2])[:,None])[:,indices]
    pooled=Mixture(u,labels)
    def normmass(lw,key):
        return np.bincount(key,weights=np.exp(lw-logsumexp(lw)),minlength=4)
    pool_error=max(abs(normmass(-pooled.logden,indices)-p_ref))
    m=(np.arange(len(indices)))%3
    expanded=np.repeat(np.arange(len(indices)),m)
    pfull=Mixture(u[:,expanded],labels[expanded])
    boot_error=max(abs(normmass(pooled.bootstrap(m),indices)-normmass(-pfull.logden,indices[expanded])))
    assert pool_error<5e-8 and boot_error<5e-8
    out=dict(maximum_bin_probability_error=float(np.max(abs(probs-expected))),
             analytical_T_K=target_t,recovered_T_K=fit["T_K"],
             analytical_mu_eV=target_mu,recovered_mu_eV=fit["mu_eV"],
             analytical_Al_atpct=25.,recovered_Al_atpct=fit["x_atpct"],
             analytical_heat_meV_atom=1000*1.4/n,recovered_heat_meV_atom=fit["latent_meV_atom"],
             multitemperature_MBAR_exact_histogram_error=float(pool_error),
             bootstrap_vs_explicit_repeat_MBAR_max_error=float(boot_error))
    target=Path(__file__).resolve().parent/"analysis/thermal/validation.json"
    target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
