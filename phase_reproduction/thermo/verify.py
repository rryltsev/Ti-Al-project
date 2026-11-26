"""Analytic regression checks for QHA volume selection, TI and discrete WHAM."""
import numpy as np
from scipy.interpolate import CubicSpline
from mfa import KB, VREF, VolumeIsotherm, Isotherm, sublattice_weights
from vcsgc import integrate, lower_hull, wham


def main():
    weights=sublattice_weights()
    v=VREF*np.array([.94,.96,.98,1.02,1.04,1.06])
    j=np.zeros((len(v),20))
    center=VREF
    j[:,0]=(v-center)**2*2
    j[:,1]=1.2
    j[:,2]=.05
    model=VolumeIsotherm(weights,v,j,1000)
    c=np.array([.1,.16,.25,.34])
    test_j=j[0].copy(); test_j[0]=0
    ref=Isotherm(weights,test_j,1000)
    value,q=model.equilibrium(c)
    expected,_=ref.equilibrium(c)
    interpolated_min=CubicSpline(v,(v-center)**2,bc_type="natural")(center)
    np.testing.assert_allclose(value,expected+interpolated_min,atol=1e-11)
    np.testing.assert_allclose(model.minimize_volume(c,q)[1],center,atol=1e-8)
    # Monotone volume input must return the endpoint.
    j[:,0]=v
    edge=VolumeIsotherm(weights,v,j,1000)
    np.testing.assert_allclose(edge.minimize_volume(c,q)[1],v[0])
    # Both convex-hull contacts are known for this tilted double-well.
    c=np.linspace(.01,.39,151)
    mu=2*(c-.1)*(c-.3)*(2*c-.4)+.06
    _,_,intervals=integrate(c,mu,.005)
    np.testing.assert_allclose([intervals[0]["alpha_at_pct"],intervals[0]["alpha2_at_pct"]],[10.,30.],atol=.015)
    # A common tangent can exist mathematically but be below the SI resolution.
    concentrated=np.linspace(.235,.26,1001)
    profile=(concentrated-.245)**2*(concentrated-.2495)**2
    assert len(lower_hull(concentrated,profile,0))==1
    assert lower_hull(concentrated,profile,.005)==[]
    # Generate integer population histograms directly from specified weights;
    # compare the WHAM curve with the known Boltzmann free energy on well-sampled bins.
    n,t=64,1000.
    bins=np.arange(n+1); c=bins/n
    settings=[(x,9,.12) for x in (.08,.14,.2,.26,.32)]
    true=3*(c-.1)**2*(c-.3)**2+.12*c
    windows=[]
    for c0,k,mu0 in settings:
        logp=-n*(true-mu0*c)/(KB*t)-n*k*(c-c0)**2
        p=np.exp(logp-logp.max()); p/=p.sum()
        windows.append(np.repeat(bins,np.rint(p*100000).astype(int)))
    used=np.unique(np.concatenate(windows))
    wh,curve=wham(used,windows,settings,n,t,.005)
    assert wh["status"]=="connected"
    x,f=np.array(list(curve)).T
    central=(x>.03)&(x<.38)
    diff=f[central]-np.interp(x[central],c,true)
    assert np.ptp(diff)<3e-6
    split,_=wham(np.array([1,2,60,61]),[np.array([1,2]),np.array([60,61])],settings[:2],n,t,.005)
    assert split["overlap_components"]==2
    print("PASS: interior/endpoint QHA minima, analytic tangent, connected WHAM (energy), disconnected WHAM rejection")


if __name__=="__main__":
    main()
