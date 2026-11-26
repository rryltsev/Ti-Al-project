"""Equal 20-ps spectra; uncertainty clusters both halves of each new replica."""
import json
import attrs
import numpy as np
import analysis_common as a
from stacie import compute_spectrum,estimate_acint,LorentzModel
from stacie.utils import UnitConfig

def spectrum(seq,pref):
    return compute_spectrum(seq,prefactors=np.asarray(pref),timestep=1e-15)

def merge(spectra):
    assert spectra
    first=spectra[0]
    assert all(s.nstep==first.nstep and np.array_equal(s.freqs,first.freqs) for s in spectra)
    weights=np.array([s.ndofs[0] for s in spectra]); weights=weights/weights.sum()
    return attrs.evolve(first,mean=sum(w*s.mean for w,s in zip(weights,spectra)),
        variance=sum(w*s.variance for w,s in zip(weights,spectra)),
        amplitudes=sum(w*s.amplitudes for w,s in zip(weights,spectra)),
        ndofs=sum(s.ndofs for s in spectra))

def estimate(s):
    uc=UnitConfig(acint_fmt='.4f',acint_symbol='eta',acint_unit_str='mPa*s',acint_unit=1e-3,
        freq_unit_str='THz',freq_unit=1e12,time_fmt='.3f',time_unit_str='ps',time_unit=1e-12)
    r=estimate_acint(s,LorentzModel(),verbose=False,uc=uc,rng=np.random.default_rng(42))
    prop=lambda k:float(r.props[k]) if r.props.get(k) is not None and np.isfinite(r.props[k]) else None
    out=dict(eta_mPa_s=float(r.acint)*1000,eta_std_mPa_s=float(r.acint_std)*1000,
        relative_std=float(r.acint_std/r.acint),neff=float(r.neff),cost_z=prop('cost_zscore'),criterion_z=prop('criterion_zscore'))
    assert all(np.isfinite(v) for v in out.values() if v is not None) and out['eta_mPa_s']>0
    return out,r,uc

def legacy_clusters():
    cfg=a.CFG['legacy'];path=a.ROOT/cfg['manifest']
    assert a.sha(path)==cfg['sha256']
    manifest=json.loads(path.read_text()); assert a.sha(a.ROOT/'legacy/stress.npz')==manifest['stress_sha256']
    sources=manifest['raw_manifest']; assert len({x['source_pressure_sha256'] for x in sources})==len(sources)
    clusters=[]
    with np.load(a.ROOT/'legacy/stress.npz') as d:
        assert d['pressure_bar'].shape==(cfg['n_replicas'],20001,6)
        assert float(d['temperature_K'])==a.CFG['states'][0]['temperature_K'] and float(d['dt_ps'])==.001
        for p,v in zip(d['pressure_bar'],d['volume_A3']):
            clusters.append(spectrum(a.channels(p*1e5),[a.prefactor(float(v),float(d['temperature_K']))]*5))
    return clusters,manifest

def new_cluster(seq,pref):
    assert seq.shape==(5,40001) and len(pref)==5
    return spectrum(np.concatenate([seq[:,:20001],seq[:,20000:]],axis=0),np.tile(pref,2))

def jackknife(values):
    values=np.asarray(values);assert len(values)>1
    return float(np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean())**2)))

def combine(clusters,n_legacy,n_new):
    out,r,uc=estimate(merge(clusters))
    leave=[]
    for i in range(len(clusters)):
        item,_,_=estimate(merge(clusters[:i]+clusters[i+1:]));leave.append(item['eta_mPa_s'])
    out['formal_std_mPa_s']=out['eta_std_mPa_s'];out['cluster_jackknife_std_mPa_s']=jackknife(leave)
    out['eta_std_mPa_s']=max(out['formal_std_mPa_s'],out['cluster_jackknife_std_mPa_s'])
    out['relative_std']=out['eta_std_mPa_s']/out['eta_mPa_s']
    out.update(legacy_replicas=n_legacy,new_40ps_replicas=n_new,replica_clusters=len(clusters),windows_20ps=n_legacy+2*n_new,
        leave_one_replica_out_mPa_s=leave,uncertainty='max(formal,whole-replica-cluster-jackknife)',pooling_version=a.CFG['pooling']['version'])
    return out,r,uc

def add_combined_checks(result,history):
    c=a.CFG['convergence'];row=result['points'][0];p=row['combined_20ps'];n=result['n_replicas']
    tail=[h['points'][0]['combined_20ps']['eta_mPa_s'] for h in history if h['n_replicas'] in (n-10,n-5)]
    vals=tail+[p['eta_mPa_s']]; change=(max(vals)-min(vals))/p['eta_mPa_s'] if len(vals)==3 else None
    check=result['checks'][0];flags=check['flags']
    flags.update(combined_precision=p['relative_std']<=c['max_relative_std'],
        combined_last_three_stability=change is not None and change<=c['max_last_three_range'],
        combined_fit_diagnostics=p['cost_z'] is not None and p['cost_z']<c['max_cost_z'] and p['criterion_z'] is not None and p['criterion_z']<c['max_criterion_z'])
    check['combined_last_three_relative_range']=change;check['passed']=all(flags.values())
    result['all_passed']=check['passed']
    previous=next((h for h in history if h['n_replicas']==n-5),None)
    result['pass_streak']=(1+(previous.get('pass_streak',0) if previous else 0)) if check['passed'] else 0
    result['stop_recommended']=result['pass_streak']>=c['required_consecutive_passes']
    result['review_required']=n>=c['review_at_or_above_replicas'] and flags['precision'] and flags['last_three_stability'] and not check['passed']
