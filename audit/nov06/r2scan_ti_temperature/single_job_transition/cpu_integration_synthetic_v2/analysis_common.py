"""Analyze immutable complete batches on CPUs while the next GPU batch runs."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stacie import compute_spectrum,estimate_acint,LorentzModel
from stacie.plot import plot_fitted_spectrum,plot_extras
from stacie.utils import UnitConfig
CFG=json.loads((ROOT/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def channels(p):
    """Input (samples,6) in Pa; five isotropic shear channels, no extra factor 1/5."""
    xx,yy,zz,xy,xz,yz=p.T
    return np.array([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz])

def prefactor(volume_A3,temperature_K):
    assert volume_A3>0 and temperature_K>0
    return volume_A3*1e-30/(1.380649e-23*temperature_K)

def fit(sequences,prefactors):
    spectrum=compute_spectrum(sequences,prefactors=np.asarray(prefactors),timestep=1e-15)
    uc=UnitConfig(acint_fmt='.4f',acint_symbol='РћВ·',acint_unit_str='mPa*s',acint_unit=1e-3,
                  freq_unit_str='THz',freq_unit=1e12,time_fmt='.3f',time_unit_str='ps',time_unit=1e-12)
    r=estimate_acint(spectrum,LorentzModel(),verbose=False,uc=uc,rng=np.random.default_rng(42))
    def prop(k):
        value=r.props.get(k)
        return float(value) if value is not None and np.isfinite(value) else None
    out=dict(eta_mPa_s=float(r.acint)*1000,eta_std_mPa_s=float(r.acint_std)*1000,
        relative_std=float(r.acint_std/r.acint),neff=float(r.neff),
        tau_exp_ps=float(r.corrtime_exp)*1e12,tau_int_ps=float(r.corrtime_int)*1e12,
        cost_z=prop('cost_zscore'),criterion_z=prop('criterion_zscore'))
    assert out['eta_mPa_s']>0 and out['eta_std_mPa_s']>0 and all(np.isfinite(v) for v in out.values() if v is not None)
    return out,r,uc

def convergence(rows,history,n):
    c=CFG['convergence'];row=rows[0];p=row['combined_20ps'];legacy=json.loads((ROOT/'verified_legacy_baseline.json').read_text())['baseline']
    vals=[legacy['eta_mPa_s']]+[h['points'][0]['combined_20ps']['eta_mPa_s'] for h in history]+[p['eta_mPa_s']]
    change=(max(vals[-3:])-min(vals[-3:]))/p['eta_mPa_s'] if len(vals)>=3 else None
    f=row['windows']['40ps'];first=row['windows']['first20ps'];second=row['windows']['second20ps']
    duration=max(abs(first['eta_mPa_s']/f['eta_mPa_s']-1),abs(second['eta_mPa_s']/f['eta_mPa_s']-1))
    flags=dict(combined_precision=p['relative_std']<=c['max_relative_std'],combined_last_three_stability=change is not None and change<=c['max_last_three_range'],combined_cost=p['cost_z'] is not None and p['cost_z']<c['max_cost_z'],combined_cutoff=p['criterion_z'] is not None and p['criterion_z']<c['max_criterion_z'],new_batch_consistency=row.get('new_batch_z') is not None and row['new_batch_z']<=c['max_new_batch_z'])
    passed=all(flags.values());previous=next((h for h in history if h['n_replicas']==n-5),None);streak=1+(previous.get('pass_streak',0) if previous else 0) if passed else 0
    return dict(checks=[dict(id=row['id'],flags=flags,passed=passed,last_three_relative_range=change,max_20_40_relative_difference=duration,new40_diagnostics={w:dict(relative_std=v['relative_std'],neff=v['neff'],cost_z=v['cost_z'],criterion_z=v['criterion_z']) for w,v in row['windows'].items()})],all_passed=passed,pass_streak=streak,stop_recommended=False,review_required=False,assessment_pending=True,legacy_replicas=CFG['legacy']['n_replicas'],total_replica_clusters=CFG['legacy']['n_replicas']+n,sampling_extension='Only the explicit versioned plan determines additional blocks; no mandatory Nnew quota.')
