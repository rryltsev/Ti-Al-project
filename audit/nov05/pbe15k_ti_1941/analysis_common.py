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
    uc=UnitConfig(acint_fmt='.4f',acint_symbol='η',acint_unit_str='mPa*s',acint_unit=1e-3,
                  freq_unit_str='THz',freq_unit=1e12,time_fmt='.3f',time_unit_str='ps',time_unit=1e-12)
    r=estimate_acint(spectrum,LorentzModel(),verbose=False,uc=uc)
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
    c=CFG['convergence'];checks=[]
    for row in rows:
        f=row['windows']['40ps'];a=row['windows']['first20ps'];b=row['windows']['second20ps']
        tail=[h for h in history if h['n_replicas']==n-10 or h['n_replicas']==n-5]
        vals=[next(p for p in h['points'] if p['id']==row['id'])['windows']['40ps']['eta_mPa_s'] for h in tail]+[f['eta_mPa_s']]
        change=(max(vals)-min(vals))/f['eta_mPa_s'] if len(vals)==3 else None
        duration=max(abs(a['eta_mPa_s']/f['eta_mPa_s']-1),abs(b['eta_mPa_s']/f['eta_mPa_s']-1))
        flags=dict(minimum_replicas=n>=c['min_replicas'],precision=f['relative_std']<=c['max_relative_std'],
            last_three_stability=change is not None and change<=c['max_last_three_range'],
            duration_stability=duration<=c['max_20_40_relative_difference'],
            spectral_points=f['neff']>=c['min_neff'],
            cost=f['cost_z'] is not None and f['cost_z']<c['max_cost_z'],
            cutoff=f['criterion_z'] is not None and f['criterion_z']<c['max_criterion_z'],
            new_batch_consistency=row.get('new_batch_z') is not None and row['new_batch_z']<=c['max_new_batch_z'])
        diagnostics=list(row['windows'].values())+[row.get('new_batch_40ps',{})]
        flags['window_fit_diagnostics']=all(v.get('cost_z') is not None and v['cost_z']<c['max_cost_z'] and v.get('criterion_z') is not None and v['criterion_z']<c['max_criterion_z'] for v in diagnostics)
        checks.append(dict(id=row['id'],flags=flags,passed=all(flags.values()),
                           last_three_relative_range=change,max_20_40_relative_difference=duration))
    passed=all(r['passed'] for r in checks)
    previous=next((h for h in history if h['n_replicas']==n-5),None)
    streak=1+(previous.get('pass_streak',0) if previous else 0) if passed else 0
    return dict(checks=checks,all_passed=passed,pass_streak=streak,
                stop_recommended=streak>=c['required_consecutive_passes'])

