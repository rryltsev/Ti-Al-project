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
            new_batch_consistency=row.get('new_batch_z') is not None and row['new_batch_z']<=2.5)
        checks.append(dict(id=row['id'],flags=flags,passed=all(flags.values()),
                           last_three_relative_range=change,max_20_40_relative_difference=duration))
    passed=all(r['passed'] for r in checks)
    previous=next((h for h in history if h['n_replicas']==n-5),None)
    streak=1+(previous.get('pass_streak',0) if previous else 0) if passed else 0
    return dict(checks=checks,all_passed=passed,pass_streak=streak,
                stop_recommended=streak>=c['required_consecutive_passes'])

def plot_history(history,dest):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    baseline=json.loads((ROOT/'baseline_5k.json').read_text())
    for col,s in enumerate(CFG['states']):
        points=[next(r for r in h['points'] if r['id']==s['id']) for h in history]
        ns=[h['n_replicas'] for h in history]
        ref=next(r for r in baseline if r['c_Ti']==s['c_Ti'])
        a,b=axs[:,col]
        a.axhspan(ref['eta_mPa_s']-ref['eta_std_mPa_s'],ref['eta_mPa_s']+ref['eta_std_mPa_s'],color='.5',alpha=.16)
        a.axhline(ref['eta_mPa_s'],color='.35',ls='--',label='5k: 50 replicas, 20 ps')
        for window,color,label,offset in [('40ps','#c45615','15k: 40 ps',.10),('first20ps','#2375ae','15k: first 20 ps',-.10)]:
            a.errorbar(np.array(ns)+offset,[p['windows'][window]['eta_mPa_s'] for p in points],
                yerr=[p['windows'][window]['eta_std_mPa_s'] for p in points],fmt='o-',color=color,capsize=3,ms=4,label=label)
        a.set(title=f"x(Al) = {s['x_Al']:.2f} | 1873 K",ylabel='Viscosity (mPa s)',xlabel='Replicas per composition')
        a.legend(fontsize=8,frameon=False)
        b.plot(ns,[100*p['windows']['40ps']['relative_std'] for p in points],'o-',color='#c45615',label='Relative 1σ (40 ps)')
        b.plot(ns,[100*max(abs(p['windows'][w]['eta_mPa_s']/p['windows']['40ps']['eta_mPa_s']-1) for w in ['first20ps','second20ps']) for p in points],
            's-',color='#2375ae',label='Max. difference: 20 vs 40 ps')
        b.axhline(3,color='.5',ls='--',lw=1)
        b.set(ylabel='Percent',xlabel='Replicas per composition',ylim=(0,None))
        b.legend(fontsize=8,frameon=False)
        for ax in [a,b]:ax.grid(alpha=.18)
    fig.suptitle(f"PBE 15k — cumulative viscosity, N = {history[-1]['n_replicas']} per composition\nError bars: STACIE 1σ; sequential convergence diagnostics",fontsize=13)
    fig.savefig(dest/'convergence.png',dpi=180);fig.savefig(dest/'convergence.svg');plt.close(fig)

def main(n):
    assert n>=5 and n%5==0
    dest=ROOT/f'analysis/batch_{n:04d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return
    points=[]
    for s in CFG['states']:
        allseq=[];pref=[];sources=[]
        for i in range(1,n+1):
            p=ROOT/'states'/s['id']/'replicas'/f'replica_{i:04d}'/'summary.json'
            m=json.loads(p.read_text());data=ROOT/m['stress_file']
            assert m['protocol_sha256']==sha(ROOT/'protocol.json') and m['checkpoint_sha256']==CFG['checkpoint']['sha256']
            assert sha(data)==m['stress_sha256']
            with np.load(data) as d:
                pressure=d['pressure_bar'];vol=float(d['volume_A3'])
                assert pressure.shape==(40001,6) and float(d['dt_ps'])==.001
                assert abs(vol/m['volume_A3']-1)<1e-12
                allseq.append(channels(pressure*1e5));pref.extend([vol*1e-30/(1.380649e-23*1873)]*5)
            sources.append(dict(replica=i,stress_file=m['stress_file'],sha256=m['stress_sha256']))
        seq=np.concatenate(allseq,axis=0);pref=np.array(pref)
        row=dict(id=s['id'],x_Al=s['x_Al'],actual_x_Al=s['actual_x_Al'],n_replicas=n,windows={},sources=sources)
        for window,sl in [('40ps',slice(None)),('first20ps',slice(0,20001)),('second20ps',slice(20000,40001))]:
            out,r,uc=fit(seq[:,sl],pref);row['windows'][window]=out
            folder=dest/s['id']/window;folder.mkdir(parents=True,exist_ok=True)
            save(folder/'result.json',out)
            fig,ax=plt.subplots();plot_fitted_spectrum(ax,uc,r);fig.savefig(folder/'spectrum.png',dpi=120);plt.close(fig)
            fig,axs=plt.subplots(2,2,figsize=(10,8));plot_extras(axs,uc,r);fig.savefig(folder/'extras.png',dpi=120);plt.close(fig)
        if n>=10:
            new,_,_=fit(seq[-25:],pref[-25:]);old,_,_=fit(seq[:-25],pref[:-25])
            row['new_batch_40ps']=new;row['previous_replicas_40ps']=old
            row['new_batch_z']=float(abs(new['eta_mPa_s']-old['eta_mPa_s'])/np.hypot(new['eta_std_mPa_s'],old['eta_std_mPa_s']))
        points.append(row)
        print(s['id'],json.dumps(row['windows']),flush=True)
    history=[json.loads(p.read_text()) for p in sorted((ROOT/'analysis').glob('batch_*/summary.json'))]
    history=[h for h in history if h['n_replicas']<n]
    data=dict(n_replicas=n,temperature_K=1873,points=points,unix=time.time(),
        protocol_sha256=sha(ROOT/'protocol.json'),checkpoint_sha256=CFG['checkpoint']['sha256'],
        caveats=['Cumulative estimates share trajectories; successive error bars are correlated.',
            '20 ps windows overlap the 40 ps estimate. Duration ratios are diagnostic, not independent-sample tests.',
            'STACIE errors do not include finite-size, thermostat, preparation, or model errors.',
            'Optional sequential stopping: nominal intervals are descriptive.',
            'NPT reservoir snapshots are 5 ps apart; complete statistical independence is not proven.'])
    data.update(convergence(points,history,n))
    plot_history(history+[data],dest)
    save(dest/'summary.json',data)
    save(ROOT/'analysis/latest.json',dict(data,plot=str((dest/'convergence.png').relative_to(ROOT))))
    print('BATCH_ANALYZED',n,'stop_recommended',data['stop_recommended'],flush=True)

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception:
        save(ROOT/'analysis/failure.json',dict(batch=sys.argv[1:],traceback=traceback.format_exc(),unix=time.time()))
        raise
