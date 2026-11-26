"""Temperature-specific STACIE analysis; verified legacy and new results kept traceable."""
from pathlib import Path
import sys,json,time,traceback
import analysis_common as a
import pooled_analysis as pool
import numpy as np
import matplotlib.pyplot as plt
from stacie.plot import plot_fitted_spectrum,plot_extras
ROOT=a.ROOT;CFG=a.CFG

def write_fit(folder,seq,pref):
    folder.mkdir(parents=True,exist_ok=True)
    out,r,uc=a.fit(seq,pref);a.save(folder/'result.json',out)
    fig,ax=plt.subplots();plot_fitted_spectrum(ax,uc,r);fig.savefig(folder/'spectrum.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8));plot_extras(axs,uc,r);fig.savefig(folder/'extras.png',dpi=130);plt.close(fig)
    return out

def main(sid,n):
    assert n>=5 and n%5==0
    s=next(s for s in CFG['states'] if s['id']==sid);T=s['temperature_K']
    dest=ROOT/f'analysis/{sid}/batch_{n:04d}'
    if (dest/'summary.json').exists():return
    seqs=[];pref=[];sources=[]
    for i in range(1,n+1):
        m=json.loads((ROOT/f'states/{sid}/replicas/replica_{i:04d}/summary.json').read_text())
        data=ROOT/m['stress_file']
        assert m['protocol_sha256']==a.sha(ROOT/'protocol.json') and m['checkpoint_sha256']==CFG['checkpoint']['sha256']
        assert m['temperature_K']==T and a.sha(data)==m['stress_sha256']
        with np.load(data) as d:
            pressure=d['pressure_bar'];vol=float(d['volume_A3'])
            assert pressure.shape==(40001,6) and float(d['dt_ps'])==.001
            assert abs(vol/m['volume_A3']-1)<1e-12
            seqs.append(a.channels(pressure*1e5));pref.extend([a.prefactor(vol,T)]*5)
        sources.append(dict(replica=i,stress_file=m['stress_file'],sha256=m['stress_sha256']))
    seq=np.concatenate(seqs,axis=0);pref=np.array(pref)
    row=dict(id=sid,temperature_K=T,x_Al=0.,n_replicas=n,windows={},sources=sources)
    for window,sl in [('40ps',slice(None)),('first20ps',slice(0,20001)),('second20ps',slice(20000,40001))]:
        row['windows'][window]=write_fit(dest/window,seq[:,sl],pref)
    clusters,legacy=pool.legacy_clusters()
    for rep_seq,rep_pref in zip(seqs,np.asarray(pref).reshape(n,5)):
        clusters.append(pool.new_cluster(rep_seq,rep_pref))
    combined,r,uc=pool.combine(clusters,CFG['legacy']['n_replicas'],n)
    row['combined_20ps']=combined
    folder=dest/'combined_20ps';folder.mkdir(parents=True,exist_ok=True)
    a.save(folder/'result.json',combined)
    fig,ax=plt.subplots();plot_fitted_spectrum(ax,uc,r);fig.savefig(folder/'spectrum.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8));plot_extras(axs,uc,r);fig.savefig(folder/'extras.png',dpi=130);plt.close(fig)
    row['legacy_manifest_sha256']=CFG['legacy']['sha256']
    history=[json.loads(p.read_text()) for p in sorted((ROOT/f'analysis/{sid}').glob('batch_*/summary.json'))]
    history=[h for h in history if h['n_replicas']<n]
    result=dict(temperature_K=T,n_replicas=n,points=[row],unix=time.time(),protocol_sha256=a.sha(ROOT/'protocol.json'),checkpoint_sha256=CFG['checkpoint']['sha256'])
    previous=next((h['points'][0]['combined_20ps'] for h in reversed(history) if h['n_replicas']==n-5),None)
    if previous is None:previous=json.loads((ROOT/'verified_legacy_baseline.json').read_text())['baseline']
    newest,rnew,ucnew=pool.combine(clusters[-5:],0,5)
    row.update(new_batch_40ps=row['windows']['40ps'] if n==5 else write_fit(dest/'new_batch_40ps',seq[-25:],pref[-25:]),new_batch_20ps=newest,previous_accumulated_20ps=previous,
        new_batch_z=float(abs(newest['eta_mPa_s']-previous['eta_mPa_s'])/np.hypot(newest['eta_std_mPa_s'],previous['eta_std_mPa_s'])))
    if n==5:
        import shutil
        shutil.copytree(dest/'40ps',dest/'new_batch_40ps',dirs_exist_ok=True)
    folder=dest/'new_batch_20ps';folder.mkdir(parents=True,exist_ok=True);a.save(folder/'result.json',newest)
    fig,ax=plt.subplots();plot_fitted_spectrum(ax,ucnew,rnew);fig.savefig(folder/'spectrum.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8));plot_extras(axs,ucnew,rnew);fig.savefig(folder/'extras.png',dpi=130);plt.close(fig)
    result.update(a.convergence([row],history,n))
    result['caveats']=['Cumulative and20/40ps estimates are correlated.',
        'Combined20ps includes legacy20ps and both halves of each new40ps; both halves are one uncertainty cluster.',
        'AdoptedSE=max(formalSE,whole-replica jackknifeSE); unequal cluster lengths and reservoir dependence limit its interpretation.',
        'Legacy and new preparation differ; no PBE trajectories are pooled.',
        'Finite refinement plan controls new blocks; diagnostics do not impose a new30-replica quota; errors exclude model/protocol bias.']
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    hist=history+[result];ns=[h['n_replicas'] for h in hist]
    for w,color in [('40ps','#2878B5'),('first20ps','#D55E00'),('second20ps','#39835B')]:
        rr=[h['points'][0]['windows'][w] for h in hist]
        axs[0].errorbar(ns,[x['eta_mPa_s'] for x in rr],yerr=[x['eta_std_mPa_s'] for x in rr],fmt='o-',capsize=3,color=color,label=w)
    axs[0].set(xlabel='New replicas',ylabel='Viscosity (mPa s)',title=f'r2SCAN, pure Ti, {T} K');axs[0].legend()
    axs[1].plot(ns,[100*h['points'][0]['windows']['40ps']['relative_std'] for h in hist],'o-',label='Relative error (40ps)')
    axs[1].plot(ns,[100*h['checks'][0]['max_20_40_relative_difference'] for h in hist],'s-',label='Max20/40ps difference')
    axs[1].axhline(3,ls='--',color='.5');axs[1].set(xlabel='New replicas',ylabel='Percent');axs[1].legend()
    fig2,ax=plt.subplots(figsize=(6,4),layout='constrained')
    pp=[json.loads((ROOT/'verified_legacy_baseline.json').read_text())['baseline']]+[h['points'][0]['combined_20ps'] for h in hist]
    ax.errorbar([0]+ns,[p['eta_mPa_s'] for p in pp],yerr=[p['eta_std_mPa_s'] for p in pp],fmt='o-',capsize=3,label='Legacy20ps + new40ps')
    ax.set(xlabel='New40ps replicas',ylabel='Viscosity (mPa s)',title=f'r2SCAN pooled refinement, {T} K');ax.legend()
    fig2.savefig(dest/'combined_convergence.png',dpi=180);plt.close(fig2)
    fig.savefig(dest/'convergence.png' ,dpi=180);plt.close(fig)
    a.save(dest/'summary.json',result)
    a.save(ROOT/f'analysis/latest_{sid}.json',result)
    index={'points':{},'unix':time.time()}
    for st in CFG['states']:
        path=ROOT/f"analysis/latest_{st['id']}.json"
        if path.exists():index['points'][st['id']]=json.loads(path.read_text())
    a.save(ROOT/'analysis/latest.json',index)
    print('POINT_ANALYZED',T,n,'stop',result['stop_recommended'],'review',result['review_required'],flush=True)

if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]))
    except Exception:
        a.save(ROOT/'analysis/failure.json',dict(args=sys.argv[1:],traceback=traceback.format_exc(),unix=time.time()))
        raise
