"""Post-hoc diagnostic only: original production analysis and physics remain unchanged."""
from pathlib import Path
import os,sys,json,hashlib,time,itertools
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[4]
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
sys.path.insert(0,str(HERE/'source'))
os.environ['MPLCONFIGDIR']=str(HERE/'mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import attrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stacie import compute_spectrum,estimate_acint,LorentzModel,CV2LCriterion
from stacie.estimate import fit_model_spectrum
CAMPAIGN=HERE.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=read(CAMPAIGN/'collected/analysis/Ti_2100K/batch_0010/summary.json')
windows={'40ps':slice(None),'first20ps':slice(0,20001),'second20ps':slice(20000,40001)}
seq=[];pref=[];thermo=[]
for i in range(1,11):
    d=HERE/'source/states/Ti_2100K/replicas'/f'replica_{i:04d}'
    m=read(d/'summary.json');f=HERE/'source'/m['stress_file']
    assert sha(f)==m['stress_sha256']==ref['points'][0]['sources'][i-1]['sha256']
    assert m['protocol_sha256']==ref['protocol_sha256'] and m['checkpoint_sha256']==ref['checkpoint_sha256']
    with np.load(f) as a:
        p=a['pressure_bar']*1e5;v=float(a['volume_A3'])
        assert p.shape==(40001,6) and np.isfinite(p).all()
        assert float(a['dt_ps'])==.001
    xx,yy,zz,xy,xz,yz=p.T
    seq.append(np.array([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz]))
    pref.append(np.full(5,v*1e-30/(1.380649e-23*2100)))
    blocks=[];block=[];header=None
    for line in (f.parent/'log.lammps').read_text(errors='replace').splitlines():
        parts=line.split()
        if parts and parts[0]=='Step' and 'Temp' in parts:
            if block:blocks.append((header,block));block=[]
            header=parts
        elif header:
            try:
                vals=[float(z) for z in parts]
                if len(vals)==len(header):block.append(vals)
            except ValueError:
                if block:blocks.append((header,block));block=[]
                header=None
    if block:blocks.append((header,block))
    h,b=max(blocks,key=lambda x:len(x[1]));b=np.array(b)
    means={}
    for label,mask in [('first20ps',b[:,0]<=20000),('second20ps',b[:,0]>=20000)]:
        means[label]={k:float(b[mask,h.index(k)].mean()) for k in ['Temp','PotEng','Press','Volume']}
    thermo.append({'replica':i,'volume_A3':v,'thermo_rows':len(b),'window_means':means})
seq=np.array(seq);pref=np.array(pref)
def spec(w,ids=range(10)):
    ids=list(ids)
    return compute_spectrum(np.concatenate(seq[ids],axis=0)[:,windows[w]],prefactors=np.concatenate(pref[ids]),timestep=1e-15)
def scalar(p):
    keys=['acint','acint_std','fcut','cost_zscore','criterion_zscore','neff','criterion','fcut_weight']
    return {k:float(p[k]) for k in keys if k in p and np.isfinite(p[k])}
def fixed(s,fc):
    p=fit_model_spectrum(s,LorentzModel(),fc*1e12,8.,CV2LCriterion(),np.random.default_rng(42),100)
    assert 'acint' in p,p.get('msg')
    return scalar(p)
out={'purpose':'Post-hoc sensitivity and paired-replica diagnostic; not production replacement','n':10,'thermo':thermo,'sources':ref['points'][0]['sources'],'automatic':{},'fixed':{},'leave_one_replica_out':[],'caveats':['Only ten replica pairs; resampling is exploratory.','Same-replica windows and stress channels are kept together.','Fixed cutoffs are sensitivity tests, not selected final viscosity estimates.']}
def save():
    (HERE/'diagnostic_results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
spectra={w:spec(w) for w in windows}
for w,s in spectra.items():
    r=estimate_acint(s,LorentzModel(),verbose=False)
    out['automatic'][w]={'summary':scalar(r.props),'history':[scalar(p) for p in r.history]}
    assert abs(r.acint*1000-ref['points'][0]['windows'][w]['eta_mPa_s'])<2e-5
    print('REPRODUCED',w,scalar(r.props),flush=True);save()
fc_grid=[.8,1.,1.2,1.5,1.6,1.8,2.,2.2,2.5,3.,4.,5.,6.,7.,8.]
for w,s in spectra.items():
    out['fixed'][w]=[fixed(s,fc) for fc in fc_grid]
    print('FIXED_SCAN',w,flush=True);save()
for omit in range(10):
    row={'omitted_replica':omit+1,'automatic':{},'fixed':{}}
    for w in ['first20ps','second20ps']:
        s=spec(w,[i for i in range(10) if i!=omit])
        row['automatic'][w]=scalar(estimate_acint(s,LorentzModel(),verbose=False).props)
        row['fixed'][w]={str(fc):fixed(s,fc) for fc in [1.5,2.]}
    out['leave_one_replica_out'].append(row);save()
    print('LEAVE_ONE_OUT',omit+1,{w:[p['acint']*1000,p['fcut']/1e12] for w,p in row['automatic'].items()},flush=True)
# Exact paired label swaps at a common diagnostic cutoff. Each replica keeps all five channels.
individual={w:[spec(w,[i]) for i in range(10)] for w in ['first20ps','second20ps']}
def mix(items):
    return attrs.evolve(items[0],mean=np.mean([s.mean for s in items]),variance=np.mean([s.variance for s in items]),amplitudes=np.mean([s.amplitudes for s in items],axis=0),ndofs=np.sum([s.ndofs for s in items],axis=0))
swaps=[]
for pattern in itertools.product([0,1],repeat=9):
    bits=(0,)+pattern
    a=[individual['second20ps' if b else 'first20ps'][i] for i,b in enumerate(bits)]
    b=[individual['first20ps' if v else 'second20ps'][i] for i,v in enumerate(bits)]
    delta=1000*(fixed(mix(a),1.5)['acint']-fixed(mix(b),1.5)['acint'])
    swaps.append({'swaps':list(bits),'delta_mPa_s':delta})
    swaps.append({'swaps':[1-x for x in bits],'delta_mPa_s':-delta})
    if len(swaps)%128==0:print('PAIRED_SWAPS',len(swaps),'/1024',flush=True)
obs=swaps[0]['delta_mPa_s']
out['paired_swap_test']={'cutoff_THz':1.5,'observed_delta_mPa_s':obs,'two_sided_p':sum(abs(x['delta_mPa_s'])>=abs(obs)-1e-7 for x in swaps)/1024,'all1024':swaps,'assumption':'Exchangeability of the two windows within a replica under the stationary null; only ten pairs and no guarantee of independence between reservoir-spaced replicas.'}
out['paired_jackknife']={}
for fc in [1.5,2.]:
    diffs=np.array([(r['fixed']['first20ps'][str(fc)]['acint']-r['fixed']['second20ps'][str(fc)]['acint'])*1000 for r in out['leave_one_replica_out']])
    full=next(r['acint'] for r in out['fixed']['first20ps'] if abs(r['fcut']/1e12-fc)<1e-9)-next(r['acint'] for r in out['fixed']['second20ps'] if abs(r['fcut']/1e12-fc)<1e-9)
    out['paired_jackknife'][str(fc)]={'difference_mPa_s':full*1000,'standard_error_mPa_s':float(np.sqrt(9/10*np.sum((diffs-diffs.mean())**2))),'leave_one_out_differences':diffs.tolist()}
out['finished_unix']=time.time();save()
colors={'40ps':'#2878B5','first20ps':'#D55E00','second20ps':'#39835B'}
labels={'40ps':'40 ps','first20ps':'First 20 ps','second20ps':'Second 20 ps'}
fig,axs=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
for w in windows:
    hist=out['automatic'][w]['history'];x=np.array([r['fcut']/1e12 for r in hist]);y=np.array([r['acint']*1000 for r in hist]);e=np.array([r['acint_std']*1000 for r in hist]);c=colors[w]
    axs[0].plot(x,y,color=c,label=labels[w]);axs[0].fill_between(x,y-e,y+e,color=c,alpha=.12)
    axs[1].plot(x,[r['fcut_weight'] for r in hist],color=c,label=labels[w])
    axs[2].plot(x,[r['criterion_zscore'] for r in hist],color=c,label=labels[w])
    a=out['automatic'][w]['summary'];axs[0].errorbar(a['fcut']/1e12,a['acint']*1000,yerr=a['acint_std']*1000,fmt='o',color=c,capsize=3)
for ax in axs:
    ax.set_xscale('log');ax.set_xlim(.65,10.5);ax.set_xlabel('Cutoff frequency (THz)');ax.grid(alpha=.2)
axs[0].set(ylabel='Viscosity (mPa s)',title='Same data, different cutoff',ylim=(1.8,3.7));axs[0].legend(fontsize=8)
axs[1].set(ylabel='Automatic cutoff weight',title='Where automatic fitting puts weight')
axs[2].axhline(2,color='black',ls='--',label='Threshold 2');axs[2].set(ylabel='Cutoff criterion Z',title='Cutoff diagnostic',ylim=(-1.5,10));axs[2].legend(fontsize=8)
fig.suptitle('Pure Ti, PBE 15k, 2100 K: ten replicas | bands: ±1 standard error')
fig.savefig(HERE/'cutoff_diagnostic.png',dpi=180);fig.savefig(HERE/'cutoff_diagnostic.pdf');plt.close(fig)
print('DONE',json.dumps(out['paired_jackknife']),out['paired_swap_test']['two_sided_p'],flush=True)
