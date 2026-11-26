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

low={'cutoff_THz':.8,'fixed':{w:fixed(spec(w),.8) for w in windows},'leave_one_replica_out':[]}
for omit in range(10):
    ids=[i for i in range(10) if i!=omit]
    row={'omitted_replica':omit+1,'windows':{w:fixed(spec(w,ids),.8) for w in ['first20ps','second20ps']}}
    low['leave_one_replica_out'].append(row)
diffs=np.array([(r['windows']['first20ps']['acint']-r['windows']['second20ps']['acint'])*1000 for r in low['leave_one_replica_out']])
low['paired_jackknife']={'difference_mPa_s':1000*(low['fixed']['first20ps']['acint']-low['fixed']['second20ps']['acint']),'standard_error_mPa_s':float(np.sqrt(9/10*np.sum((diffs-diffs.mean())**2))),'leave_one_out_differences':diffs.tolist()}
print('LOW_CUTOFF_JACKKNIFE',json.dumps(low['paired_jackknife']),flush=True)
individual={w:[spec(w,[i]) for i in range(10)] for w in ['first20ps','second20ps']}
def mix(items):
    return attrs.evolve(items[0],mean=np.mean([s.mean for s in items]),variance=np.mean([s.variance for s in items]),amplitudes=np.mean([s.amplitudes for s in items],axis=0),ndofs=np.sum([s.ndofs for s in items],axis=0))
swaps=[]
for pattern in itertools.product([0,1],repeat=9):
    bits=(0,)+pattern
    a=[individual['second20ps' if b else 'first20ps'][i] for i,b in enumerate(bits)]
    b=[individual['first20ps' if v else 'second20ps'][i] for i,v in enumerate(bits)]
    delta=1000*(fixed(mix(a),.8)['acint']-fixed(mix(b),.8)['acint'])
    swaps.append({'swaps':list(bits),'delta_mPa_s':delta})
    swaps.append({'swaps':[1-x for x in bits],'delta_mPa_s':-delta})
    if len(swaps)%256==0:print('LOW_CUTOFF_SWAPS',len(swaps),'/1024',flush=True)
obs=swaps[0]['delta_mPa_s']
low['paired_swap_test']={'observed_delta_mPa_s':obs,'two_sided_p':sum(abs(x['delta_mPa_s'])>=abs(obs)-1e-7 for x in swaps)/1024,'all1024':swaps,'assumption':'Window exchangeability within each replica under a stationary null; exploratory test with ten pairs.'}
low['interpretation']='Post-hoc low-cutoff diagnostic. Passing cost/cutoff does not establish validity: effective spectral points are about17 for20ps and33 for40ps, below60.'
(HERE/'low_cutoff_results.json').write_text(json.dumps(low,indent=2)+'\n',encoding='utf-8')
print('LOW_DONE',low['paired_swap_test']['two_sided_p'],flush=True)
