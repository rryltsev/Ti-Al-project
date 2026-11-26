"""Post-hoc diagnostic only: original production analysis and physics remain unchanged."""
from pathlib import Path
import os,sys,json,hashlib,time,itertools
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'Ti_2100K_N15_cutoff'
MID=HERE.parent/'Ti_2100K_N20_cutoff'
WORK=HERE.parents[4]
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
sys.path.insert(0,str(OLD/'source'))
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
ref=read(CAMPAIGN/'collected/analysis/Ti_2100K/batch_0025/summary.json')
windows={'40ps':slice(None),'first20ps':slice(0,20001),'second20ps':slice(20000,40001)}
seq=[];pref=[];thermo=[]
for i in range(1,26):
    source=(OLD/'source') if i<=15 else ((MID/'source') if i<=20 else (HERE/'source'))
    d=source/'states/Ti_2100K/replicas'/f'replica_{i:04d}'
    m=read(d/'summary.json');f=source/m['stress_file']
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
def spec(w,ids=range(25)):
    ids=list(ids)
    return compute_spectrum(np.concatenate(seq[ids],axis=0)[:,windows[w]],prefactors=np.concatenate(pref[ids]),timestep=1e-15)
def scalar(p):
    keys=['acint','acint_std','fcut','cost_zscore','criterion_zscore','neff','criterion','fcut_weight']
    return {k:float(p[k]) for k in keys if k in p and np.isfinite(p[k])}
def fixed(s,fc):
    p=fit_model_spectrum(s,LorentzModel(),fc*1e12,8.,CV2LCriterion(),np.random.default_rng(42),100)
    assert 'acint' in p,p.get('msg')
    return scalar(p)
out={'purpose':'N25 post-hoc common-cutoff follow-up only; not a production replacement','n':25,'thermo':thermo,'sources':ref['points'][0]['sources'],'automatic':{},'fixed_1p5':{},'caveats':['Windows and cumulative estimates are correlated.','Paired delete-one-replica jackknife keeps all five shear channels together.','Fixed cutoff is a sensitivity check; protocol and official estimates remain unchanged.']}
def save():
    (HERE/'diagnostic_results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
spectra={w:spec(w) for w in windows}
spectra['new_batch_40ps']=spec('40ps',range(20,25))
spectra['previous20_40ps']=spec('40ps',range(20))
for w,spectrum in spectra.items():
    fixed_result=fixed(spectrum,1.5);out['fixed_1p5'][w]=fixed_result
    if w!='previous20_40ps':
        result=estimate_acint(spectrum,LorentzModel(),verbose=False)
        expected=ref['points'][0]['new_batch_40ps'] if w=='new_batch_40ps' else ref['points'][0]['windows'][w]
        assert abs(result.acint*1000-expected['eta_mPa_s'])<2e-5
        hist=[scalar(x) for x in result.history]
        out['automatic'][w]={'summary':scalar(result.props),'history':hist,'weight_above5THz':sum(x.get('fcut_weight',0) for x in hist if x['fcut']>5e12)}
    save();print('FIT',w,fixed_result,flush=True)
diffs=[]
for omit in range(25):
    ids=[i for i in range(25) if i!=omit]
    vals={w:fixed(spec(w,ids),1.5) for w in ['first20ps','second20ps']}
    diffs.append(1000*(vals['first20ps']['acint']-vals['second20ps']['acint']))
    if (omit+1)%5==0:print('JACKKNIFE',omit+1,flush=True)
diffs=np.array(diffs);f=out['fixed_1p5']
out['paired_jackknife']={'cutoff_THz':1.5,'difference_mPa_s':1000*(f['first20ps']['acint']-f['second20ps']['acint']),'standard_error_mPa_s':float(np.sqrt(24/25*np.sum((diffs-diffs.mean())**2))),'leave_one_out_differences':diffs.tolist()}
a=f['new_batch_40ps'];b=f['previous20_40ps'];out['fixed_new_batch_z']=float(abs(a['acint']-b['acint'])/np.hypot(a['acint_std'],b['acint_std']))
out['finished_unix']=time.time();out['source_script_sha256']=sha(Path(__file__));save()
print('DONE',json.dumps({'paired':out['paired_jackknife'],'new_batch_z_at1p5':out['fixed_new_batch_z'],'weight_above5THz':{w:x['weight_above5THz'] for w,x in out['automatic'].items()}}),flush=True)
