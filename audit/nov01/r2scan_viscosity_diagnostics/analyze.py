"""Read-only audit of existing Ti r2SCAN trajectories; no MD or article edits."""
from pathlib import Path
import sys, os, json, hashlib, time, re
OUT=Path(__file__).resolve().parent; WORK=OUT.parents[2]
sys.path.insert(0,str(OUT/'deps'))
sys.path.insert(0,str(OUT/'source'))
os.environ['MPLCONFIGDIR']=str(OUT/'mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np, scipy, attrs
from scipy.stats import chi2
from stacie import compute_spectrum,estimate_acint,LorentzModel,ExpPolyModel,CV2LCriterion
from stacie.estimate import fit_model_spectrum
from stacie.utils import block_average
BASE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\results-scripts-logs')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def channels(p):
    xx,yy,zz,xy,xz,yz=p.T
    return np.array([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz])
def volume(p):
    text=p.read_text(); bounds=[list(map(float,m)) for m in re.findall(r'^\s*([-+.\deE]+)\s+([-+.\deE]+)\s+[xyz]lo [xyz]hi',text,re.M)]
    assert len(bounds)==3 and '512 atoms' in text
    atoms=text.split('Atoms # atomic')[1].split('Velocities')[0].strip().splitlines()
    assert len(atoms)==512 and all(a.split()[1]=='2' for a in atoms)
    return float(np.prod([b-a for a,b in bounds]))
def merge(ss):
    w=np.array([s.ndofs[0] for s in ss]);w=w/w.sum()
    return attrs.evolve(ss[0],mean=sum(a*s.mean for a,s in zip(w,ss)),variance=sum(a*s.variance for a,s in zip(w,ss)),amplitudes=sum(a*s.amplitudes for a,s in zip(w,ss)),ndofs=sum(s.ndofs for s in ss))
def pack(r):
    d=r.props if hasattr(r,'props') else r
    out={k:float(d[k]) for k in ['acint','acint_std','fcut','cost_zscore','criterion_zscore'] if k in d}
    out['eta']=out['acint']*1000;out['se']=out['acint_std']*1000;out['fcut_THz']=out['fcut']/1e12
    out['neff']=float(r.neff if hasattr(r,'neff') else d['neff'])
    if 'corrtime_exp' in d:out['tau_exp_ps']=float(d['corrtime_exp']*1e12)
    return out
def auto(s,model=None,**kw):return estimate_acint(s,model or LorentzModel(),verbose=False,rng=np.random.default_rng(42),**kw)
def fixed(s,fc):
    r=fit_model_spectrum(s,LorentzModel(),fc*1e12,8.,CV2LCriterion(),np.random.default_rng(42),100)
    assert 'acint' in r,r.get('msg')
    return pack(r)
def jk(a):
    a=np.array(a);return float(np.sqrt((len(a)-1)/len(a)*sum((a-a.mean())**2)))
def arrhenius(T,y,se):
    T=np.array(T);y=np.array(y);se=np.array(se);X=np.array([np.ones(len(T)),1000/T]).T;w=(y/se)**2
    cov=np.linalg.inv(X.T@(w[:,None]*X));b=cov@(X.T@(w*np.log(y)));fit=np.exp(X@b);z=(np.log(y)-X@b)*np.sqrt(w);q=sum(z*z)
    return {'ln_eta0':float(b[0]),'B_K':float(b[1]*1000),'Ea_eV':float(b[1]*1000*8.617333262e-5),'cov':cov.tolist(),'fitted':fit.tolist(),'z':z.tolist(),'chi2':float(q),'dof':len(T)-2,'p':float(chi2.sf(q,len(T)-2))}

def main():
    out={'created_unix':time.time(),'purpose':'Exploratory processing audit, article unchanged','dependencies':{'numpy':np.__version__,'scipy':scipy.__version__},'source_root':str(BASE),'temperatures':{},'inventory':{}}
    for d in sorted((BASE/'Ti-viscosity-r2SCAN').iterdir()):
        root=d/'scan'; case=root/'03_stress_autocorr/c_Ti_1.00'
        out['inventory'][d.name]={'pure_Ti_pressure_files':len(list(case.glob('visc*/pressure_*.data'))),'pure_Ti_sacf_files':len(list(case.glob('visc*/sacf_*.data'))),'md_params':read(root/'03_stress_autocorr/checkpoint.json').get('_meta',{}).get('params_snapshot',{})}
    published=read(WORK/'manuscript/supporting_data/viscosity_temperature_current.json')['temperature']['scan']
    roots={T:(BASE/f'Ti-viscosity-r2SCAN/scan_{T}K_wsl2/scan' if T in [1873,2000] else BASE/f'models_check_cluster/results/viscosity/dpa4/neo/finetune/AlTi/{T}K/scan') for T in [1873,1941,2000,2100]}
    grid=[.5,.65,.8,1.,1.25,1.5,1.75,2.,2.5,3.,4.]
    for T,root in roots.items():
        case=root/'03_stress_autocorr/c_Ti_1.00';files=sorted(case.glob('visc*/pressure_*.data'),key=lambda p:int(p.parent.name[4:]))
        seq=[];pref=[];manifest=[]
        for f in files:
            a=np.loadtxt(f);assert a.shape==(20001,7) and np.isfinite(a).all() and np.array_equal(a[:,0],np.arange(20001))
            m=f.parent/'mol.data';v=volume(m);seq.append(channels(a[:,1:]*1e5));pref.append(np.full(5,v*1e-30/(1.380649e-23*T)))
            manifest.append({'replica':int(f.parent.name[4:]),'pressure':str(f),'pressure_sha256':sha(f),'mol_sha256':sha(m),'volume_A3':v})
        assert len(set(m['pressure_sha256'] for m in manifest))==len(files)
        x=np.array(seq);p=np.array(pref);np.savez_compressed(OUT/f'input_{T}.npz',seq=x,pref=p)
        ss=[compute_spectrum(a,prefactors=f,timestep=1e-15) for a,f in zip(x,p)];s=merge(ss);r=auto(s);ref=next(a for a in published if a['T']==T)
        assert abs(r.acint*1000-ref['eta_mPa_s'])<2e-5,(T,r.acint*1000,ref)
        e={'n_replicas':len(x),'duration_ps':20.,'manifest':manifest,'published':ref,'auto':pack(r),'history':[{**pack(h),'weight':float(h.get('fcut_weight',0))} for h in r.history if 'acint' in h], 'scan':[fixed(s,fc) for fc in grid]}
        for stage in ['02_nvt_snapshots','03_stress_autocorr']:
            cp=root/stage/'checkpoint.json'
            if cp.exists():
                j=read(cp);e[stage]={'meta':j.get('_meta'),'case':j.get('cases',{}).get('c_Ti_1.00')}
        e['protocol_lines']=[l.strip() for l in (root/'log.lammps').read_text(errors='replace').splitlines() if any(v in l for v in ['pair_style deepmd','fix NVT','timestep '])] if (root/'log.lammps').exists() else []
        e['variants']={}
        for label,variant,kw in [('neff20',s,{'neff_min':20}),('neff30',s,{'neff_min':30}),('dense_grid',s,{'fcut_spacing':.25}),('noDC',s.without_zero_freq(),{})]:
            e['variants'][label]=pack(auto(variant,**kw))
        for b in [5,10,20]:
            sb=compute_spectrum(block_average(x.reshape(-1,x.shape[-1]),b),prefactors=p.ravel(),timestep=b*1e-15)
            e['variants'][f'block{b}']=pack(auto(sb))
        for label,sl in [('first10',slice(0,10000)),('second10',slice(10000,20000)),('discard1ps',slice(1000,None))]:
            e['variants'][label]=pack(auto(compute_spectrum(x[:,:,sl].reshape(-1,x[:,:,sl].shape[-1]),prefactors=p.ravel(),timestep=1e-15)))
        e['variants']['offdiag3']=pack(auto(compute_spectrum(x[:,2:,:].reshape(-1,x.shape[-1]),prefactors=p[:,2:].ravel(),timestep=1e-15)))
        e['channels']=[pack(auto(compute_spectrum(x[:,c,:],prefactors=p[:,c],timestep=1e-15))) for c in range(5)]
        e['batches5']=[pack(auto(merge(ss[i:i+5]))) for i in range(0,len(ss),5)]
        e['jackknife']={}
        for group_size in [1,5]:
            va=[];vf={str(fc):[] for fc in [1.,1.25,1.5,2.]}
            for start in range(0,len(ss),group_size):
                sub=merge([s0 for i,s0 in enumerate(ss) if i not in range(start,start+group_size)])
                va.append(pack(auto(sub))['eta'])
                for fc in vf:vf[fc].append(fixed(sub,float(fc))['eta'])
            e['jackknife'][str(group_size)]={'auto_se':jk(va),'auto_values':va,'fixed':{fc:{'se':jk(v),'values':v} for fc,v in vf.items()}}
        # Equal-time cross-replica stress correlations: diagnostic, not proof of independence.
        e['adjacent_stress_correlations']={}
        for label,sl in [('first1ps',slice(0,1000)),('last10ps',slice(10000,20000))]:
            cors=[]
            for i in range(len(x)-1):
                aa=x[i,:,sl].ravel();bb=x[i+1,:,sl].ravel();cors.append(float(np.corrcoef(aa,bb)[0,1]))
            e['adjacent_stress_correlations'][label]={'median':float(np.median(cors)),'max':float(max(cors)),'values':cors}
        np.savez_compressed(OUT/f'spectra_{T}.npz',freqs=s.freqs,amplitudes=s.amplitudes,individual=np.array([v.amplitudes for v in ss]))
        out['temperatures'][str(T)]=e;save('results.json',out)
        print('COMPLETED',T,json.dumps({'auto':e['auto'],'variants':e['variants'],'jk':{k:v['auto_se'] for k,v in e['jackknife'].items()}}),flush=True)
    es=list(out['temperatures'].values());temps=list(roots)
    out['arrhenius']=arrhenius(temps,[e['auto']['eta'] for e in es],[e['auto']['se'] for e in es])
    out['finished_unix']=time.time();save('results.json',out);print('ARRHENIUS',out['arrhenius'],flush=True)
if __name__=='__main__':main()
