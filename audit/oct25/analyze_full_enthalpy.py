"""Reproducible production and equilibration-window sensitivity with block uncertainties."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_full_enthalpy_gpu';DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
from scipy.stats import t as student,linregress
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(OUT/'protocol.json');tasks=read(OUT/'tasks.json');base=read(OUT/'baseline_5k.json')
factor=96.4853321233
fields={'rho':'rho_g_cm3','H':'H_external_eV_atom','U':'Epot_eV_atom','T':'T_K','P':'P_GPa'}
units={'rho':'g/cm3','H':'kJ/mol','U':'kJ/mol','T':'K','P':'GPa'}
raw={'5000':{},'15000':{},'20000':{}}
def convert(rows):
    return {'time':np.array([r['time_ps'] for r in rows]),**{k:np.array([r[f] for r in rows])*(factor if k in ['H','U'] else 1) for k,f in fields.items()}}
for name,d in base.items():raw['5000'][name]={'x_Al':d['x_Al'],'production':convert(d['rows'])}
for task in tasks:
    step=str(task['checkpoint']);state=cfg['states'][task['state']];p=OUT/step/'results'/str(task['id']);meta=read(p/'summary.json')
    assert meta['status']=='completed' and meta['protocol_sha256']==sha(OUT/'protocol.json') and meta['model_sha256']==cfg['models'][step]['sha256']
    raw[step][state['composition']]={'x_Al':state['x_Al']}
    for stage in ['equilibration','production']:
        rows=[json.loads(s) for s in (p/(stage+'-thermo.jsonl')).read_text().splitlines()]
        assert len(rows)==3001 and rows[0]['step']==0 and rows[-1]['step']==30000
        assert [r['step'] for r in rows]==list(range(0,30001,10))
        raw[step][state['composition']][stage]=convert(rows)
def selected(d,stage,lo,hi):
    a=d[stage];mask=(a['time']>lo+1e-9)&(a['time']<=hi+1e-9)
    return {k:v[mask] for k,v in a.items()}
def acf_diag(a,dt):
    x=np.asarray(a)-np.mean(a);n=len(x);f=np.fft.rfft(x,n=2*n);cov=np.fft.irfft(f*f.conj())[:n]/np.arange(n,0,-1)
    if cov[0]==0:return {'tau_ps':0.,'effective_n':n}
    ac=cov/cov[0];summ=0.
    for i in range(1,min(n//2,2000)-1,2):
        pair=ac[i]+ac[i+1]
        if pair<0:break
        summ+=pair
    ineff=max(1.,1+2*summ)
    return {'tau_ps':float(.5*ineff*dt),'effective_n':float(n/ineff)}
def stats(segments,key):
    arrays=[s[key] for s in segments];a=np.concatenate(arrays);assert np.isfinite(a).all()
    mean=float(a.mean());checks={}
    for length in [1,2,3,5,6,10]:
        blocks=[]
        for seg in segments:
            dt=float(np.median(np.diff(seg['time'])));duration=len(seg['time'])*dt;n=int(np.floor((duration+1e-7)/length))
            if n<1:continue
            blocks.extend(np.array_split(seg[key],n))
        if len(blocks)<2:continue
        b=np.array([v.mean() for v in blocks]);w=np.array([len(v) for v in blocks],float);w/=w.sum();eff=1/np.sum(w*w)
        se=float(np.sqrt(np.sum(w*(b-np.sum(w*b))**2)/(1-np.sum(w*w))*np.sum(w*w)))
        checks[str(length)]={'sem':se,'df':float(eff-1),'blocks':len(b),'block_means':b.tolist()}
    candidates={k:v for k,v in checks.items() if k in ['3','5','6'] and v['blocks']>=4}
    if not candidates:candidates={k:v for k,v in checks.items() if k=='2' and v['blocks']>=4}
    chosen=max(candidates,key=lambda k:candidates[k]['sem']);s=candidates[chosen];ci=float(student.ppf(.975,s['df'])*s['sem'])
    return {'mean':mean,'sem':s['sem'],'df':s['df'],'ci95':[mean-ci,mean+ci],'selected_block_ps':int(chosen),'block_checks':checks,
      'n':len(a),'acf':acf_diag(a,float(np.median(np.diff(segments[-1]['time'])))) if len(segments)==1 else None}
def contrast(items):
    mean=sum(c*s['mean'] for c,s in items);var=sum(c*c*s['sem']**2 for c,s in items)
    den=sum((c*c*s['sem']**2)**2/s['df'] for c,s in items if s['df']>0)
    df=var*var/den if den else 1e12;se=var**.5;ci=float(student.ppf(.975,df)*se)
    return {'mean':float(mean),'sem':float(se),'df':float(df),'ci95':[float(mean-ci),float(mean+ci)],'z':float(mean/se) if se else 0.}
windows={'production':lambda d:[selected(d,'production',0,30)],'production_guarded':lambda d:[selected(d,'production',1,30)]}
for cut in [5,10,15,20]:windows['pooled_'+str(cut)]=lambda d,cut=cut:[selected(d,'equilibration',cut,30),selected(d,'production',1,30)]
results={}
for win,select in windows.items():
    groups={};mix={};changes={};comparison={}
    for step,states in raw.items():
        groups[step]={};mix[step]={}
        for name,d in states.items():
            segs=windows['production'](d) if step=='5000' and win.startswith('pooled_') else select(d)
            groups[step][name]={'x_Al':d['x_Al'],**{k:stats(segs,k) for k in fields}}
        g=groups[step]
        for name,d in g.items():
            x=d['x_Al'];mix[step][name]={'x_Al':x}
            for k in ['H','U']:
                mix[step][name][k]=contrast([(1,d[k]),(-(1-x),g['Ti'][k]),(-x,g['Al'][k])]) if 0<x<1 else {'mean':0.,'sem':0.,'df':1e12,'ci95':[0.,0.],'z':0.}
    for a,b in [('15000','5000'),('20000','5000'),('20000','15000')]:
        key=a+'-'+b;changes[key]={}
        for name,d in groups[a].items():
            x=d['x_Al'];changes[key][name]={'x_Al':x,'rho':contrast([(1,d['rho']),(-1,groups[b][name]['rho'])])}
            for k in ['H','U']:
                items=[]
                for step,sgn in [(a,1),(b,-1)]:
                    g=groups[step];items.extend([(sgn,g[name][k]),(-sgn*(1-x),g['Ti'][k]),(-sgn*x,g['Al'][k])])
                changes[key][name][k]=contrast(items) if 0<x<1 else {'mean':0.,'sem':0.,'df':1e12,'ci95':[0.,0.],'z':0.}
    results[win]={'groups':groups,'mixing':mix,'differences':changes}
qc={}
for step,states in raw.items():
    qc[step]={}
    for name,d in states.items():
        item={}
        for k in fields:
            first=stats([selected(d,'production',0,15)],k);last=stats([selected(d,'production',15,30)],k)
            item[k]={'production_second_minus_first':contrast([(1,last),(-1,first)])}
            if step!='5000':
                for cut in [5,10,15,20]:
                    early=stats([selected(d,'equilibration',cut,30)],k);late=results['production_guarded']['groups'][step][name][k]
                    item[k]['eq_'+str(cut)+'_minus_prod']=contrast([(1,early),(-1,late)])
                # Block-level trend within the late equilibration segment, kept separate from the restart.
                seg=selected(d,'equilibration',10,30);b=np.array_split(seg[k],6);ts=np.array_split(seg['time'],6)
                fit=linregress([v.mean() for v in ts],[v.mean() for v in b])
                item[k]['eq_10_30_slope_per_ps']={'slope':float(fit.slope),'stderr':float(fit.stderr),'p':float(fit.pvalue)}
        qc[step][name]=item
result={'status':'complete','windows':results,'qc':qc,'units':units,
 'uncertainty':'Maximum block SEM at 3/5/6 ps (>=4 blocks); 2 ps blocks only for short 10 ps QC segments. Approximate Welch-Satterthwaite pointwise 95% intervals. No correction for multiple comparisons. One trajectory per state/model; independent-run propagation approximation. Shared endpoints correlate mixing values across compositions.',
 'pooling':'Pooled windows combine equilibration after stated cut through 30 ps with production 1-30 ps. Blocks do not cross thermostat/velocity reset at 30 ps. Archived 5k has production only and is not extended.',
 'enthalpy':'H=(Epot+Ekin+P_external V)/N; legacy stress-sign expression is not used.',
 'protocol_sha256':sha(OUT/'protocol.json')}
(DEST/'statistics.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(DEST/'time_series.npz',**{step+'|'+name+'|'+stage+'|'+k:v for step,states in raw.items() for name,d in states.items() for stage in ['equilibration','production'] if stage in d for k,v in d[stage].items()})
for step in ['15000','20000']:
    print('XC PBE',step)
    for name,g in results['production']['groups'][step].items():
        m=results['production']['mixing'][step][name]['H'];delta=results['production']['differences'][step+'-5000'][name]['H']
        print(name,'rho',round(g['rho']['mean'],5),round(g['rho']['sem'],5),'Hmix',round(m['mean'],3),round(m['sem'],3),'deltaH',round(delta['mean'],3),[round(x,3) for x in delta['ci95']])
print('Saved full window/QC statistics')
