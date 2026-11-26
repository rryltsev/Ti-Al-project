"""Raw E/F/stress metrics, paired test uncertainty, and parity plots for every checkpoint."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(Path(p).read_text())
cfg=read(OUT/'protocol.json');tasks=read(OUT/'tasks.json');reuse=read(OUT/'reuse.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pred={k:{} for k in cfg['models']}
for p in (HERE/'pbe_checkpoint_sweep').glob('result-*.json'):
    meta=read(p);key='PBE-'+meta['checkpoint'];assert meta['model_sha256']==cfg['models'][key]['sha256']
    src=np.load(p.with_suffix('.npz'));mapping={int(old):int(new) for new,old in reuse['PBE_inference']['mapping_new_to_old'].items()}
    for old in meta['indices']:
        if old in mapping:
            i=mapping[old];pred[key][i]={field:src[f'{field}_{old}'] for field in ['energy','forces','virial']}
for t in tasks:
    if t['kind']!='inference':continue
    folder=OUT/'results'/str(t['id'])
    if not (folder/'summary.json').exists():continue
    meta=read(folder/'summary.json');assert meta['status']=='passed' and meta['model_sha256']==cfg['models'][t['key']]['sha256']
    src=np.load(folder/'predictions.npz')
    for i in t['indices']:
        assert i not in pred[t['key']]
        pred[t['key']][i]={field:src[f'{field}_{i}'] for field in ['energy','forces','virial']}
rows={xc:read(OUT/d['rows_file']) for xc,d in cfg['datasets'].items()}
refs={xc:np.load(OUT/(xc+'-labels.npz')) for xc in cfg['datasets']}
allrows={};groups={};tests={};missing={}
def metric(rr):
    ee=np.array([r['error_E_meV_atom'] for r in rr]);out={'n':len(rr),'E_RMSE_meV_atom':float(np.sqrt(np.mean(ee**2))),
      'E_MAE_meV_atom':float(np.mean(abs(ee))),'E_bias_meV_atom':float(ee.mean()),
      'E_centered_RMSE_diagnostic_meV_atom':float(ee.std())}
    for prop,unit in [('F','eV_A'),('S','GPa')]:
        n=sum(r[prop+'_n'] for r in rr)
        out[prop+'_RMSE_'+unit]=float(np.sqrt(sum(r[prop+'_sse'] for r in rr)/n))
        out[prop+'_MAE_'+unit]=sum(r[prop+'_sae'] for r in rr)/n
    out['F_norm_MAE_eV_A']=sum(r['F_norm_sae'] for r in rr)/sum(r['n_atoms'] for r in rr)
    return out
for key,m in cfg['models'].items():
    xc=m['functional'];records=[];test=[]
    for r in rows[xc]:
        i=r['i']
        if i not in pred[key]:continue
        pr=pred[key][i];ref=refs[xc];n=r['n_atoms'];V=n*r['volume_A3_atom']
        e=float(pr['energy']-ref[f'energy_{i}'])/n*1000
        f=pr['forces']-ref[f'forces_{i}'];sv=-(pr['virial']-ref[f'virial_{i}'])/V*160.21766208
        sv=(sv+sv.T)/2;s=sv.flat[[0,4,8,5,2,1]]
        rec={**r,'error_E_meV_atom':e,'F_sse':float(np.sum(f*f)),'F_sae':float(np.sum(abs(f))),'F_n':f.size,
             'F_norm_sae':float(np.linalg.norm(f,axis=1).sum()),
             'S_sse':float(np.sum(s*s)),'S_sae':float(np.sum(abs(s))),'S_n':6}
        records.append(rec)
        if r['split']=='test':test.append({'id':r['id'],'x_Ti':r['x_Ti'],'V_A3_atom':r['volume_A3_atom'],
          'E_ref':float(ref[f'energy_{i}'])/n,'E_pred':float(pr['energy'])/n,
          'F_ref':ref[f'forces_{i}'].reshape(-1),'F_pred':pr['forces'].reshape(-1),
          'S_ref':(-ref[f'virial_{i}']/V*160.21766208).flat[[0,4,8,5,2,1]],
          'S_pred':(-pr['virial']/V*160.21766208).flat[[0,4,8,5,2,1]]})
    allrows[key]=records;groups[key]={};tests[key]=test
    for split in ['train','validation','test']:
        rr=[r for r in records if r['split']==split]
        if len(rr)==cfg['datasets'][xc]['counts'][split]:groups[key][split]=metric(rr)
    missing[key]=cfg['datasets'][xc]['n']-len(records)
# Paired bootstrap over configurations, conditional on the currently available test subset.
rng=np.random.default_rng(20261025);paired={}
for key,m in cfg['models'].items():
    base=next(k for k,v in cfg['models'].items() if v['functional']==m['functional'] and v['article_checkpoint'])
    aa=[r for r in allrows[key] if r['split']=='test'];bb=[r for r in allrows[base] if r['split']=='test']
    if len(aa)!=cfg['datasets'][m['functional']]['counts']['test'] or len(bb)!=len(aa):continue
    assert [r['id'] for r in aa]==[r['id'] for r in bb]
    ix=rng.integers(0,len(aa),size=(3000,len(aa)));res={}
    for field,label in [('error_E_meV_atom','E_RMSE_meV_atom'),('F_sse','F_RMSE_eV_A'),('S_sse','S_RMSE_GPa')]:
        a=np.array([r[field] for r in aa]);b=np.array([r[field] for r in bb])
        if field=='error_E_meV_atom':a=a*a;b=b*b;denom=np.ones(len(aa))
        else:denom=np.array([r[field[0]+'_n'] for r in aa])
        dd=np.sqrt(a[ix].sum(axis=1)/denom[ix].sum(axis=1))-np.sqrt(b[ix].sum(axis=1)/denom[ix].sum(axis=1))
        res[label]={'delta':groups[key]['test'][label]-groups[base]['test'][label],
                    'ci95':np.quantile(dd,[.025,.975]).tolist()}
    paired[key]={'baseline':base,'metrics':res}
pretrained=read(ROOT/'testset/monitor/results/n20-47a8b37210fc502b/metrics.json')
assert pretrained['common_ids']==read(OUT/'test_snapshot.json')['common_ids']
summary={'groups':groups,'paired_test_bootstrap_vs_article':paired,'missing_predictions':missing,
         'pretrained_metrics_common20':pretrained['pretrained_metrics'],
         'test_count':len(read(OUT/'test_snapshot.json')['records']),
         'limitations':'Raw energies; no fitted offsets or exclusions. Bootstrap conditional on current configurations, not proof of independent trajectory sampling. Test used diagnostically for checkpoint audit; fresh configurations required for a final blind assessment.'}
(DEST/'metrics.json').write_text(json.dumps(summary,indent=2));(DEST/'per_configuration_metrics.json').write_text(json.dumps(allrows,indent=2))
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
cols=['#2874a6','#20916f','#8e54ad','#d38c20','#bd4545'];xcs=['PBE','PBEsol','r2SCAN']
for prop,unit in [('E','eV/atom'),('F','eV/Å'),('S','GPa')]:
    fig,axs=plt.subplots(5,3,figsize=(10,14),layout='constrained')
    for j,xc in enumerate(xcs):
        keys=[k for k,m in cfg['models'].items() if m['functional']==xc]
        for row,key in enumerate(keys):
            ax=axs[row,j];tt=tests[key]
            if len(tt)!=summary['test_count']:ax.text(.5,.5,'Pending',ha='center');continue
            xx=np.concatenate([np.atleast_1d(t[prop+'_ref']) for t in tt]);yy=np.concatenate([np.atleast_1d(t[prop+'_pred']) for t in tt])
            ax.scatter(xx,yy,s=12 if prop=='E' else 1.4,alpha=.8 if prop=='E' else .22,color=cols[row],rasterized=True)
            low=min(xx.min(),yy.min());high=max(xx.max(),yy.max());pad=.04*(high-low)
            ax.plot([low-pad,high+pad],[low-pad,high+pad],'--',lw=.7,color='gray')
            field=prop+'_RMSE_'+{'E':'meV_atom','F':'eV_A','S':'GPa'}[prop]
            title=f"{xc}, {cfg['models'][key]['step']/1000:g}k; RMSE {groups[key]['test'][field]:.4g}"
            if prop=='E':title+=' meV/atom'
            ax.set(title=title,xlabel='DFT ('+unit+')',ylabel='Model ('+unit+')',xlim=(low-pad,high+pad),ylim=(low-pad,high+pad))
            ax.set_aspect('equal',adjustable='box')
    fig.suptitle(f'{prop} parity: {summary["test_count"]} identical test configurations; each model vs its own functional',fontsize=11)
    fig.savefig(DEST/('parity_'+prop+'.png'));fig.savefig(DEST/('parity_'+prop+'.pdf'));plt.close(fig)
fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
for j,xc in enumerate(xcs):
    keys=[k for k,m in cfg['models'].items() if m['functional']==xc]
    for i,(field,label) in enumerate([('E_RMSE_meV_atom','Energy RMSE (meV/atom)'),('F_RMSE_eV_A','Force RMSE (eV/Å)'),('S_RMSE_GPa','Stress RMSE (GPa)')]):
        for split,color,style in [('train','#2874a6','o-'),('validation','#ca6b2f','s-'),('test','#674c96','^-')]:
            kk=[k for k in keys if split in groups[k]]
            axs[i,j].plot([cfg['models'][k]['step']/1000 for k in kk],[groups[k][split][field] for k in kk],style,color=color,label=split,ms=4)
        axs[i,j].set(xlabel='Steps (thousands)',ylabel=label,title=xc if i==0 else None);axs[i,j].grid(alpha=.2)
        if i==0:axs[i,j].legend()
fig.suptitle('Full-set checkpoint errors (raw energies, Cartesian force components)',fontsize=12)
fig.savefig(DEST/'checkpoint_metrics.png');fig.savefig(DEST/'checkpoint_metrics.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
for j,xc in enumerate(xcs):
    for color,key in zip(cols,[k for k,m in cfg['models'].items() if m['functional']==xc]):
        tt=tests[key]
        axs[j].scatter([r['x_Ti'] for r in tt],[1000*(r['E_pred']-r['E_ref']) for r in tt],s=20,color=color,label=f"{cfg['models'][key]['step']/1000:g}k")
    axs[j].axhline(0,lw=.6,c='gray');axs[j].set(xlabel=r'$x_{Ti}$',ylabel='Energy residual (meV/atom)',title=xc);axs[j].legend(fontsize=7);axs[j].grid(alpha=.15)
fig.savefig(DEST/'test_energy_residuals.png');fig.savefig(DEST/'test_energy_residuals.pdf');plt.close(fig)
print(json.dumps({'missing':missing,'test':{k:g.get('test') for k,g in groups.items()}},indent=2))
