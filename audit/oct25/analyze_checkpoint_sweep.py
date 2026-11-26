"""Audit all five CPU checkpoints and quantify train/validation/holdout evolution."""
from pathlib import Path
import sys,json,hashlib,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_sweep'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'));os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(OUT/'input-manifest.json');cfg=read(OUT/'remote-input.json');rows=read(OUT/'rows.json')
assert sha(OUT/'labels-local-only.npz')==m['labels_sha256'] and sha(OUT/'rows.json')==m['rows_sha256']
refs=np.load(OUT/'labels-local-only.npz',allow_pickle=False);keys=list(m['models']);allrows={};provenance={}
for ik,key in enumerate(keys):
    rr={}
    for chunk in range(8):
        task=8*ik+chunk;p=OUT/f'result-{task}.npz';j=OUT/f'result-{task}.json';meta=read(j)
        assert meta['status']=='passed' and meta['checkpoint']==key and meta['model_sha256']==m['models'][key]['sha256']
        assert meta['predictions_sha256']==sha(p) and meta['input_sha256']==sha(OUT/'remote-input.json')
        assert meta['script_sha256']==sha(OUT/'checkpoint_sweep_remote.py')
        assert meta['indices']==list(range(chunk,m['total'],8))
        pred=np.load(p,allow_pickle=False)
        for i in meta['indices']:
            r=rows[i];n=r['n_atoms'];e=1000*(float(pred[f'energy_{i}'])-float(refs[f'energy_{i}']))/n
            f=pred[f'forces_{i}']-refs[f'forces_{i}'];v=-(pred[f'virial_{i}']-refs[f'virial_{i}'])/(n*r['volume_A3_atom'])*160.21766208
            assert f.shape==(n,3) and v.shape==(3,3) and all(np.isfinite(a).all() for a in [e,f,v])
            s=v[np.triu_indices(3)]
            rr[i]={**r,'energy_error_meV_atom':float(e),'force_sse':float((f*f).sum()),'force_sae':float(abs(f).sum()),
                   'force_count':f.size,'force_vector_norm_mae':float(np.linalg.norm(f,axis=1).mean()),
                   'stress_sse':float((s*s).sum()),'stress_sae':float(abs(s).sum()),'stress_count':6}
        pred.close();provenance[str(p)]=sha(p);provenance[str(j)]=sha(j)
    assert len(rr)==m['total'];allrows[key]=[rr[i] for i in range(m['total'])]
def metric(rr):
    if not rr:return None
    e=np.array([r['energy_error_meV_atom'] for r in rr])
    out={'n':len(rr),'energy_mae_meV_atom':float(abs(e).mean()),'energy_rmse_meV_atom':float(np.sqrt(np.mean(e**2))),
         'energy_bias_meV_atom':float(e.mean()),'energy_n_positive':int(sum(e>0)),
         'energy_centered_rmse_diagnostic_meV_atom':float(e.std())}
    for prop,unit in [('force','eV_A'),('stress','GPa')]:
        count=sum(r[prop+'_count'] for r in rr)
        out[prop+'_rmse_'+unit]=float(np.sqrt(sum(r[prop+'_sse'] for r in rr)/count))
        out[prop+'_mae_'+unit]=sum(r[prop+'_sae'] for r in rr)/count
    return out
groups={key:{split:metric([r for r in rr if r['split']==split]) for split in ['train','validation','test20']} for key,rr in allrows.items()}
subgroups={}
for key,rr in allrows.items():
    subgroups[key]={}
    for split in ['train','validation','test20']:
        ss=[r for r in rr if r['split']==split]
        subgroups[key][split]={name:metric([r for r in ss if fn(r)]) for name,fn in {
            'xTi_lt_0.5':lambda r:r['x_Ti']<.5,'xTi_ge_0.5':lambda r:r['x_Ti']>=.5,
            'xTi_ge_0.9':lambda r:r['x_Ti']>=.9,'pure_Ti':lambda r:r['x_Ti']==1,
            'Ti_rich_V_lt18':lambda r:r['x_Ti']>=.5 and r['volume_A3_atom']<18,
            'Ti_rich_V_18to20':lambda r:r['x_Ti']>=.5 and 18<=r['volume_A3_atom']<20,
            'Ti_rich_V_ge20':lambda r:r['x_Ti']>=.5 and r['volume_A3_atom']>=20}.items()}
# Paired, whole-configuration descriptive bootstrap on unchanged common20 only.
aa=[r for r in allrows['5000'] if r['split']=='test20'];bb=[r for r in allrows['23400'] if r['split']=='test20']
assert [r['id'] for r in aa]==[r['id'] for r in bb]
rng=np.random.default_rng(20261025);draws=rng.integers(0,20,size=(2000,20));boot={};late_boot={}
for prop in ['energy','force']:
    def sample_rms(rr):
        if prop=='energy':return np.sqrt(np.mean(np.array([r['energy_error_meV_atom']**2 for r in rr])[draws],axis=1))
        sse=np.array([r['force_sse'] for r in rr]);count=np.array([r['force_count'] for r in rr])
        return np.sqrt(sse[draws].sum(axis=1)/count[draws].sum(axis=1))
    a=sample_rms(aa);b=sample_rms(bb)
    boot[prop]={'5000_rmse_ci95':np.percentile(a,[2.5,97.5]).tolist(),'23400_rmse_ci95':np.percentile(b,[2.5,97.5]).tolist(),
                'paired_delta_final_minus_5000_ci95':np.percentile(b-a,[2.5,97.5]).tolist()}
    late=sample_rms([r for r in allrows['20000'] if r['split']=='test20'])
    late_boot[prop]={'paired_delta_23400_minus_20000_ci95':np.percentile(b-late,[2.5,97.5]).tolist()}
old=read(HERE/'pbe_train_comparison/analysis.json')['groups']['train']['all']['fine_tuned']
assert abs(groups['5000']['train']['energy_rmse_meV_atom']-old['rmse'])<.005
assert abs(groups['5000']['test20']['force_rmse_eV_A']-.067542311)<1e-5
pair={}
for split in ['train','validation','test20']:
    a=[r for r in allrows['5000'] if r['split']==split];b=[r for r in allrows['23400'] if r['split']==split]
    pair[split]={'n':len(a),'final_lower_absolute_energy':sum(abs(v['energy_error_meV_atom'])<abs(u['energy_error_meV_atom']) for u,v in zip(a,b)),
                 'final_lower_force_rmse_per_frame':sum(v['force_sse']<u['force_sse'] for u,v in zip(a,b))}
result={'groups':groups,'subgroups':subgroups,'common20_paired_bootstrap':boot,'common20_late_checkpoint_bootstrap':late_boot,'paired_frame_counts':pair,
        'test009':{k:next(r for r in rr if r['id']=='TiAl-test-009') for k,rr in allrows.items()},
        'definitions':'Raw energy meV/atom, component force eV/A, six independent stresses GPa. No fitted shifts; all checkpoints included. Bootstrap conditional on common20, no trajectory independence or sequential coverage claimed.',
        'selection_note':'Post-hoc checkpoint diagnostic; no change to the article model, original registry, or common20 cache. Phonon comparison is not recalculated here.'}
(OUT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'per_configuration.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')
for p in [OUT/'input-manifest.json',OUT/'remote-input.json',OUT/'rows.json',OUT/'labels-local-only.npz',Path(__file__)]:provenance[str(p)]=sha(p)
(OUT/'analysis-source-manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained');steps=[int(k)/1000 for k in keys]
for split,col,lab in [('train','#276FBF','Train (850)'),('validation','#D97917','Validation (149)'),('test20','#8C3F88','Common20 test')]:
    for ax,field in [(axs[0,0],'energy_rmse_meV_atom'),(axs[0,1],'force_rmse_eV_A')]:
        ax.plot(steps,[groups[k][split][field] for k in keys],'o-',color=col,label=lab)
    axs[1,0].plot(steps,[subgroups[k][split]['xTi_ge_0.9']['energy_bias_meV_atom'] for k in keys],'o-',color=col,label=lab)
axs[0,0].set(title='(a) Energy error',ylabel='RMSE (meV/atom)');axs[0,0].legend(frameon=False)
axs[0,1].set(title='(b) Force-component error',ylabel=r'RMSE (eV/$\AA$)')
axs[1,0].set(title=r'(c) Ti-rich energy bias, $x_{\mathrm{Ti}}\geq0.9$',ylabel='Mean error (meV/atom)');axs[1,0].axhline(0,c='.5',ls='--',lw=.7)
for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set(xlabel='Optimization steps (thousands)',xticks=steps)
for key,col,lab in [('5000','#276FBF','5000 steps'),('23400','#D97917','23400 steps')]:
    rr=[r for r in allrows[key] if r['split']=='train']
    axs[1,1].scatter([r['x_Ti'] for r in rr],[r['energy_error_meV_atom'] for r in rr],s=8,alpha=.35,color=col,label=lab)
axs[1,1].axhline(0,c='.5',ls='--',lw=.7);axs[1,1].set(title='(d) Training energy residuals',xlabel=r'$x_{\mathrm{Ti}}$',ylabel='Model − DFT (meV/atom)');axs[1,1].legend(frameon=False)
fig.suptitle('PBE checkpoint comparison on identical configurations; native CPU inference')
fig.savefig(OUT/'checkpoint_comparison.png',dpi=200)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,x,title in zip(axs,[205/216,1.0],[r'Al$_{11}$Ti$_{205}$ composition','Pure Ti']):
    for key,col in [('5000','#276FBF'),('23400','#D97917')]:
        tr=[r for r in allrows[key] if r['split']=='train' and abs(r['x_Ti']-x)<1e-12]
        te=[r for r in allrows[key] if r['split']=='test20' and abs(r['x_Ti']-x)<1e-12]
        ax.scatter([r['volume_A3_atom'] for r in tr],[r['energy_error_meV_atom'] for r in tr],s=30,color=col,label=f'{key}: train (n={len(tr)})')
        ax.scatter([r['volume_A3_atom'] for r in te],[r['energy_error_meV_atom'] for r in te],s=70,color=col,marker='D',edgecolors='red',linewidths=1.2,label=f'{key}: test')
    ax.axhline(0,color='.4',ls='--',lw=.7);ax.set(title=title,xlabel=r'Volume per atom ($\AA^3$)',ylabel='Energy error (meV/atom)');ax.legend(frameon=False,fontsize=8)
fig.suptitle('Volume dependence at fixed composition; no energy shifts')
fig.savefig(OUT/'fixed_composition_volume.png',dpi=200)
print(json.dumps({'groups':groups,'paired_bootstrap':boot,'paired_counts':pair,
                  'test009_energy':{k:v['energy_error_meV_atom'] for k,v in result['test009'].items()}},indent=2))
