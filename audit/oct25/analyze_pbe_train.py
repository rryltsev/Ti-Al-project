"""Compare raw energy residuals on identical training frames and frozen common20."""
from pathlib import Path
import json, hashlib, sys, os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'pbe_train_comparison';MON=ROOT/'testset/monitor';CAL=MON/'alignment/train-offsets-v1'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'));os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(CAL/'input-audit.json');cfg=read(OUT/'inference-input.json')
assert sha(CAL/'training-geometries.npz')==audit['geometries_sha256']==cfg['geometries_sha256']
assert sha(CAL/'training-labels.json')==audit['labels_sha256']
registry=read(MON/'model_registry.json');pre=read(MON/'pretrained_registry.json')
assert registry['models']['PBE']['sha256']==cfg['model_sha256']
provenance={}
def load_chunks(folder,model_sha):
    energies={}
    for i in range(8):
        p=folder/f'result-{i}.json';r=read(p)
        assert sha(p)==(folder/f'result-{i}.sha256').read_text().strip()
        assert r['model_sha256']==model_sha and r['status']=='passed'
        assert r['geometries_sha256']==audit['geometries_sha256'] and r['task']==i
        assert not (energies.keys() & r['energies_eV'].keys())
        if folder==OUT:
            assert r['input_sha256']==sha(OUT/'inference-input.json')
            assert r['script_sha256']==sha(OUT/'pbe_train_compare_remote.py')
        energies.update(r['energies_eV']);provenance[str(p)]=sha(p)
    return energies
preds={'fine_tuned':load_chunks(OUT,cfg['model_sha256']),
       'pretrained':load_chunks(CAL,pre['sha256'])}
geoms=np.load(CAL/'training-geometries.npz',allow_pickle=False)
train=[]
for row in read(CAL/'training-labels.json'):
    if row['functional']!='PBE':continue
    i=row['geometry_id'];n=row['n_atoms']
    types=geoms[f'types_{i}'];cell=geoms[f'cell_{i}'];coords=geoms[f'coords_{i}']
    assert hashlib.sha256(types.astype('<i4').tobytes()+cell.astype('<f8').tobytes()+coords.astype('<f8').tobytes()).hexdigest()==row['geometry_sha256']
    assert len(types)==n and int(sum(types==1))==row['n_Ti']
    d={**row,'id':f'train-geom-{i:04d}','x_Ti':row['n_Ti']/n,
       'volume_A3_atom':float(abs(np.linalg.det(cell))/n)}
    for role,pred in preds.items():
        d[role]=1000*(pred[str(i)]-row['energy_eV'])/n
    train.append(d)
assert len(train)==850 and set(map(str,cfg['geometry_ids']))==preds['fine_tuned'].keys()
test=[]
for r in read(HERE/'test20_bundle.json')['records']:
    m=r['metadata'];d={'id':m['structure_id'],'n_atoms':m['n_atoms'],
        'x_Ti':m['n_Ti']/m['n_atoms'],'volume_A3_atom':m['volume_per_atom_A3']}
    for role,name in [('fine_tuned','PBE'),('pretrained','pretrained')]:
        p=MON/'predictions'/name/(d['id']+'.json');pred=read(p)
        assert pred['poscar_sha256']==m['poscar_sha256']
        assert pred['model_sha256']==(cfg['model_sha256'] if role=='fine_tuned' else pre['sha256'])
        d[role]=1000*(pred['energy_eV']-r['labels']['PBE']['energy_eV'])/m['n_atoms']
        provenance[str(p)]=sha(p)
    test.append(d)
def stat(rr,role):
    e=np.array([r[role] for r in rr]);v=np.array([r['volume_A3_atom'] for r in rr])
    return {'n':len(e),'bias':float(e.mean()),'mae':float(abs(e).mean()),
            'rmse':float(np.sqrt(np.mean(e**2))),'median':float(np.median(e)),
            'centered_rmse':float(e.std()),'n_positive':int(sum(e>0)),
            'volume_range':[float(v.min()),float(v.max())]}
def both(rr):return {role:stat(rr,role) for role in preds} if rr else None
selectors={'all':lambda r:True,'xTi_lt_0.5':lambda r:r['x_Ti']<.5,
    'xTi_ge_0.5':lambda r:r['x_Ti']>=.5,'xTi_ge_0.75':lambda r:r['x_Ti']>=.75,
    'xTi_ge_0.9':lambda r:r['x_Ti']>=.9,'pure_Ti':lambda r:r['x_Ti']==1,
    'xTi_ge_0.5_N216':lambda r:r['x_Ti']>=.5 and r['n_atoms']==216}
groups={split:{name:both([r for r in rr if fn(r)]) for name,fn in selectors.items()}
        for split,rr in [('train',train),('test20',test)]}
volume_bins=[('V_lt_18',0,18),('V_18_to_20',18,20),('V_ge_20',20,float('inf'))]
volume_groups={split:{name:both([r for r in rr if r['x_Ti']>=.5 and lo<=r['volume_A3_atom']<hi])
                     for name,lo,hi in volume_bins} for split,rr in [('train',train),('test20',test)]}
composition=[]
for x in sorted(set(r['x_Ti'] for r in train)):
    rr=[r for r in train if r['x_Ti']==x]; entry={'x_Ti':x,'metrics':both(rr)}
    if len(rr)>2 and np.ptp([r['volume_A3_atom'] for r in rr])>0:
        entry['volume_slopes_meV_per_A3']={role:float(np.polyfit([r['volume_A3_atom'] for r in rr],[r[role] for r in rr],1)[0]) for role in preds}
    composition.append(entry)
# Center within each exact composition to diagnose structural energy differences.
within={}
for split,rr in [('train',train),('test20',test)]:
    within[split]={}
    eligible=[r for r in rr if sum(v['x_Ti']==r['x_Ti'] for v in rr)>1]
    for role in preds:
        centered=[];volumes=[]
        for r in eligible:
            peers=[v for v in eligible if v['x_Ti']==r['x_Ti']]
            centered.append(r[role]-np.mean([v[role] for v in peers]))
            volumes.append(r['volume_A3_atom']-np.mean([v['volume_A3_atom'] for v in peers]))
        within[split][role]={'n':len(eligible),'centered_within_composition_rmse':float(np.sqrt(np.mean(np.array(centered)**2))),
             'pooled_within_composition_volume_slope':float(np.dot(volumes,centered)/np.dot(volumes,volumes))}
# Test errors in relation to exact-composition train coverage; no reweighting/exclusion.
coverage=[]
for t in test:
    peers=[r for r in train if abs(r['x_Ti']-t['x_Ti'])<1e-12]
    d=dict(t);d['train_same_composition']=both(peers)
    if peers:
        lo=min(r['volume_A3_atom'] for r in peers);hi=max(r['volume_A3_atom'] for r in peers)
        d['outside_train_volume_range']=t['volume_A3_atom']<lo or t['volume_A3_atom']>hi
        d['nearest_volume_train']=min(peers,key=lambda r:abs(r['volume_A3_atom']-t['volume_A3_atom']))
    coverage.append(d)
summary={'energy_units':'meV/atom; model minus PBE DFT; raw energies; no shifts fitted/applied',
    'groups':groups,'volume_groups_Ti_rich':volume_groups,'exact_composition':composition,
    'within_composition_diagnostic':within,'test_vs_train_coverage':coverage,
    'counts_fine_tuned_lower_absolute':{name:sum(abs(r['fine_tuned'])<abs(r['pretrained']) for r in rr) for name,rr in [('train',train),('test20',test)]}}
for p in [CAL/'training-geometries.npz',CAL/'training-labels.json',CAL/'input-audit.json',HERE/'test20_bundle.json',MON/'model_registry.json',MON/'pretrained_registry.json',OUT/'inference-input.json']:
    provenance[str(p)]=sha(p)
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'all_residuals.json').write_text(json.dumps({'train':train,'test20':test},indent=2),encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for col,(role,title) in enumerate([('pretrained','Pretrained'),('fine_tuned','Fine-tuned PBE')]):
    ax=axs[0,col]
    sc=ax.scatter([r['x_Ti'] for r in train],[r[role] for r in train],c=[r['volume_A3_atom'] for r in train],cmap='viridis',vmin=14,vmax=23,s=12,alpha=.65,label='Training (850)')
    ax.scatter([r['x_Ti'] for r in test],[r[role] for r in test],c=[r['volume_A3_atom'] for r in test],cmap='viridis',vmin=14,vmax=23,s=75,marker='D',edgecolors='red',linewidths=.9,label='Common20 test')
    ax.set(title=title,xlabel=r'$x_{\mathrm{Ti}}$',ylabel='Energy error (meV/atom)');ax.legend(loc='upper left',fontsize=8)
    ax=axs[1,col];rich=[r for r in train if r['x_Ti']>=.5];tt=[r for r in test if r['x_Ti']>=.5]
    sc2=ax.scatter([r['volume_A3_atom'] for r in rich],[r[role] for r in rich],c=[r['x_Ti'] for r in rich],cmap='plasma',vmin=.5,vmax=1,s=12,alpha=.65)
    ax.scatter([r['volume_A3_atom'] for r in tt],[r[role] for r in tt],c=[r['x_Ti'] for r in tt],cmap='plasma',vmin=.5,vmax=1,s=75,marker='D',edgecolors='red',linewidths=.9)
    ax.set(title=title+r': $x_{\mathrm{Ti}}\geq0.5$',xlabel=r'Volume per atom ($\AA^3$)',ylabel='Energy error (meV/atom)')
    if role=='fine_tuned':
        t=next(r for r in tt if r['id']=='TiAl-test-009');ax.annotate('Test009',(t['volume_A3_atom'],t[role]),xytext=(-65,18),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','lw':.7},bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
for row in axs:
    lo=min(ax.get_ylim()[0] for ax in row);hi=max(ax.get_ylim()[1] for ax in row)
    for ax in row:ax.set_ylim(lo,hi);ax.axhline(0,ls='--',lw=.7,color='.4')
fig.colorbar(sc,ax=list(axs[0]),label=r'Volume per atom ($\AA^3$)',shrink=.85)
fig.colorbar(sc2,ax=list(axs[1]),label=r'$x_{\mathrm{Ti}}$',shrink=.85)
fig.suptitle('PBE energy residuals: 850 training frames and 20 held-out liquid frames')
fig.savefig(OUT/'train_test_residuals.png',dpi=200)
print(json.dumps({'groups':groups,'volume_groups':volume_groups,'within_composition':within},indent=2))
print('Selected exact composition train summaries: xTi N Vmin Vmax biasFT RMSEFT biasPT RMSEPT')
for r in composition:
    if not any(abs(r['x_Ti']-x)<1e-12 for x in [.75,183/216,205/216,1]):continue
    a=r['metrics']['fine_tuned'];b=r['metrics']['pretrained']
    print(r['x_Ti'],a['n'],*a['volume_range'],a['bias'],a['rmse'],b['bias'],b['rmse'])
