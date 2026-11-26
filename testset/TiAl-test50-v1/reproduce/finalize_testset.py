from audit_sources import ROOT, BASE, CACHE
from select_testset import QUOTAS
import csv, hashlib, json, pickle, shutil, zipfile
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=ROOT/'TiAl-test50-v1'
m=json.loads((out/'manifest.json').read_text())
v=json.loads((out/'validation.json').read_text())
assert v['status']=='PASS'
data=pickle.load((CACHE/'reference.pkl').open('rb'))
solid=pickle.load((CACHE/'solid_candidates.pkl').open('rb'))
candidates=[dict(r,family='liquid',trajectory=None,frame=int(r['member'].rsplit('.',1)[1])) for r in data['liquids'] if not r['matches']]+solid
assert len(candidates)==1707
pool=[dict({k:v for k,v in r.items() if k not in ('atoms','matches')},candidate_index=i,n_atoms=len(r['atoms']),x_Al=float(np.mean(r['atoms'].numbers==13)),volume_per_atom_A3=float(r['atoms'].get_volume()/len(r['atoms']))) for i,r in enumerate(candidates)]
(out/'candidate_manifest.json').write_text(json.dumps(pool,indent=2))
for r,check in zip(m,v['source_checks']):
    assert r['structure_id']==check['structure_id']
    r['kinetic_temperature_K']=check['kinetic_temperature_K']
(out/'manifest.json').write_text(json.dumps(m,indent=2))
with (out/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in m))));w.writeheader();w.writerows(m)
protocol=json.loads((out/'selection_protocol.json').read_text())
protocol['limitations'].append('The liquid-source label is not a verified phase assignment; selected instantaneous kinetic temperatures span 590–3062 K.')
protocol['sources']=['https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html','https://docs.ase-lib.org/ase/neighborlist.html']
(out/'selection_protocol.json').write_text(json.dumps(protocol,indent=2))
for name in ['source_audit.json','trajectory_audit.json']:
    shutil.copy2(ROOT/name,out/name)
aux={}
for r in json.loads((ROOT/'source_audit.json').read_text())['reference_files']:
    p=BASE/r['file']
    if p.name=='coord.npy':
        for q in [p.with_name('box.npy'),p.parent.parent/'type.raw',p.parent.parent/'type_map.raw']:
            rel=q.relative_to(BASE).as_posix()
            if rel not in aux:aux[rel]=hashlib.sha256(q.read_bytes()).hexdigest()
(out/'reference_auxiliary_sha256.json').write_text(json.dumps(aux,indent=2))
repro=out/'reproduce';repro.mkdir(exist_ok=True)
for name in ['audit_sources.py','select_testset.py','verify_testset.py','finalize_testset.py']:
    shutil.copy2(ROOT/name,repro/name)
(out/'audit-environment.txt').write_text('Audit executed with Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, ASE 3.29.0.\nSelection and validation executed in the isolated Python 3.11.5 environment recorded in requirements-lock.txt.\nAll source files are read only. No DFT calculation submitted.\n')
desc=np.load(out/'selection_descriptors.npz');distances=cdist(desc['candidate_soap'],desc['reference_soap']);selected=desc['selected_indices']
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for color,fam in zip(['#167ca6','#a64b00','#13845c','#8a4a9f'],QUOTAS):
    pp=[r for r in pool if r['family']==fam];ss=[r for r in m if r['family']==fam]
    axes[0].scatter([r['x_Al'] for r in pp],[r['volume_per_atom_A3'] for r in pp],s=8,alpha=.12,color=color)
    axes[0].scatter([r['x_Al'] for r in ss],[r['volume_per_atom_A3'] for r in ss],s=40,color=color,label=fam.replace('POSCAR_',''))
axes[0].set(xlabel='Al atomic fraction',ylabel='Volume / atom (Å³)')
axes[0].legend(fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2)
bins=np.linspace(0,distances.min(axis=1).max()*1.01,36)
axes[1].hist(distances.min(axis=1),bins=bins,alpha=.45,label='Eligible pool (1707)')
axes[1].hist(distances[selected].min(axis=1),bins=bins,alpha=.8,label='Test50')
axes[1].set(xlabel='Distance to nearest reference (normalized SOAP)',ylabel='Configurations',yscale='log');axes[1].legend()
fig.savefig(out/'coverage.png',dpi=180);fig.savefig(out/'coverage.pdf');plt.close(fig)
print('Finalized',len(m),'structures; auxiliary source hashes',len(aux))
