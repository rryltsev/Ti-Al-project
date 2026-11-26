"""Freeze all-functional checkpoint audit inputs without changing article/monitor data."""
from pathlib import Path
import sys,json,hashlib,shlex,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'audit/oct20/deps'),str(ROOT/'testset/dft')]
import numpy as np
from ase.io import read
import cluster_test50 as c
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2),encoding='utf-8')
assert not (OUT/'protocol.json').exists(),'Frozen audit already exists'
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
REMOTE='/home6/u3129/tial-testsets/all-xc-checkpoint-audit-20261025-01'
snapshot=OUT/'test_snapshot.json'
if snapshot.exists():bundle=js(snapshot)
else:
    code=(ROOT/'testset/monitor/collect_remote.py').read_text()
    bundle=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
    assert not bundle['problems'];save(snapshot.name,bundle)
print('DFT snapshot',bundle['validated_counts'],'common',len(bundle['records']),flush=True)
assert bundle['manifest_sha256']==sha(ROOT/'testset/TiAl-test50-v1/manifest.json')
registry=js(ROOT/'testset/monitor/model_registry.json')['models']
spec={'PBE':('pbe','20260920_140047','TiAl_PBE_upd',23400,5000),
      'PBEsol':('sol','20260920_040140','TiAl_PBEsol_upd',23100,23100),
      'r2SCAN':('scan','20260919_153828','TiAl_scan_upd',23200,23200)}
models={};datasets={};liquids={};configs={};all_sources={};reuse={}
for xc,(tag,date,ds,final,article) in spec.items():
    run=SOURCE/f'DPA4-checkpoints/TiAl_tune_{tag}_Neo_DPA4_shard_{date}'
    models_xc={}
    for step in [5000,10000,15000,20000,final]:
        key=xc+'-'+str(step);path=run/f'output_model/ckpt/model.ckpt-{step}.pt'
        h=sha(path);models[key]={'functional':xc,'step':step,'local_path':str(path),'sha256':h,
             'path':REMOTE+'/models/'+key+'.pt','article_checkpoint':step==article}
        if xc=='PBE':models[key]['path']=js(HERE/'pbe_checkpoint_sweep/remote-input.json')['models'][str(step)]['path']
        if step==article:assert h==registry[xc]['checkpoint_sha256'],(xc,h)
        models_xc[step]=h
    root_sha=sha(run/'model.ckpt.pt');assert root_sha in models_xc.values(),(xc,'unlisted root checkpoint')
    for filename in ['lcurve.out','input.json','out.json']:
        source=run/filename;dest=OUT/(xc+'-'+filename);shutil.copyfile(source,dest);all_sources[str(source)]=sha(source)
    inp=js(run/'input.json');configs[xc]={'input':inp,'root_checkpoint_sha256':root_sha,'article_step':article}
    base=SOURCE/'DPA4-finetune'/ds/'npy_format';geo={};labels={};rows=[];sources={}
    def add(pos,cell,types,energy,forces,virial,meta):
        i=len(rows);n=len(types)
        geo[f'coords_{i}']=np.array(pos,dtype='<f8').reshape(n,3)
        geo[f'cell_{i}']=np.array(cell,dtype='<f8').reshape(3,3)
        geo[f'types_{i}']=np.array(types,dtype='<i4')
        labels[f'energy_{i}']=float(energy);labels[f'forces_{i}']=np.array(forces).reshape(n,3)
        labels[f'virial_{i}']=np.array(virial).reshape(3,3)
        rows.append({**meta,'i':i,'n_atoms':n,'x_Ti':sum(t==1 for t in types)/n,
                     'volume_A3_atom':float(abs(np.linalg.det(geo[f'cell_{i}']))/n)})
    for split,filename in [('train','training_systems.json'),('validation','validation_systems.json')]:
        p=base/filename;sources[str(p)]=sha(p)
        for rel in sorted(s.split('/npy_format/',1)[1] for s in js(p)):
            folder=base/rel;types=np.loadtxt(folder/'type.raw',dtype=int,ndmin=1)
            names=(folder/'type_map.raw').read_text().split();types=[['Al','Ti'].index(names[t]) for t in types]
            for p in [folder/'type.raw',folder/'type_map.raw']:sources[str(p)]=sha(p)
            for sd in sorted(folder.glob('set.*')):
                arrays={}
                for field in ['coord','box','energy','force','virial']:
                    p=sd/(field+'.npy');sources[str(p)]=sha(p);arrays[field]=np.load(p,allow_pickle=False)
                for j,e in enumerate(arrays['energy'].reshape(-1)):
                    add(arrays['coord'][j],arrays['box'][j],types,e,arrays['force'][j],arrays['virial'][j],
                        {'split':split,'id':f'{split}/{folder.name}/{sd.name}/{j}','system':rel,'frame':j,'set':sd.name})
    for r in bundle['records']:
        lab=r['labels'][xc];cell=np.array(r['cell_A'])
        add(np.array(r['fractional_positions'])@cell,cell,[['Al','Ti'].index(s) for s in r['symbols']],
            lab['energy_eV'],lab['forces_eV_A'],-np.array(lab['stress_eV_A3'])*lab['volume_A3'],
            {'split':'test','id':r['metadata']['structure_id'],'family':r['metadata']['family']})
    counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}
    np.savez_compressed(OUT/(xc+'-geometries.npz'),**geo)
    np.savez_compressed(OUT/(xc+'-labels.npz'),**labels);save(xc+'-rows.json',rows)
    datasets[xc]={'counts':counts,'n':len(rows),'geometry_file':xc+'-geometries.npz',
      'geometry_sha256':sha(OUT/(xc+'-geometries.npz')),'rows_file':xc+'-rows.json',
      'rows_sha256':sha(OUT/(xc+'-rows.json')),'labels_sha256':sha(OUT/(xc+'-labels.npz')),'sources':sources}
    if xc=='PBE':
        old=np.load(HERE/'pbe_checkpoint_sweep/geometries.npz');oldlab=np.load(HERE/'pbe_checkpoint_sweep/labels-local-only.npz')
        oldrows=js(HERE/'pbe_checkpoint_sweep/rows.json');byid={r['id']:r['i'] for r in oldrows}
        mapping={}
        for r in rows:
            if r['id'] not in byid:continue
            j=byid[r['id']];i=r['i']
            for field in ['coords','cell','types']:assert np.array_equal(geo[f'{field}_{i}'],old[f'{field}_{j}'])
            for field in ['energy','forces','virial']:assert np.array_equal(labels[f'{field}_{i}'],oldlab[f'{field}_{j}'])
            mapping[i]=j
        reuse['PBE_inference']={'mapping_new_to_old':mapping,'directory':str(HERE/'pbe_checkpoint_sweep')}
    # Own-functional archived liquid ensembles, identical selection rule.
    g={};cache={};rr=[];src=[]
    base=SOURCE/f'results-scripts-logs/models_check_wsl2/results/mixing_enthalpy/dpa4/neo/finetune/AlTi/2000K/{tag}'
    for name,x in [('Ti',0.),('Al0.1Ti0.9',.1),('Al0.3Ti0.7',.3),('Al0.5Ti0.5',.5),('Al',1.)]:
        p=base/(name+'_2000K.traj');frames=read(p,index=':');assert len(frames)==301,(xc,name,len(frames))
        src.append({'path':str(p),'sha256':sha(p)})
        for j in range(6,301,6):
            at=frames[j];i=len(rr);symbols=at.get_chemical_symbols();assert len(at)==200 and symbols.count('Al')/200==x
            g[f'coords_{i}']=at.positions;g[f'cell_{i}']=at.cell.array
            g[f'types_{i}']=np.array([['Al','Ti'].index(s) for s in symbols],dtype='i4')
            for field in ['energy','forces','stress']:cache[f'{field}_{i}']=np.array(at.calc.results[field])
            rr.append({'i':i,'composition':name,'x_Al':x,'frame':j,'time_ps':j*.1,'n_atoms':len(at),
                       'V_A3_atom':at.get_volume()/len(at),'T_K':at.get_temperature()})
    np.savez_compressed(OUT/(xc+'-liquid.npz'),**g);np.savez_compressed(OUT/(xc+'-archive.npz'),**cache);save(xc+'-liquid-rows.json',rr)
    liquids[xc]={'geometry_file':xc+'-liquid.npz','geometry_sha256':sha(OUT/(xc+'-liquid.npz')),
                 'rows_file':xc+'-liquid-rows.json','sources':src,'T_K':2000,'frames':250,'article_step':article}
    print(xc,counts,'liquid frames',len(rr),flush=True)
save('configurations.json',configs);save('reuse.json',reuse)
shutil.copyfile(HERE/'pbe_phonons_dft_geometry/POSCAR-DFT',OUT/'POSCAR-DFT')
shutil.copyfile(HERE/'pbe_phonons_dft_geometry/DFT-total_dos.dat',OUT/'DFT-total_dos.dat')
tasks=[]
def task(kind,key,**kw):tasks.append({'id':len(tasks),'kind':kind,'key':key,**kw})
# Tests first, then all property tasks, then comprehensive train/validation in bounded chunks.
for key,m in models.items():
    rows=js(OUT/(m['functional']+'-rows.json'))
    indices=[r['i'] for r in rows if r['split']=='test' and (m['functional']!='PBE' or str(r['i']) not in {str(i) for i in reuse['PBE_inference']['mapping_new_to_old']})]
    if indices:task('inference',key,indices=indices,split='test')
for key,m in models.items():
    # Reuse all existing PBE fixed-cell phonons, and completed 5k/15k static/liquid/equilibrium results.
    if m['functional']=='PBE' and m['step'] in [5000,15000]:continue
    task('properties',key,fixed_phonons=m['functional']!='PBE',equilibrium_phonons=True)
for key,m in models.items():
    if m['functional']=='PBE':continue
    rows=js(OUT/(m['functional']+'-rows.json'));indices=[r['i'] for r in rows if r['split']!='test']
    for chunk in range(4):task('inference',key,indices=indices[chunk::4],split='train_validation')
save('tasks.json',tasks)
cfg={'id':'all-xc-checkpoint-audit-20261025-01','remote':REMOTE,'models':models,'datasets':datasets,'liquids':liquids,
 'geometry_sha256':sha(OUT/'POSCAR-DFT'),'tasks_sha256':sha(OUT/'tasks.json'),'source_files':all_sources,
 'eos':{'Ti_hcp_reference_V_A3_atom':17.5,'Ti_bcc_reference_V_A3_atom':17.5,'Al_fcc_reference_V_A3_atom':16.6,
 'volume_factors':np.linspace(.88,1.08,13).tolist(),'hcp_c_over_a_bracket':[1.40,1.80]},
 'phonons':{'supercell':[9,9,9],'delta_A':.01,'mass_amu':47.867,'DFT_reference_functional':'PBE'},
 'policy':'Checkpoint sensitivity audit, not final untouched test validation. No exclusions or fitted energy shifts. Existing article models and DFT jobs remain unchanged. All saved checkpoints included; PBE prior results reused after identity checks.'}
save('protocol.json',cfg)
print('TASKS',len(tasks),'MODELS',len(models),'reuse PBE',len(reuse['PBE_inference']['mapping_new_to_old']),flush=True)
