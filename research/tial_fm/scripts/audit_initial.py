"""Read-only source audit; write a new immutable evidence run. No DFT or training."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, shutil, sys, time
import numpy as np
from ase import Atoms
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
SOURCE = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
XC = {'PBE':'TiAl_PBE_upd','PBEsol':'TiAl_PBEsol_upd','r2SCAN':'TiAl_scan_upd'}
SOURCES = {}

def sha(p):
    p=Path(p); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    v=h.hexdigest(); SOURCES[str(p.resolve())]={'sha256':v,'bytes':p.stat().st_size}; return v

def js(p):
    sha(p); return json.loads(Path(p).read_text(encoding='utf8'))

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def same(a,b,tol=5e-5):
    """Species-bijective matching, periodic images/common shifts, fixed cell axes."""
    if Counter(a.numbers)!=Counter(b.numbers) or np.max(abs(a.cell.array-b.cell.array))>tol:return False
    inv=np.linalg.inv(b.cell.array); pa=a.positions@inv; pb=b.positions@inv
    def check(shift):
        for z in set(a.numbers):
            x=pa[a.numbers==z]+shift;y=pb[b.numbers==z]
            d=x[:,None,:]-y[None,:,:];d-=np.rint(d)
            dist=np.linalg.norm(d@b.cell.array,axis=2); near=dist.argmin(axis=1)
            if len(set(near))!=len(x) or np.max(dist[np.arange(len(x)),near])>tol:return False
        return True
    if check(np.zeros(3)):return True
    z=min(set(a.numbers),key=lambda x:np.count_nonzero(a.numbers==x))
    anchor=pa[np.where(a.numbers==z)[0][0]]
    return any(check(target-anchor) for target in pb[b.numbers==z])

def geom_id(i):return f'tial-selected-{i:04d}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
    out=ROOT/'runs'/a.run_id;out.mkdir(exist_ok=False);start=time.perf_counter()
    sha(Path(__file__));sha(ROOT/'protocol/CODEX_TiAl_FM_RESEARCH_PLAN.md')
    selected=SOURCE/'MACE-dataset-gen/MACE-database/TiAl-liq-relax-ss_select-1000.xyz'
    selected_hash=sha(selected);frames=read(selected,index=':');assert len(frames)==1000
    origins=js(WORK/'testset/TiAl-test50-v1/reference_origins.json')
    origin_by_frame={}
    for group in origins:
        hits=[r for r in group if r['file']==selected.relative_to(SOURCE).as_posix()]
        assert len(hits)==1
        origin_by_frame[hits[0]['frame']]=group
    assert set(origin_by_frame)==set(range(1000))
    old_sources=js(WORK/'testset/source_audit.json')
    for s in old_sources['reference_files']:
        if s['file']==selected.relative_to(SOURCE).as_posix():assert s['sha256']==selected_hash
    # Check the structural comparator on transformations relevant to these exports.
    sample=frames[100];perm=sample[np.random.default_rng(42).permutation(len(sample))]
    perm.positions+=np.array([.123,.456,.789])+sample.cell.array[0]
    assert same(sample,perm)
    bad=sample.copy();bad.positions[0]+=[.01,0,0];assert not same(sample,bad)
    cache={}; records=[];hist={xc:{'train':[],'validate':[]} for xc in XC}
    failures=[];mapping_count=0
    trajectories=js(WORK/'testset/trajectory_audit.json')
    provenance=trajectories['reference_provenance']
    # Historical reference indices coincide with selected frame order; verify below.
    for idx,group in enumerate(origins):
        assert any(r['frame']==idx and r['file']==selected.relative_to(SOURCE).as_posix() for r in group)
    for i,atoms in enumerate(frames):
        labels={}
        for xc,dirname in XC.items():
            matches=[r for r in origin_by_frame[i] if r['file'].startswith(f'DPA4-finetune/{dirname}/npy_format/') and r['file'].endswith('coord.npy')]
            assert len(matches)<=1,(i,xc,matches)
            if not matches:continue
            r=matches[0];p=SOURCE/r['file'];system=p.parent.parent;frame=r['frame']
            if str(p) not in cache:
                mp=(system/'type_map.raw').read_text().split();ty=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
                syms=[mp[t] for t in ty];arr={k:np.load(p.with_name(k+'.npy'),allow_pickle=False) for k in ['coord','box','energy','force','virial']}
                assert all(np.isfinite(v).all() for v in arr.values())
                for k in arr:sha(p.with_name(k+'.npy'))
                sha(system/'type.raw');sha(system/'type_map.raw');cache[str(p)]=(syms,arr)
            syms,arr=cache[str(p)]
            actual=Atoms(syms,positions=arr['coord'][frame].reshape(-1,3),cell=arr['box'][frame].reshape(3,3),pbc=True)
            assert same(atoms,actual),(i,xc,p,frame)
            mapping_count+=1;split=p.relative_to(SOURCE).parts[3];assert split in ['train','validate']
            hist[xc][split].append(geom_id(i))
            primary=[str(SOURCE/x['file']) for x in origin_by_frame[i] if x['file'].startswith('VASP-output/') and ('RESULTS' in x['file'] or '_database/' in x['file'])]
            labels[xc]={'split':split,'coord_path':str(p),'frame_index':frame,'energy_path':str(p.with_name('energy.npy')),'force_path':str(p.with_name('force.npy')),'virial_path':str(p.with_name('virial.npy')),'energy_definition':'legacy NPY; per-source raw VASP energy-field audit pending','dft_settings_id':'legacy-'+xc+'-pending-raw-audit','scf_status':'not_reaudited','primary_result_candidates_all_XC':primary}
        parents=provenance.get(str(i),[])
        records.append({'geometry_id':geom_id(i),'identity_basis':{'selected_file_sha256':selected_hash,'frame_index_zero_based':i},'natoms':len(atoms),'composition':dict(Counter(atoms.get_chemical_symbols())),'cell_A':atoms.cell.array.tolist(),'pbc':atoms.pbc.tolist(),'source':str(selected),'parent_status':'archived_matching_evidence' if parents else 'unresolved','parent_ids':sorted(set(r['trajectory'] for r in parents)),'parent_frames':parents,'structural_family':parents[0]['trajectory'].split('/')[-3] if parents else 'unresolved_liquid_relax_pool','temperature_K':None,'pressure':None,'labels':labels})
    common=sorted(set.intersection(*[set(hist[x]['train']+hist[x]['validate']) for x in XC]))
    alltrain=sorted(set.intersection(*[set(hist[x]['train']) for x in XC]));allval=sorted(set.intersection(*[set(hist[x]['validate']) for x in XC]))
    # Coverage checks ensure that the archived mapping did not silently omit new rows.
    actual_count={}
    for xc,dirname in XC.items():
        actual_count[xc]={}
        for split in ['train','validate']:
            paths=list((SOURCE/f'DPA4-finetune/{dirname}/npy_format/{split}').glob('*/set.*/coord.npy'))
            total=sum(len(np.load(p,mmap_mode='r',allow_pickle=False)) for p in paths)
            assert total==len(hist[xc][split])
            assert len(set(hist[xc][split]))==total
            actual_count[xc][split]=total
        assert not set(hist[xc]['train'])&set(hist[xc]['validate'])
    parent_groups=defaultdict(set)
    for r in records:
        for p in r['parent_ids']:parent_groups[p].add(r['geometry_id'])
    leaks={xc:[{'parent_id':p,'train_ids':sorted(ids&set(hist[xc]['train'])),'development_ids':sorted(ids&set(hist[xc]['validate']))} for p,ids in parent_groups.items() if ids&set(hist[xc]['train']) and ids&set(hist[xc]['validate'])] for xc in XC}
    manifest=out/'data_manifest.jsonl';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
    write(out/'splits.json',{'historical':hist,'paired_pool_geometry_ids':common,'historical_common_train':alltrain,'historical_common_validation':allval,'new_grouped_benchmark':{'state':'blocked','reason':'All parent identities, near-duplicate clusters and external-parent exclusions must be resolved; historical parent overlaps detected. No random replacement split released.'},'parent_overlap_evidence':leaks})
    # Independent arithmetic reproduction from predictions, with explicit identities.
    mon=WORK/'testset/monitor';bundle=js(WORK/'audit/nov03/article_package/test49_input_snapshot.json');published=js(WORK/'manuscript/supporting_data/test49_metrics.json')
    registry=js(mon/'model_registry.json');pre=js(mon/'pretrained_registry.json');assert bundle['common_ids']==published['common_ids']
    baseline={};residuals=[]
    def stats(v):
        v=np.asarray(v);assert np.isfinite(v).all()
        return {'mae':float(np.abs(v).mean()),'rmse':float(np.sqrt(np.mean(v*v))),'bias':float(v.mean()),'p95_abs':float(np.quantile(abs(v),.95)),'max_abs':float(abs(v).max()),'n_values':int(v.size)}
    for xc in XC:
        for role,cache_name,kind in [('ft',registry['models'][xc].get('prediction_cache',xc),'metrics'),('pretrained','pretrained','pretrained_metrics')]:
            expected=registry['models'][xc]['sha256'] if role=='ft' else pre['sha256'];values=defaultdict(list)
            for r in bundle['records']:
                sid=r['metadata']['structure_id'];pred=js(mon/'predictions'/cache_name/(sid+'.json'));ref=r['labels'][xc]
                assert pred['model_sha256']==expected and pred['poscar_sha256']==r['metadata']['poscar_sha256'] and pred['symbols']==r['symbols'] and ref['scf_converged']
                assert sha(WORK/'testset/TiAl-test50-v1'/r['metadata']['poscar'])==r['metadata']['poscar_sha256']
                n=len(r['symbols']);er=(pred['energy_eV']-ref['energy_eV'])/n*1000
                fr=np.array(pred['forces_eV_A'])-np.array(ref['forces_eV_A']);sr=(-np.array(pred['virial_eV'])/ref['volume_A3']-np.array(ref['stress_eV_A3']))[np.triu_indices(3)]*160.21766208
                for key,v in [('energy_meV_atom',[er]),('force_eV_A',fr.ravel()),('stress_GPa',sr)]:values[key].extend(v)
                residuals.append({'xc':xc,'model':role,'geometry_id':sid,'family':r['metadata']['family'],'n_atoms':n,'energy_residual_meV_atom':er,'force_rmse_eV_A':float(np.sqrt(np.mean(fr*fr))),'stress_rmse_GPa':float(np.sqrt(np.mean(sr*sr)))})
            baseline[xc+'/'+role]={k:stats(v) for k,v in values.items()}
            for k,v in baseline[xc+'/'+role].items():
                for stat in ['mae','rmse','bias']:assert np.isclose(v[stat],published[kind][xc]['all'][k][stat],rtol=1e-9,atol=1e-10),(xc,role,k,stat)
    write(out/'baseline_recomputed.json',{'status':'PASS_cached_prediction_arithmetic_only','new_model_evaluations':0,'train_alignment':'not_run','derivative_symmetry_export_checks':'not_repeated','metrics':baseline})
    with (out/'per_structure_residuals.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(residuals[0]));w.writeheader();w.writerows(residuals)
    # Recompute spectral metrics from immutable arrays, not rounded manuscript values.
    ph=WORK/'phonons/results/figure2-six-panels-20261101-02';sha(ph/'figure-data.npz');spectra=np.load(ph/'figure-data.npz',allow_pickle=False);old=js(ph/'summary.json');phon={}
    for xc in XC:
        f0=spectra[xc+'_DFT_bands'];f1=spectra[xc+'_DPA4_bands'];rmse=float(np.sqrt(np.mean((f1-f0)**2)))
        assert np.isclose(rmse,old['results'][xc]['sorted_band_RMSE_THz'],rtol=1e-12)
        phon[xc]={'sorted_band_RMSE_THz':rmse,'DFT_Gamma_THz':f0[0].tolist(),'FT_Gamma_THz':f1[0].tolist(),'DOS_integrals':{role:float(np.trapezoid(spectra[xc+'_'+role+'_DOS'],spectra['frequency_THz'])) for role in ['DFT','DPA4']}}
        for v in phon[xc]['DOS_integrals'].values():assert np.isclose(v,3,atol=1e-8)
    write(out/'phonon_recomputed.json',{'status':'PASS_archived_spectral_arithmetic_only','new_force_evaluations':0,'original_FM_harmonics':'not_measured_in_this_run','results':phon})
    closure=js(mon/'runs/early-stop-20261104/closure.json')
    # Snapshot actual small source/configuration files, not only hashes.
    snap=out/'snapshot';snap.mkdir()
    for folder in [WORK/'manuscript/sections',WORK/'manuscript/supporting_data']:
        for p in folder.glob('*'):
            if p.is_file() and p.suffix in ['.tex','.json','.csv']:
                dest=snap/p.relative_to(WORK);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sha(p)
    for p in [WORK/'manuscript/main.tex',WORK/'manuscript/supporting_materials.tex',ROOT/'protocol/CODEX_TiAl_FM_RESEARCH_PLAN.md']:
        shutil.copyfile(p,snap/p.name);sha(p)
    config_files=list((SOURCE/'DPA4-finetune').glob('*.json'))+list((SOURCE/'DPA4-checkpoints').glob('*/input*.json'))+list((SOURCE/'DPA4-checkpoints').glob('*/out.json'))
    for j,p in enumerate(config_files):shutil.copyfile(p,snap/f'config-{j:02d}-{p.name}');sha(p)
    model_inventory=[]
    for p in list((SOURCE/'DPA4-checkpoints').glob('*.pt'))+list((SOURCE/'DPA4-checkpoints').glob('*/model*.pt'))+list((mon/'models').glob('*/*.pt*')):
        if p.is_file():model_inventory.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    write(out/'models.json',{'registry':registry,'pretrained':pre,'local_files':model_inventory,'harmonic_model_selection':js(WORK/'phonons/config/dpa4-figure-models-20261101.json'),'trainable_parameter_mask':'pending_runtime_inspection'})
    summary={'run_id':a.run_id,'selected':len(frames),'sizes':dict(Counter(map(len,frames))),'matched_label_geometries':mapping_count,'historical_counts':actual_count,'common_label_triples':len(common),'common_historical_train':len(alltrain),'common_historical_validation':len(allval),'known_parent_geometries':sum(bool(r['parent_ids']) for r in records),'unresolved_parent_geometries':sum(not r['parent_ids'] for r in records),'parent_overlap_counts':{k:len(v) for k,v in leaks.items()},'common_atoms':sum(len(frames[int(i.rsplit('-',1)[1])]) for i in common),'labels_by_role':{xc:{s:{'geometries':len(ids),'atoms':sum(len(frames[int(i.rsplit('-',1)[1])]) for i in ids)} for s,ids in splits.items()} for xc,splits in hist.items()},'fixed_external_geometries':len(bundle['records']),'missing_external':{'id':closure['missing_id'],'reason':closure['decision'],'state':'cancelled_before_execution'},'new_DFT_jobs':0,'new_training_jobs':0,'new_model_evaluations':0,'elapsed_seconds':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'geometry_match_tolerance_A':5e-5,'baseline_arithmetic':'PASS','phonon_arithmetic':'PASS','scientific_gates':{'data':'blocked_parent_independence','B0':'partial_cached_raw_reproduction','H0':'partial_FT_only_no_original_FM'}}
    write(out/'summary.json',summary);write(out/'source_hashes.json',SOURCES)
    write(ROOT/'state.json',{'latest_audit_run':str(out),'status':'initial_audit_completed_with_gates_open','new_DFT_jobs':0,'summary':summary})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
