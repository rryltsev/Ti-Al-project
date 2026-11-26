"""Resume only a scheduler timeout; validate and retain all completed inference systems."""
from pathlib import Path
import json,os,subprocess,time
import numpy as np
from ase.io import read
from deepmd.infer import DeepPot
from baseline_worker import sha,dump,evaluate
ROOT=Path(__file__).resolve().parent
def main():
    assert os.environ.get('SLURM_JOB_ID')
    out=ROOT/'baseline';parent='20988440'
    if (out/'summary.json').exists():
        assert json.loads((out/'summary.json').read_text())['status']=='PASS';print('B0 already complete; no repeated evaluations');return
    accounting=subprocess.check_output(['sacct','-n','-X','-j',parent,'--format=State','-P'],text=True).strip()
    assert accounting=='TIMEOUT',('Only the diagnosed time ceiling permits automatic continuation; inspect other failures',accounting)
    cfg=json.loads((ROOT/'protocol.json').read_text());assert sha(cfg['model']['path'])==cfg['model']['sha256']
    for name,h in json.loads((ROOT/'FILES_SHA256.json').read_text()).items():assert sha(ROOT/name)==h
    checks=json.loads((out/'numerical_checks.json').read_text());assert all(x['pass_checks'] for x in checks)
    assert json.loads((out/'export_checks.json').read_text())['status']=='PASS'
    dp=DeepPot(cfg['model']['path']);retained={};computed=0;start=time.monotonic()
    for xc in ['PBE','PBEsol','r2SCAN']:
        for part in ['train','validation','test']:
            target=out/'predictions'/xc/part;target.mkdir(parents=True,exist_ok=True)
            for src in sorted((ROOT/'data'/xc/part).iterdir()):
                ids=json.loads((src/'geometry_ids.json').read_text());types=np.loadtxt(src/'type.raw',dtype=int,ndmin=1);names=(src/'type_map.raw').read_text().split();s=[names[t] for t in types]
                dest=target/(src.name+'.npz')
                if dest.exists():
                    try:
                        x=np.load(dest,allow_pickle=False)
                        assert list(x['ids'])==ids and x['force'].shape==(len(ids),len(types),3) and x['virial'].shape==(len(ids),3,3) and x['energy'].shape==(len(ids),)
                        assert all(np.isfinite(x[k]).all() for k in ['energy','force','virial']);x.close()
                    except Exception as error:
                        archive=out/'interrupted-outputs'/xc/part/dest.name;archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists()
                        dest.rename(archive);dump(archive.with_suffix('.reason.json'),dict(reason=repr(error),original_job=parent))
                    else:retained[dest.relative_to(ROOT).as_posix()]=sha(dest);continue
                es=[];fs=[];vs=[]
                for c,b in zip(np.load(src/'set.000/coord.npy'),np.load(src/'set.000/box.npy')):
                    e,f,v=evaluate(dp,c,b,s);es.append(e);fs.append(f);vs.append(v);computed+=1
                temporary=dest.with_suffix('.partial.npz');np.savez_compressed(temporary,ids=ids,energy=es,force=fs,virial=vs);temporary.rename(dest)
            print('complete',xc,part,'new_frames',computed,flush=True)
    harmonic={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        harmonic[xc]={}
        for key in ['force000','force001','force002']:
            source=ROOT/'harmonic_inputs'/xc/key/'POSCAR';a=read(source,format='vasp');e,f,v=evaluate(dp,a.positions,a.cell.array,a.get_chemical_symbols())
            dest=out/'harmonic_forces'/xc/key;dest.mkdir(parents=True,exist_ok=False);np.save(dest/'forces.npy',f)
            harmonic[xc][key]=dict(POSCAR_sha256=sha(source),energy_eV=e,virial_eV=v.tolist(),max_force_eV_A=float(np.linalg.norm(f,axis=1).max()))
    dump(out/'harmonic_predictions.json',harmonic)
    dump(out/'continuation.json',dict(original_job=parent,original_state=accounting,job_id=os.environ['SLURM_JOB_ID'],retained_system_files_sha256=retained,new_evaluated_dataset_frames=computed,seconds=time.monotonic()-start,worker_sha256=sha(__file__)))
    dump(out/'summary.json',dict(status='PASS',job_id=os.environ['SLURM_JOB_ID'],parent_job_id=parent,model=cfg['model'],evaluated_frames_per_XC=498,harmonic_frames=9,numerical_checks_reused_from_original_job=True,export_reused_from_original_job=True,qualification='A scheduler timeout split the native evaluations across two jobs; completed systems were validated and their hashes retained. All labels and input geometries remain the same.'))
if __name__=='__main__':main()
