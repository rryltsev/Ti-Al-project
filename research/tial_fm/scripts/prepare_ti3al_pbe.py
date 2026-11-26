"""Freeze Ti3Al source evidence and all completed PBE model identities."""
from pathlib import Path
import hashlib, json, shutil, tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/ti3al-pbe-phonons-20261109-02'
REMOTE='/home6/u3129/tial-fm-research/'+RUN.name
SOURCE=Path('C:/Users/Roman Ryltsev/Desktop/STARTUPS/Ti-Al-X/TiAl-Project/results-scripts-logs/Codex_DFT_Ti_Ti3Al_20261012/Codex_DFT_Ti_Ti3Al_20261012')
def load(p): return json.loads(p.read_text(encoding='utf8'))
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pkg=RUN/'package';pkg.mkdir(parents=True,exist_ok=False)
    source=pkg/'source';source.mkdir()
    manifest=load(SOURCE/'manifest.json')
    for item in manifest['files']:
        name=item['path']
        if name.startswith('Ti3Al_D019/') or name in ['DFT_Ti_Ti3Al_reference.json','README_RU.txt']:
            assert sha(SOURCE/name)==item['sha256'],name
            target=source/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(SOURCE/name,target)
    shutil.copyfile(SOURCE/'manifest.json',source/'manifest.json')
    models=[]
    for rid,col in [('dpa4-n200-pilot-20261106-02','collected-pilots'),('dpa4-replicates-full-20261106-02','collected'),('dpa4-n694-seeds-20261107-01','collected')]:
        for f in sorted((ROOT/'runs'/rid/col/'training-plan01/cases').glob('PBE*/training-summary.json')):
            s=load(f);c=s['case'];e=load(f.parent/'evaluation/summary.json');assert s['status']==e['status']=='PASS'
            name=f'DPA4-{c["method"]}-N{c["N"]}-seed{c["seed"]}'
            models.append(dict(name=name,backend='DPA4',method=c['method'],N=c['N'],seed=c['seed'],XC='PBE',model=s['final_checkpoint'],sha256=e['model_sha256'],source_run=rid,summary_sha256=sha(f)))
    for rid in ['grace-n200-fp64-pilot-20261109-02','grace-n200-mask-ablation-20261109-01']:
        run=ROOT/'runs'/rid;remote='/home6/u3129/tial-fm-research/'+rid
        for c in load(run/'package/plan.json')['cases']:
            if c['XC']!='PBE':continue
            w=run/'collected-final/cases'/c['name'];s=load(w/'training-summary.json');assert s['status']=='PASS'
            models.append(dict(name='GRACE-'+c['method']+'-N200-seed10',backend='GRACE',method=c['method'],N=200,seed=10,XC='PBE',model=remote+'/cases/'+c['name'],hashes=s['hashes'],yaml_sha256=sha(w/'seed/10/model.yaml'),source_run=rid,summary_sha256=sha(w/'training-summary.json')))
    assert len(models)==16 and len({m['name'] for m in models})==16
    dump(pkg/'models.json',models)
    # Composition screening uses metadata only; no held-out predictions or labels.
    records=[json.loads(s) for s in (ROOT/'datasets/common992-grouped-v1-20261106-02/data_manifest.jsonl').read_text().splitlines()]
    candidates=[r for r in records if r['split'] in ['train','validation'] and r['composition'].get('Ti',0)==3*r['composition'].get('Al',0)]
    geometries=[]
    for r in candidates:
        label=r['labels']['PBE'];f=Path(label['coord_path']);i=label['frame_index']
        pos=np.load(f)[i].reshape(-1,3);cell=np.load(f.parent/'box.npy')[i].reshape(3,3)
        types=np.loadtxt(f.parent.parent/'type.raw',dtype=int,ndmin=1);names=(f.parent.parent/'type_map.raw').read_text().split()
        geometries.append(dict(id=r['geometry_id'],split=r['split'],family=r['structural_family'],cell=cell.tolist(),positions=pos.tolist(),symbols=[names[t] for t in types]))
    dump(pkg/'overlap-candidates.json',geometries)
    dump(pkg/'protocol.json',dict(XC='PBE',phase='Ti3Al archive geometry (Cmcm; nominal D019)',geometry='Fixed archived cell; no relaxation',supercell=[2,2,2],displacement_scales=[.5,1.,2.],base_displacement_A=.01,mesh=[14,14,14],mesh_gamma_center=False,DOS='tetrahedron; modes per atom; archived frequency grid plus extended grid',force_constants='traditional raw primary; ASR sensitivity separately',source_force_constants_available=False,DFT_reference='archived total and projected DOS only',model_count=16,training_changes=False,test_access=False,qualification='Development property diagnostic; not independent holdout. FM pretraining overlap unknown.',max_GPU_concurrency=1))
    for name in ['ti3al_phonon_worker.py','ti3al_force_worker.py','grace_pilot_common.py']:
        shutil.copyfile(ROOT/'scripts'/name,pkg/name)
    common=f'''#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd {REMOTE}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
'''
    (pkg/'cpu.sbatch').write_text(common+'''export CUDA_VISIBLE_DEVICES=""
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python ti3al_phonon_worker.py "$1"
''',encoding='utf8',newline='\n')
    gpu=common+'''export TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export GRACE_CUDA_ROOT=/home6/u3129/mlip-research/envs/grace_ft_a100/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"
export PATH="$GRACE_CUDA_ROOT/bin:$PATH"
export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
INDEX=$SLURM_ARRAY_TASK_ID
if [ "$INDEX" -lt 12 ]; then
  PY=/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python
else
  PY=/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python
fi
"$PY" ti3al_force_worker.py "$INDEX"
'''
    (pkg/'forces.sbatch').write_text(gpu,encoding='utf8',newline='\n')
    dump(pkg/'FILES_SHA256.json',{p.relative_to(pkg).as_posix():sha(p) for p in pkg.rglob('*') if p.is_file()})
    archive=RUN/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in pkg.rglob('*'):
            if p.is_file():t.add(p,arcname=p.relative_to(pkg).as_posix())
    dump(RUN/'package.json',dict(path=str(archive),sha256=sha(archive),remote_root=REMOTE))
    state=load(ROOT/'state.json')
    state['testing_scope']=dict(active_XC=['PBE'],paused_XC=['r2SCAN'],reason='User request: expand Ti3Al property tests using PBE only.',existing_results_preserved=True)
    state['ti3al_phonons']=dict(run=str(RUN),status='prepared',models=16,DFT_reference='archived DOS',missing_reference='FORCE_SETS unavailable to u3129; requested local copy')
    dump(ROOT/'state.json',state)
    print('PREPARED',len(models),'models;',len(candidates),'composition-matched train/validation structures')
if __name__=='__main__':main()
