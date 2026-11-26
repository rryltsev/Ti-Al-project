"""Immutable matched-precision benchmark with separate CPU and queued GPU paths."""
from pathlib import Path
import ast,hashlib,json,shutil,tarfile
import numpy as np
from ase.io import read
ROOT=Path(__file__).resolve().parents[1];ID='grace-precision-20261107-02';RUN=ROOT/'runs'/ID;OUT=RUN/'package'
REMOTE='/home6/u3129/tial-fm-research/'+ID
GRACE=ROOT/'runs/grace-2l-preparation-20261107-01'
BASE=ROOT/'runs/dpa4-n694-seeds-20261107-01/package'
ENV='/home6/u3129/mlip-research/envs/grace_ft_a100'

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def xyz(p,atoms,ids=None):
    p.parent.mkdir(parents=True,exist_ok=True);f=lambda x:format(float(x),'.17g')
    with p.open('w') as out:
        for i,at in enumerate(atoms):
            comment='Lattice="'+' '.join(map(f,at.cell.array.ravel()))+'" Properties=species:S:1:pos:R:3 pbc="T T T"'
            if ids:comment+=' geometry_id='+ids[i]
            if at.calc is not None:
                comment=comment.replace('pos:R:3','pos:R:3:forces:R:3');comment+=' energy='+f(at.get_potential_energy())+' stress="'+' '.join(map(f,at.get_stress(voigt=False).ravel(order='F')))+'"'
            out.write(str(len(at))+'\n'+comment+'\n')
            for j,(s,pos) in enumerate(zip(at.get_chemical_symbols(),at.positions)):
                vals=np.r_[pos,at.get_forces()[j]] if at.calc is not None else pos
                out.write(s+' '+' '.join(map(f,vals))+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    c64=json.loads((GRACE/'package/model-card.json').read_text())
    card=dict(model='GRACE-2L-OMAT-medium-base',precision='float32',revision=c64['revision'],source_kind='controlled_cast',
        source_archive_sha256=c64['archive_sha256'],archive_sha256=None,
        provenance='Native fp32 execution built by per-checkpoint-tensor cast of pinned fp64 weights, respecting dtypes of a native fp32 model template. Not a downloaded official fp32 artifact.',
        official_fp32_download='Incomplete due to ~50KB/s transfer; partial file retained, never loaded.')
    source=GRACE/'package/checkpoint'
    (OUT/'checkpoint-fp32').mkdir()
    shutil.copyfile(source/'model.yaml',OUT/'checkpoint-fp32/model.yaml')
    dump(OUT/'model-card-fp32.json',card)
    dump(OUT/'model-card-fp64.json',c64)
    meta={r['geometry_id']:r for r in map(json.loads,(BASE/'data_manifest.jsonl').read_text().splitlines())}
    allats={xc:{at.info['geometry_id']:at for at in read(GRACE/'data'/xc/'train.extxyz',index=':')} for xc in ['PBE','r2SCAN']}
    probes=json.loads((BASE/'probes.json').read_text());chosen=[x['id'] for x in probes]
    alloy=next(sid for sid,at in sorted(allats['PBE'].items()) if len(at)==216 and set(at.get_chemical_symbols())=={'Al','Ti'} and meta[sid]['structural_family']=='unresolved_liquid_relax_pool')
    chosen.append(alloy)
    for family in sorted({meta[sid]['structural_family'] for sid in allats['PBE']}):
        candidate=next(sid for sid in sorted(allats['PBE']) if meta[sid]['structural_family']==family and sid not in chosen)
        chosen.append(candidate)
    chosen=list(dict.fromkeys(chosen));assert len(chosen)==8
    panel=[];panel_ids=[];labels={}
    for xc in ['PBE','r2SCAN']:
        for sid in chosen:
            at=allats[xc][sid];key=xc+'-'+sid;panel.append(at);panel_ids.append(key)
            labels[key]=dict(XC=xc,geometry_id=sid,family=meta[sid]['structural_family'],natoms=len(at),energy=float(at.get_potential_energy()),forces=at.get_forces().tolist(),stress=at.get_stress().tolist())
    xyz(OUT/'inputs/panel.extxyz',panel,panel_ids);dump(OUT/'inputs/panel-labels.json',labels)
    back=read(OUT/'inputs/panel.extxyz',index=':')
    for a,b in zip(panel,back):
        assert np.array_equal(a.positions,b.positions) and np.array_equal(a.get_forces(),b.get_forces()) and a.get_potential_energy()==b.get_potential_energy()
    hcp=read(BASE/'harmonic_inputs/PBE/force000/POSCAR',format='vasp');xyz(OUT/'inputs/timing/Ti96.extxyz',[hcp])
    mixed=allats['PBE'][alloy].copy();mixed.calc=None
    for n,repeat in [(216,(1,1,1)),(432,(2,1,1)),(864,(2,2,1))]:xyz(OUT/'inputs/timing'/f'TiAl{n}.extxyz',[mixed.repeat(repeat)])
    for xc in ['PBE','r2SCAN']:shutil.copytree(BASE/'harmonic_inputs'/xc,OUT/'harmonic_inputs'/xc)
    for name in ['build_pbesol_phonons.py','harmonic_worker.py','reference_figure.npz']:shutil.copyfile(BASE/name,OUT/name)
    for name in ['grace_precision_worker.py','grace_precision_runner.py','analyze_grace_precision.py']:shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    atom,axis=np.unravel_index(np.argmax(abs(allats['PBE'][alloy].get_forces())),(216,3))
    plan=dict(id=ID,model='GRACE-2L-OMAT-medium-base',revision=card['revision'],
        checkpoints={'fp32':dict(path=REMOTE+'/checkpoint-fp32',source_kind='controlled_cast',source_archive_sha256=c64['archive_sha256']),
                     'fp64':dict(path='/home6/u3129/tial-fm-research/grace-2l-preparation-20261107-01/checkpoint',archive_sha256=c64['archive_sha256'],file_hashes={name:hashlib.sha256((GRACE/'package/checkpoint'/name).read_bytes()).hexdigest() for name in ['model.yaml','checkpoint.index','checkpoint.data-00000-of-00001']})},
        panel_source='Eight fixed N200-train IDs, each at native PBE and r2SCAN coordinates; three established probes plus all four structural families. No validation/test used for this numerical comparison.',
        panel_ids=panel_ids,finite_difference=dict(geometry_id='PBE-'+alloy,component=[int(atom),int(axis)],selection='Largest absolute DFT force component; fixed before model predictions'),
        timing_cases={'cpu':['Ti96','TiAl216','TiAl432'],'gpu':['Ti96','TiAl216','TiAl432','TiAl864']},
        timing_order=['fp32-block0','fp64-block0','fp64-block1','fp32-block1'],warmup_calls={'cpu':2,'gpu':5},measured_calls_per_case={'cpu':6,'gpu':30},
        backend='TensorPotential0.6.1/TensorFlow2.20.0; source TPModel ASE diverse engine. Main compute XLA off on CPU, on on GPU. UQ/TF32 off. Fresh processes; CUDA results synchronized by numpy conversion. Not a LAMMPS/Kokkos throughput claim.',
        CPU_threads=4,GPU_count=1,GPU_dependency='afterany:20989407; no GPU overlap with the eight new N694 training repetitions',
        decision='Compare precision-induced E/F/stress and phonon changes to the DFT model error, then weigh measured throughput/memory. Keep GRACE precision provisional until GPU measurements.',
        no_new_DFT=True,no_training=True,test_access=False)
    dump(OUT/'plan.json',plan)
    common='''#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd REMOTE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_USE_LEGACY_KERAS=1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache/$SLURM_JOB_ID" GRACE_CACHE="$PWD/cache/grace"
export TF_FORCE_GPU_ALLOW_GROWTH=true
'''.replace('REMOTE',REMOTE)
    cpu=common+'export CUDA_VISIBLE_DEVICES=""\nsrun --cpu-bind=cores '+ENV+'/bin/python grace_precision_runner.py --device cpu\n'
    gpu=common+'export GRACE_CUDA_ROOT='+ENV+'/lib/python3.11/site-packages/nvidia/cuda_nvcc\nexport XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"\nexport PATH="$GRACE_CUDA_ROOT/bin:$PATH"\nsrun --cpu-bind=cores '+ENV+'/bin/python grace_precision_runner.py --device gpu\n'
    for device,body in [('cpu',cpu),('gpu',gpu)]:(OUT/(device+'.sbatch')).write_text(body,newline='\n')
    analysis=common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python analyze_grace_precision.py --device "$1"\n'
    (OUT/'analysis.sbatch').write_text(analysis,newline='\n')
    for p in OUT.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
    arc=RUN/'package.tar.gz'
    with tarfile.open(arc,'w:gz') as t:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(arc),sha256=hashlib.sha256(arc.read_bytes()).hexdigest(),bytes=arc.stat().st_size))
    print(json.dumps(dict(status='PREPARED',panel=len(panel),bytes=arc.stat().st_size)))

if __name__=='__main__':main()
