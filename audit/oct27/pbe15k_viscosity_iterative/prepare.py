"""Prepare an isolated, reproducible three-composition viscosity calculation."""
from pathlib import Path
import hashlib
import json
import shutil
from collections import Counter

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
BASE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (ROOT/'staging-intent.json').exists(), 'Do not change a staged protocol'
model=json.loads((WORK/'audit/oct26/pbe15k_melting/attempt2/protocol.json').read_text())['checkpoint']
assert sha(model['local_path'])==model['sha256']
states=[]
for i,(x,c,surface) in enumerate([(.20,.80,'cluster'),(.45,.55,'wsl2'),(.80,.20,'cluster')]):
    source=BASE/f'results-scripts-logs/models_check_{surface}/results/viscosity/dpa4/neo/finetune/AlTi/1873K/pbe/01_generate_liquid/c_Ti_{c:.2f}/data.lammps'
    lines=source.read_text().splitlines()
    start=next(j for j,l in enumerate(lines) if l.startswith('Atoms #'))+2
    atoms=[]
    for line in lines[start:]:
        if not line.strip(): break
        atoms.append(line.split())
    counts=Counter(int(row[1]) for row in atoms)
    assert len(atoms)==512 and set(counts)=={1,2}
    assert abs(counts[1]/512-x)<1/512
    name=f'initial_xAl_{x:.2f}.data'
    shutil.copy2(source,ROOT/name)
    states.append(dict(id=f'xAl_{x:.2f}',x_Al=x,c_Ti=c,counts={'Al':counts[1],'Ti':counts[2]},
        actual_x_Al=counts[1]/512,initial_file=name,initial_sha256=sha(source),
        initial_source=str(source),seed=27102601+i*100000))
baseline=[r for r in json.loads((WORK/'audit/oct25/viscosity/viscosity_merged.json').read_text())
          if r['T']==1873 and r['dft']=='pbe' and r['c_Ti'] in [.8,.55,.2]]
assert len(baseline)==3 and all(r['n_replicas']==50 for r in baseline)
cfg=dict(run_id='pbe15k-viscosity-1873-20261027-01',
    remote='/home6/u3129/tial-testsets/pbe15k-viscosity-1873-20261027-01',
    checkpoint=model,
    frozen_model={'path':'/home6/u3129/tial-testsets/pbe15k-melting-20261026-02/model_15000.pt2',
                  'sha256':'1a59af825e47b585d3b49789f65bed0ac1fce0286630f9484f0687cbac3d0f65'},
    states=states,temperature_K=1873,pressure_bar=1,atoms=512,dt_ps=.001,
    initial_npt_steps=20000,reservoir_spacing_steps=5000,nvt_equil_steps=1000,
    production_steps=40000,matched_window_steps=20000,
    preparation_Tdamp_ps=.1,preparation_Pdamp_ps=.5,production_Tdamp_ps=.2,
    masses=[26.98,47.88],elements=['Al','Ti'],batch_size=5,
    allocation={'partition':'v100','node':'tesla-a101','account':'imet-2026-2','qos':'normal',
                'gpu':'a100:1','cpus':6,'md_threads':4,'analysis_threads':2,'memory':'32G','walltime':'20:00:00'},
    convergence={'min_replicas':30,'max_relative_std':.03,'max_last_three_range':.02,
        'max_20_40_relative_difference':.03,'min_neff':60,'max_cost_z':2,'max_criterion_z':2,
        'required_consecutive_passes':2,
        'interpretation':'Operational stopping rule for sampling; nominal uncertainties after sequential inspection are descriptive, not coverage guarantees.'},
    duration_policy='Full 40 ps and matched first 20 ps analyzed separately; never pool historical 5k trajectories.',
    restart_policy='One Slurm job ID, requeued before walltime at a completed-replica boundary; immutable completed replicas are reused.',
    preparation_note='Start from archived 512-atom liquid structures; re-equilibrate under 15k in NPT for 20 ps, then take successive NPT reservoir snapshots 5 ps apart, independently reseed velocities and equilibrate 1 ps NVT. This strengthens preparation relative to historical sampling, so the difference is not a perfectly isolated checkpoint intervention.')
for name,data in [('protocol.json',cfg),('baseline_5k.json',baseline)]:
    (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'states':states,'baseline_count':len(baseline),'checkpoint_sha256':model['sha256']},indent=2))
