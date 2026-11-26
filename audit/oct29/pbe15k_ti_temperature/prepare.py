"""Prepare a separately versioned pure-Ti temperature campaign; no submission."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
OLD=WORK/'audit/oct27/pbe15k_viscosity_iterative'
BASE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project/results-scripts-logs')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (ROOT/'staging-complete.json').exists()
cfg=json.loads((OLD/'protocol.json').read_text())
cfg['run_id']='pbe15k-ti-temperature-20261029-01'
cfg['remote']='/home6/u3129/tial-testsets/'+cfg['run_id']
cfg.pop('temperature_K')
cfg['states']=[]
for k,(t,surface) in enumerate([(2100,'cluster'),(1873,'wsl2'),(2000,'wsl2')]):
    source=BASE/f'models_check_{surface}/results/viscosity/dpa4/neo/finetune/AlTi/{t}K/pbe/01_generate_liquid/c_Ti_1.00/data.lammps'
    lines=source.read_text().splitlines();start=next(i for i,x in enumerate(lines) if x.startswith('Atoms #'))+2
    atoms=[]
    for line in lines[start:]:
        if not line.strip():break
        atoms.append(line.split())
    assert len(atoms)==512 and Counter(int(x[1]) for x in atoms)=={2:512}
    name=f'initial_Ti_{t}K.data';shutil.copy2(source,ROOT/name)
    cfg['states'].append(dict(id=f'Ti_{t}K',temperature_K=t,x_Al=0.,c_Ti=1.,actual_x_Al=0.,counts={'Al':0,'Ti':512},
        initial_file=name,initial_sha256=sha(source),initial_source=str(source),seed=29102601+k*100000))
cfg['scheduling']={'priority_temperatures_K':[2100,1873,2000],'initial_replicas_each':10,'batch_size':5,
 'policy':'First obtain 5 and 10 new replicas at 2100K, then 10 at1873K, then10 at2000K. Continue five-replica rounds in that order, skipping converged or review-held temperatures.'}
cfg['convergence']['max_new_batch_z']=2.5
cfg['convergence']['review_at_or_above_replicas']=50
cfg['duration_policy']='New 40ps replicas and first/second20ps fits; historical 5k and exported15k estimates remain separate, with no pooling.'
cfg['preparation_note']='Archived pure-liquid Ti structure at each T, re-equilibrated under verified15k for20ps NPT; fresh5ps reservoir spacing and1ps NVT,40ps production. Historical15k raw provenance unavailable; new protocol version and sources remain separate.'
cfg['restart_policy']='Reuse existing Slurm20962153 through a guarded dispatcher and requeue. One A100, sequential temperatures, immutable completed replicas. Future requeue at replica boundary.'
cfg['inherited_job_id']='20962153'
(ROOT/'protocol.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
old=json.loads((WORK/'audit/oct25/viscosity/viscosity_merged.json').read_text())
baseline=[r for r in old if r['dft']=='pbe' and r['c_Ti']==1]
comparison=json.loads((WORK/'audit/oct27/viscosity_5k_vs_15k/comparison.json').read_text())
known=[]
for item in comparison['comparison']:
    r=dict(item['checkpoint_15k']);source=Path(item['source_15k'])
    assert json.loads(source.read_text())==r
    known.append(dict(r,checkpoint_step=15000,source=str(source),source_sha256=sha(source),
                      provenance='Supplied15k export; raw trajectories/model hashes absent from export; kept separate from new run.'))
for name,data in [('baseline_5k.json',baseline),('known_15k.json',known)]:
    (ROOT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
# Keep provenance and reproducible calculation primitives from the checked alloy run.
src=(OLD/'workflow.py').read_text(encoding='utf-8').split('def finish_analysis(')[0]
src=src.replace('velocity all create 1873 {s[\'seed\']}','velocity all create {s[\'temperature_K\']} {s[\'seed\']}')
src=src.replace('fix bench all nvt temp 1873 1873 0.2',"fix bench all nvt temp {s['temperature_K']} {s['temperature_K']} 0.2")
src=src.replace('def npt(data,folder,steps,seed,tdamp=.1):','def npt(data,folder,steps,seed,temperature_K,tdamp=.1):')
src=src.replace('velocity all create 1873 {seed}', 'velocity all create {temperature_K} {seed}')
src=src.replace('fix prepare all npt temp 1873 1873 {tdamp}', 'fix prepare all npt temp {temperature_K} {temperature_K} {tdamp}')
src=src.replace("CFG['initial_npt_steps'],s['seed'])", "CFG['initial_npt_steps'],s['seed'],s['temperature_K'])")
src=src.replace("CFG['reservoir_spacing_steps'],seed)", "CFG['reservoir_spacing_steps'],seed,s['temperature_K'])")
src=src.replace('velocity all create 1873 {seed+1}',"velocity all create {s['temperature_K']} {seed+1}")
src=src.replace('fix equilibrate all nvt temp 1873 1873 0.1',"fix equilibrate all nvt temp {s['temperature_K']} {s['temperature_K']} 0.1")
src=src.replace('fix sample all nvt temp 1873 1873 0.2',"fix sample all nvt temp {s['temperature_K']} {s['temperature_K']} 0.2")
src=src.replace("result=dict(status='completed',composition=s['id'],replica=i,seed=seed+1,", "result=dict(status='completed',composition=s['id'],temperature_K=s['temperature_K'],replica=i,seed=seed+1,")
assert '1873' not in src
(ROOT/'md_core.py').write_text(src,encoding='utf-8')
common=(OLD/'analyze_batches.py').read_text(encoding='utf-8').split('def plot_history(')[0]
common=common.replace("row['new_batch_z']<=2.5", "row['new_batch_z']<=c['max_new_batch_z']")
(ROOT/'analysis_common.py').write_text(common,encoding='utf-8')
tests=(OLD/'test_analysis.py').read_text().replace('import analyze_batches as a','import analysis_common as a')
tests=tests.replace("['workflow.py','analyze_batches.py']","['workflow.py','md_core.py','analyze_point.py','analysis_common.py']")
(ROOT/'test_analysis.py').write_text(tests,encoding='utf-8')
(ROOT/'submission.json').write_text(json.dumps(dict(job_id='20962153',inherited=True,previous_remote=json.loads((OLD/'protocol.json').read_text())['remote']),indent=2)+'\n')
before=ROOT/'before';before.mkdir(exist_ok=True)
for path in [WORK/'manuscript/figures/viscosity.pdf',WORK/'manuscript/sections/results.tex',WORK/'manuscript/supporting_data/viscosity_current_oct29.json',OLD/'workflow.py',OLD/'monitor.json']:
    dest=before/path.name
    if not dest.exists():shutil.copy2(path,dest)
(ROOT/'preparation_manifest.json').write_text(json.dumps(dict(source_md_sha256=sha(OLD/'workflow.py'),source_analysis_sha256=sha(OLD/'analyze_batches.py'),known_points=len(known),baseline_points=len(baseline),protocol_sha256=sha(ROOT/'protocol.json')),indent=2)+'\n')
print(json.dumps({'temperatures':[s['temperature_K'] for s in cfg['states']],'known15k':known,'baseline5k_temperatures':[r['T'] for r in baseline]},ensure_ascii=True))
