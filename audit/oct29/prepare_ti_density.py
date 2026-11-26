"""Prepare the single-allocation PBE15k pure-Ti temperature scan."""
from pathlib import Path
import hashlib, json, shutil, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'audit/oct20/deps'))
from ase.build import bulk
from ase.io import read, write

OUT = Path(__file__).parent/'pbe15k_ti_density_temperature'
OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists(), 'Preparation is deliberately non-overwriting'
load = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
save = lambda n,d: (OUT/n).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
prior = ROOT/'audit/oct26/pbe15k_density_1873'
old = load(prior/'protocol.json')
eospath = ROOT/'audit/oct25/pbe_property_screen/retry15k/15000/eos.json'
eos = load(eospath)
print({k:list(v) for k,v in eos.items()})
h = eos['Ti_hcp']['equilibrium']; b = eos['Ti_bcc']['equilibrium']
liquidpath = ROOT/'audit/oct25/pbe_full_enthalpy_gpu/15000/results/0/production-final.traj'
a = read(liquidpath); assert len(a)==200 and set(a.get_chemical_symbols())=={'Ti'}
a.calc=None; a.set_momenta([[0,0,0]]*len(a)); write(OUT/'liquid.traj',a)
alpha=bulk('Ti','hcp',a=h['cell_A'][0][0],c=h['cell_A'][2][2]).repeat((5,5,5))
beta=bulk('Ti','bcc',a=b['cell_A'][0][0],cubic=True).repeat((5,5,5))
assert len(alpha)==len(beta)==250
for phase,atoms in [('alpha',alpha),('beta',beta)]:write(OUT/(phase+'.traj'),atoms)
states=[]
for phase,ts in [('liquid',[2500,2400,2300,2200,2100]),('alpha',[300,500,700,900,1100]),('beta',[1200,1400,1600,1800,1900])]:
    for T in ts:
        states.append(dict(id=f'{phase}-{T}',phase=phase,T_K=T,N=200 if phase=='liquid' else 250,
            initial_file=phase+'.traj',initial_sha256=sha(OUT/(phase+'.traj')),
            bulk_modulus_GPa=110.,seed=202610290+len(states)*17,
            equil_steps=30000 if phase=='liquid' else 20000,production_steps=30000))
tasks=[dict(id=s['id'],state=i,checkpoint=15000) for i,s in enumerate(states)]
save('tasks.json',tasks)
cfg=dict(remote='/home6/u3129/tial-testsets/ti-density-pbe15k-20261029-01',
    models={'15000':old['models']['15000']},states=states,tasks_sha256=sha(OUT/'tasks.json'),
    P_bar=1.,dt_fs=1.,ttime_fs=25.,thermo_interval=10,trajectory_interval=100,
    allocation_wall_hours=20,priority='Descending liquid temperature, then alpha, then beta',
    initial_sources={'liquid':{'path':str(liquidpath),'sha256':sha(liquidpath)},
                     'solids':{'path':str(eospath),'sha256':sha(eospath),'equilibria':{'hcp':h,'bcc':b}}},
    protocol_notes=['Each temperature starts independently from its phase-specific source cell.',
      'Liquid source is equilibrated PBE15k Ti at 2000 K, followed by 30 ps at the new temperature.',
      'Crystals are 5x5x5 repetitions of the PBE15k static equilibrium cells.',
      'Isotropic Melchionna NPT; reset velocities and thermostat at production boundary.',
      'Nominal thermostat temperature refers to 3(N-1) kinetic degrees of freedom.',
      'Phase labels describe initialization. Production mobility must be inspected for melting or transformation.'])
save('protocol.json',cfg)
common=(prior/'md_common.py').read_text()
assert '200<T<6000' in common
(OUT/'md_common.py').write_text(common.replace('200<T<6000','20<T<6000'),newline='\n')
job=(prior/'job.sbatch').read_text()
job=job[:job.index('srun --exclusive')]+('srun --exclusive --exact -N1 -n1 -c4 --gres=gpu:a100:1 --mem=28G --cpu-bind=cores python -u workflow.py > worker.log 2>&1\n')
(OUT/'job.sbatch').write_text(job,newline='\n')
for filename in ['workflow.py']:
    shutil.copyfile(Path(__file__).parent/('ti_density_'+filename),OUT/filename)
save('preparation-verification.json',{'states':len(states),'total_md_steps':sum(s['equil_steps']+s['production_steps'] for s in states),
    'N_alpha':len(alpha),'N_beta':len(beta),'N_liquid':len(a),
    'V_alpha_A3_atom':alpha.get_volume()/len(alpha),'V_beta_A3_atom':beta.get_volume()/len(beta),
    'checkpoint_sha256':cfg['models']['15000']['sha256']})
print('Prepared',OUT,len(states),'points')
