"""One GPU production loop with asynchronous CPU analysis and immutable replicas."""
from pathlib import Path
import fcntl,hashlib,importlib.metadata,json,os,subprocess,sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=time.monotonic()
ANALYSIS=None
ANALYSIS_LOG=None
DEVICE=None

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def status(phase,**extra):
    save(ROOT/'progress.json',dict(status='running',phase=phase,job_id=os.environ['SLURM_JOB_ID'],
        unix=time.time(),**extra))

def check_device():
    import torch
    from deepmd.pt.utils import env
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert env.DEVICE.type=='cuda'
    assert 'A100' in torch.cuda.get_device_name(0)
    return dict(name=torch.cuda.get_device_name(0),visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                job_id=os.environ['SLURM_JOB_ID'])

def make_lammps(data,folder):
    from lammps import lammps
    import deepmd
    L=lammps(cmdargs=['-log',str(folder/'log.lammps'),'-screen',str(folder/'screen.txt')])
    if not L.has_style('pair','deepmd'):
        L.command('plugin load '+str(Path(deepmd.__file__).parent/'lib/libdeepmd_lmpplugin.so'))
    assert L.has_style('pair','deepmd')
    cmd=f'''units metal
boundary p p p
atom_style atomic
atom_modify map yes
read_data {data} nocoeff
mass 1 {CFG['masses'][0]}
mass 2 {CFG['masses'][1]}
pair_style deepmd {CFG['frozen_model']['path']}
pair_coeff * * Al Ti
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
timestep {CFG['dt_ps']}
thermo 100
thermo_style custom step time atoms temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz
thermo_modify flush yes lost error
'''
    (folder/'setup.in').write_text(cmd);L.commands_string(cmd)
    return L

def attempt(parent):
    parent.mkdir(parents=True,exist_ok=True)
    i=1
    while (parent/f'attempt_{i:03d}').exists():i+=1
    dest=parent/f'attempt_{i:03d}';dest.mkdir();return dest

def required_type_indices(type_map,elements):
    # DPA4 retains the full periodic-table type map after binary fine-tuning.
    names=list(type_map)
    assert len(names)==len(set(names)), 'Duplicate elements in model type map'
    missing=set(elements)-set(names)
    assert not missing, f'Model type map is missing required elements: {sorted(missing)}'
    return {el:names.index(el) for el in elements}

def preflight():
    global DEVICE
    DEVICE=check_device()
    for name in ['checkpoint','frozen_model']:
        assert sha(CFG[name]['path'])==CFG[name]['sha256']
    for s in CFG['states']:assert sha(ROOT/s['initial_file'])==s['initial_sha256']
    if (ROOT/'preflight.json').exists():
        pre=json.loads((ROOT/'preflight.json').read_text())
        assert pre['protocol_sha256']==sha(ROOT/'protocol.json') and pre['status']=='passed'
        return pre
    import torch
    from ase.io import read
    from deepmd.infer import DeepPot
    meta=torch.load(CFG['checkpoint']['path'],map_location='cpu',weights_only=True)['model']['_extra_state']
    assert meta['train_infos']['step']==14999
    dp=DeepPot(CFG['checkpoint']['path'])
    type_map=list(dp.get_type_map())
    type_indices=required_type_indices(type_map,CFG['elements'])
    assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
    checks=[];seconds=[]
    for s in CFG['states']:
        status('preflight',composition=s['id'])
        folder=attempt(ROOT/'preflight'/s['id']);data=ROOT/s['initial_file']
        a=read(data,format='lammps-data',atom_style='atomic',Z_of_type={1:13,2:22})
        assert len(a)==512 and sum(a.numbers==13)==s['counts']['Al']
        e,f,v=dp.eval(a.positions.reshape(1,-1),a.cell.array.reshape(1,9),
                      [type_indices[el] for el in a.get_chemical_symbols()])[:3]
        L=make_lammps(data,folder)
        try:
            L.commands_string('velocity all set 0 0 0\nrun 0')
            ff=np.ctypeslib.as_array(L.gather_atoms('f',1,3),shape=(1536,)).copy().reshape(-1,3)
            p=np.array([L.get_thermo(k) for k in ['pxx','pyy','pzz','pxy','pxz','pyz']])
            vref=v.reshape(3,3).flat[[0,4,8,1,2,5]]/a.get_volume()*1602176.6208
            d=dict(composition=s['id'],E_diff_eV_atom=abs(L.get_thermo('pe')-float(e.flat[0]))/512,
                F_diff_rms_eV_A=float(np.sqrt(np.mean((ff-f.reshape(-1,3))**2))),
                pressure_diff_max_GPa=float(np.max(abs(p-vref)))/10000)
            assert d['E_diff_eV_atom']<1e-5 and d['F_diff_rms_eV_A']<1e-4 and d['pressure_diff_max_GPa']<.002,d
            checks.append(d)
            L.commands_string(f"velocity all create {s['temperature_K']} {s['seed']} dist gaussian mom yes\nfix bench all nvt temp {s['temperature_K']} {s['temperature_K']} 0.2\nrun 10")
            t=time.monotonic();L.command('run 100');seconds.append((time.monotonic()-t)/100)
        finally:L.close()
    pre=dict(status='passed',device=DEVICE,checks=checks,seconds_per_step=seconds,
        native_type_map=type_map,native_type_indices=type_indices,
        estimated_batch_hours=sum(seconds)*5*46000/3600,
        checkpoint_sha256=CFG['checkpoint']['sha256'],frozen_sha256=CFG['frozen_model']['sha256'],
        protocol_sha256=sha(ROOT/'protocol.json'),
        versions={k:importlib.metadata.version(k) for k in ['deepmd-kit','torch','numpy','ase']})
    save(ROOT/'preflight.json',pre);print('PREFLIGHT',json.dumps(pre),flush=True)
    return pre

def npt(data,folder,steps,seed,temperature_K,tdamp=.1):
    L=make_lammps(data,folder)
    try:
        cmd=f'''reset_timestep 0
velocity all create {temperature_K} {seed} dist gaussian mom yes
fix prepare all npt temp {temperature_K} {temperature_K} {tdamp} iso 1.0 1.0 0.5
run {steps}
write_data {folder}/reservoir.data
'''
        (folder/'run.in').write_text(cmd);L.commands_string(cmd)
        assert L.get_thermo('atoms')==512 and np.isfinite(L.get_thermo('pe'))
        d=dict(temperature_K=L.get_thermo('temp'),volume_A3=L.get_thermo('vol'),steps=steps,
               reservoir_file=str((folder/'reservoir.data').relative_to(ROOT)))
        save(folder/'result.json',d);return d
    finally:L.close()

def prepare(s):
    dest=ROOT/'states'/s['id'];marker=dest/'prepared.json'
    if marker.exists():return json.loads(marker.read_text())
    status('equilibration',composition=s['id'])
    folder=attempt(dest/'preparation')
    result=npt(ROOT/s['initial_file'],folder,CFG['initial_npt_steps'],s['seed'],s['temperature_K'])
    save(marker,result);return result

def completed(s):
    out=[];p=ROOT/'states'/s['id']/'replicas'
    for r in sorted(p.glob('replica_*/summary.json')):
        obj=json.loads(r.read_text());assert obj['replica']==len(out)+1
        assert obj['protocol_sha256']==sha(ROOT/'protocol.json')
        assert (ROOT/obj['stress_file']).exists();out.append(obj)
    return out

def replica(s,i):
    done=completed(s);assert len(done)==i-1
    source=(done[-1] if done else prepare(s))['reservoir_file']
    dest=ROOT/'states'/s['id']/'replicas'/f'replica_{i:04d}'
    folder=attempt(dest);start=time.monotonic()
    status('replica_preparation',composition=s['id'],replica=i)
    seed=s['seed']+10*i
    reservoir=npt(ROOT/source,folder,CFG['reservoir_spacing_steps'],seed,s['temperature_K'])
    prod=folder/'production';prod.mkdir()
    L=make_lammps(ROOT/reservoir['reservoir_file'],prod)
    try:
        # Fresh thermostat and velocities for each NVT trajectory; deterministic unique seed.
        cmd=f'''reset_timestep 0
velocity all create {s['temperature_K']} {seed+1} dist gaussian mom yes
fix equilibrate all nvt temp {s['temperature_K']} {s['temperature_K']} 0.1
run {CFG['nvt_equil_steps']}
unfix equilibrate
write_data {prod}/mol.data
reset_timestep 0
fix sample all nvt temp {s['temperature_K']} {s['temperature_K']} 0.2
variable pxx equal pxx
variable pyy equal pyy
variable pzz equal pzz
variable pxy equal pxy
variable pxz equal pxz
variable pyz equal pyz
run 0
'''
        (prod/'run.in').write_text(cmd);L.commands_string(cmd)
        volume=L.get_thermo('vol')
        initial=np.array([0]+[L.get_thermo(k) for k in ['pxx','pyy','pzz','pxy','pxz','pyz']])
        status('production',composition=s['id'],replica=i,production_ps=40)
        cmd=f'''fix pressure all ave/time 1 1 1 v_pxx v_pyy v_pzz v_pxy v_pxz v_pyz file {prod}/pressure.data
restart 10000 {prod}/restart.a {prod}/restart.b
run {CFG['production_steps']}
write_restart {prod}/final.restart
write_data {prod}/final.data
'''
        with (prod/'run.in').open('a') as f:f.write(cmd)
        L.commands_string(cmd)
        assert L.get_thermo('atoms')==512 and np.isfinite(L.get_thermo('pe'))
        raw=np.loadtxt(prod/'pressure.data',comments='#')
        if raw[0,0]==1:raw=np.vstack([initial,raw])
        assert raw.shape==(40001,7) and np.array_equal(raw[:,0],np.arange(40001))
        assert np.all(np.isfinite(raw)) and abs(L.get_thermo('vol')/volume-1)<1e-12
        np.savez_compressed(prod/'stress.npz',pressure_bar=raw[:,1:],volume_A3=volume,dt_ps=.001)
        result=dict(status='completed',composition=s['id'],temperature_K=s['temperature_K'],replica=i,seed=seed+1,
            reservoir_file=reservoir['reservoir_file'],stress_file=str((prod/'stress.npz').relative_to(ROOT)),
            stress_sha256=sha(prod/'stress.npz'),n_samples=len(raw),volume_A3=volume,
            final_temperature_K=L.get_thermo('temp'),wall_seconds=time.monotonic()-start,
            checkpoint_sha256=CFG['checkpoint']['sha256'],protocol_sha256=sha(ROOT/'protocol.json'),device=DEVICE)
        save(dest/'summary.json',result)
        print('REPLICA_COMPLETED',s['id'],i,round(result['wall_seconds'],1),flush=True)
    finally:L.close()

