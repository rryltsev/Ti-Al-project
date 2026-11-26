"""Final PBEsol/r2SCAN liquid statistics in one A100 allocation."""
from md_common import *
import subprocess, traceback
from deepmd.pt.utils import env as dp_env

def save(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

if '--cpu-reference' in sys.argv:
    assert dp_env.DEVICE.type=='cpu'
    arr={}; calculators={}
    for s in cfg['states']:
        k=s['id'];step=s['checkpoint']
        if step not in calculators:calculators[step]=Native(step)
        a=initial(s,calculators[step])
        arr[k+'_E']=a.get_potential_energy();arr[k+'_F']=a.get_forces();arr[k+'_S']=a.get_stress()
    np.savez_compressed('cpu_reference.npz',**arr);sys.exit(0)
assert torch.cuda.is_available() and dp_env.DEVICE.type=='cuda'
assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=dict(name=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),deepmd_device=str(dp_env.DEVICE))
subprocess.run([sys.executable,__file__,'--cpu-reference'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','DEVICE':'cpu'},check=True)
ref=np.load('cpu_reference.npz');calculators={};checks=[];benchmarks={}
for s in cfg['states']:
    step=s['checkpoint'];k=s['id']
    if step not in calculators:calculators[step]=Native(step)
    a=initial(s,calculators[step])
    d=dict(id=k,E_diff_eV_atom=abs(a.get_potential_energy()-float(ref[k+'_E']))/len(a),
           F_diff_RMS_eV_A=float(np.sqrt(np.mean((a.get_forces()-ref[k+'_F'])**2))),
           stress_max_diff_GPa=float(np.max(abs(a.get_stress()-ref[k+'_S']))/units.GPa),
           archived_energy_diff_eV_atom=abs(a.get_potential_energy()-s['archived_last_energy_eV'])/len(a))
    assert d['E_diff_eV_atom']<1e-5 and d['F_diff_RMS_eV_A']<1e-4 and d['stress_max_diff_GPa']<.002,d
    assert d['archived_energy_diff_eV_atom']<1e-4,('Archived model identity check failed',d)
    checks.append(d)
    if s['functional'] not in benchmarks:
        folder=ROOT/'benchmark'/s['functional'];folder.mkdir(parents=True,exist_ok=False)
        cfg['T_K']=s['T_K'];benchmarks[s['functional']]=stage(a,s,100,folder,'pilot',s['seed'])/100
projected=sum(benchmarks[s['functional']]*(s['equil_steps']+s['production_steps']) for s in cfg['states'])/3600
save('preflight.json',dict(status='passed',device=device,checks=checks,seconds_per_step=benchmarks,projected_hours=projected,
     models_sha256={k:v['sha256'] for k,v in cfg['models'].items()},protocol_sha256=sha('protocol.json'),
     versions={p:importlib.metadata.version(p) for p in ['ase','deepmd-kit','torch','numpy']}))
print('PREFLIGHT PASSED projected_hours',projected,flush=True)
completed=[];failed=[]
for s in cfg['states']:
    folder=ROOT/'results'/s['id'];folder.mkdir(parents=True,exist_ok=False)
    save('progress.json',dict(active=s['id'],completed=completed,failed=failed))
    try:
        cfg['T_K']=s['T_K'];a=initial(s,calculators[s['checkpoint']]);assert len(a)==s['N']
        print('START',s['id'],flush=True);start=time.monotonic()
        eq=stage(a,s,s['equil_steps'],folder,'equilibration',s['seed'])
        prod=stage(a,s,s['production_steps'],folder,'production',s['seed']+1)
        rows=[json.loads(x) for x in (folder/'production-thermo.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(0,s['production_steps']+1,cfg['thermo_interval']))
        with Trajectory(str(folder/'production.traj')) as tr:
            assert len(tr)==s['production_steps']//cfg['trajectory_interval']+1
            rho=np.array([f.get_masses().sum()/f.get_volume()*1.66053906660 for f in tr])
            np.testing.assert_allclose(rho,[r['rho_g_cm3'] for r in rows[::10]],atol=1e-10,rtol=0)
        save(folder/'summary.json',dict(status='completed',state=s,job_id=os.environ['SLURM_JOB_ID'],device=device,
            model_sha256=cfg['models'][str(s['checkpoint'])]['sha256'],protocol_sha256=sha('protocol.json'),
            worker_sha256=sha(__file__),common_sha256=sha('md_common.py'),
            wall_seconds=time.monotonic()-start,equil_seconds=eq,production_seconds=prod,
            rho_g_cm3=float(np.mean([r['rho_g_cm3'] for r in rows[1:]]))))
        completed.append(s['id']);print('COMPLETED',s['id'],flush=True)
    except Exception:
        d=dict(id=s['id'],traceback=traceback.format_exc());save(folder/'failure.json',d);failed.append(d)
        print('FAILED',s['id'],d['traceback'],flush=True)
    save('progress.json',dict(active=None,completed=completed,failed=failed))
save('summary.json',dict(status='completed' if not failed else 'failed',completed=completed,failed=failed))
sys.exit(1 if failed else 0)
