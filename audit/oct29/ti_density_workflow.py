"""One allocated A100; ordered, independently initialized temperature points."""
from md_common import *
import subprocess, traceback
from deepmd.pt.utils import env as dp_env

def save(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

unique=[next(s for s in cfg['states'] if s['phase']==p) for p in ['liquid','alpha','beta']]
if '--cpu-reference' in sys.argv:
    assert dp_env.DEVICE.type=='cpu'
    calc=Native(15000);arr={}
    for s in unique:
        a=initial(s,calc);key=s['phase']
        arr[key+'_E']=a.get_potential_energy();arr[key+'_F']=a.get_forces();arr[key+'_S']=a.get_stress()
    np.savez_compressed('cpu_reference.npz',**arr);sys.exit(0)
assert torch.cuda.is_available() and dp_env.DEVICE.type=='cuda'
assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device={'name':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'deepmd_device':str(dp_env.DEVICE)}
subprocess.run([sys.executable,__file__,'--cpu-reference'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','DEVICE':'cpu'},check=True)
ref=np.load('cpu_reference.npz');calc=Native(15000);checks=[];benchmarks={}
for s in unique:
    a=initial(s,calc);k=s['phase']
    d={'phase':k,'E_diff_eV_atom':abs(a.get_potential_energy()-float(ref[k+'_E']))/len(a),
       'F_diff_RMS_eV_A':float(np.sqrt(np.mean((a.get_forces()-ref[k+'_F'])**2))),
       'stress_max_diff_GPa':float(np.max(abs(a.get_stress()-ref[k+'_S']))/units.GPa)}
    assert d['E_diff_eV_atom']<1e-5 and d['F_diff_RMS_eV_A']<1e-4 and d['stress_max_diff_GPa']<.002,d
    checks.append(d);folder=ROOT/'benchmark'/k;folder.mkdir(parents=True,exist_ok=False)
    cfg['T_K']=s['T_K'];benchmarks[k]=stage(a,s,100,folder,'pilot',s['seed'])/100
projected=sum(benchmarks[s['phase']]*(s['equil_steps']+s['production_steps']) for s in cfg['states'])/3600
pre=dict(status='passed',device=device,checks=checks,seconds_per_step=benchmarks,projected_hours=projected,
    model_sha256=cfg['models']['15000']['sha256'],protocol_sha256=sha('protocol.json'),
    versions={p:importlib.metadata.version(p) for p in ['ase','deepmd-kit','torch','numpy']})
save('preflight.json',pre);print('PREFLIGHT',json.dumps(pre),flush=True)
completed=[];failed=[]
for s in cfg['states']:
    folder=ROOT/'results'/s['id'];folder.mkdir(parents=True,exist_ok=False)
    save('progress.json',dict(active=s['id'],completed=completed,failed=failed))
    try:
        cfg['T_K']=s['T_K'];a=initial(s,calc);assert len(a)==s['N']
        print('START',s['id'],flush=True);start=time.monotonic()
        eq=stage(a,s,s['equil_steps'],folder,'equilibration',s['seed'])
        prod=stage(a,s,s['production_steps'],folder,'production',s['seed']+1)
        rows=[json.loads(x) for x in (folder/'production-thermo.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(0,s['production_steps']+1,cfg['thermo_interval']))
        with Trajectory(str(folder/'production.traj')) as tr:
            assert len(tr)==s['production_steps']//cfg['trajectory_interval']+1
            rho=np.array([f.get_masses().sum()/f.get_volume()*1.66053906660 for f in tr])
            np.testing.assert_allclose(rho,[r['rho_g_cm3'] for r in rows[::cfg['trajectory_interval']//cfg['thermo_interval']]],atol=1e-10,rtol=0)
        summary=dict(status='completed',state=s,job_id=os.environ['SLURM_JOB_ID'],device=device,
            model_sha256=cfg['models']['15000']['sha256'],protocol_sha256=sha('protocol.json'),
            worker_sha256=sha(__file__),common_sha256=sha('md_common.py'),
            wall_seconds=time.monotonic()-start,equil_seconds=eq,production_seconds=prod,
            rho_g_cm3=float(np.mean([r['rho_g_cm3'] for r in rows[1:]])))
        save(folder/'summary.json',summary);completed.append(s['id']);print('COMPLETED',s['id'],flush=True)
    except Exception:
        d={'id':s['id'],'traceback':traceback.format_exc()};save(folder/'failure.json',d);failed.append(d)
        print('FAILED',s['id'],d['traceback'],flush=True)
    save('progress.json',dict(active=None,completed=completed,failed=failed))
save('summary.json',dict(status='completed' if not failed else 'failed',completed=completed,failed=failed))
sys.exit(1 if failed else 0)
