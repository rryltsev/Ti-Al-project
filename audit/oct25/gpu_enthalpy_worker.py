"""One GPU, one checkpoint, all compositions without further queue submissions."""
from md_common import *
import subprocess,traceback
step=int(sys.argv[1]);reference='--cpu-reference' in sys.argv
from deepmd.pt.utils import env as dp_env
outroot=ROOT/str(step);outroot.mkdir(exist_ok=True)
if reference:
    assert dp_env.DEVICE.type=='cpu'
    calc=Native(step);arr={}
    for i,state in enumerate(cfg['states']):
        a=initial(state,calc);arr[f'E_{i}']=a.get_potential_energy();arr[f'F_{i}']=a.get_forces();arr[f'S_{i}']=a.get_stress()
    np.savez_compressed(outroot/'cpu_reference.npz',**arr)
    sys.exit(0)
assert torch.cuda.is_available() and dp_env.DEVICE.type=='cuda'
assert torch.cuda.device_count()==1,('Expected one allocated GPU per worker',os.environ.get('CUDA_VISIBLE_DEVICES'))
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert 'A100' in torch.cuda.get_device_name(0)
device={'name':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'deepmd_device':str(dp_env.DEVICE)}
print('GPU_WORKER',step,device,flush=True)
cpuenv={**os.environ,'CUDA_VISIBLE_DEVICES':'','DEVICE':'cpu'}
subprocess.run([sys.executable,__file__,str(step),'--cpu-reference'],env=cpuenv,check=True)
ref=np.load(outroot/'cpu_reference.npz');calc=Native(step);checks=[]
for i,state in enumerate(cfg['states']):
    a=initial(state,calc);E=a.get_potential_energy();F=a.get_forces();S=a.get_stress()
    d={'composition':state['composition'],'E_diff_eV_atom':abs(E-float(ref[f'E_{i}']))/len(a),
       'F_diff_RMS_eV_A':float(np.sqrt(np.mean((F-ref[f'F_{i}'])**2))),'stress_max_diff_GPa':float(np.max(abs(S-ref[f'S_{i}']))/units.GPa)}
    assert d['E_diff_eV_atom']<1e-5 and d['F_diff_RMS_eV_A']<1e-4 and d['stress_max_diff_GPa']<.002,d
    checks.append(d)
bench=outroot/'benchmark';bench.mkdir(exist_ok=False);state=next(s for s in cfg['states'] if s['x_Al']==.5)
a=initial(state,calc);seconds=stage(a,state,100,bench,'pilot',state['seed'])
projected=seconds/100*60000*11/3600
pre={'status':'passed','device':device,'checks':checks,'seconds_per_100_steps':seconds,'projected_series_hours':projected,
     'protocol_sha256':sha('protocol.json'),'model_sha256':cfg['models'][str(step)]['sha256'],
     'versions':{p:importlib.metadata.version(p) for p in ['ase','deepmd-kit','torch','numpy']}}
(outroot/'preflight.json').write_text(json.dumps(pre,indent=2));print('PREFLIGHT',step,json.dumps(pre),flush=True)
assert projected<cfg['allocation_wall_hours']*.85,('Projected series too long; review performance before MD',projected)
failed=[];completed=[]
for task in [t for t in tasks if t['checkpoint']==step]:
    state=cfg['states'][task['state']];out=outroot/'results'/str(task['id']);out.mkdir(parents=True,exist_ok=False)
    try:
        a=initial(state,calc);start=time.monotonic();print('START_COMPOSITION',step,state['composition'],flush=True)
        eq=stage(a,state,cfg['equil_steps'],out,'equilibration',state['seed'])
        prod=stage(a,state,cfg['production_steps'],out,'production',state['seed']+1)
        summary={'status':'completed','task':task,'composition':state['composition'],'model_sha256':cfg['models'][str(step)]['sha256'],
          'protocol_sha256':sha('protocol.json'),'worker_sha256':sha(__file__),'common_sha256':sha('md_common.py'),
          'job_id':os.environ['SLURM_JOB_ID'],'device':device,'wall_seconds':time.monotonic()-start,'equil_seconds':eq,'production_seconds':prod}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));completed.append(task['id'])
        print('COMPLETED_COMPOSITION',step,state['composition'],flush=True)
    except Exception:
        failure={'task':task,'traceback':traceback.format_exc()};(out/'failure.json').write_text(json.dumps(failure,indent=2));failed.append(failure)
        print('FAILED_COMPOSITION',step,state['composition'],failure['traceback'],flush=True)
    (outroot/'progress.json').write_text(json.dumps({'checkpoint':step,'completed':completed,'failed':failed},indent=2))
(outroot/'summary.json').write_text(json.dumps({'status':'completed' if not failed else 'failed','completed':completed,'failed':failed},indent=2))
sys.exit(1 if failed else 0)
