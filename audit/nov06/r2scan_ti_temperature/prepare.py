from pathlib import Path
import json,hashlib,shutil,time,sys,re
R=Path(__file__).resolve().parent;W=R.parents[2]
sys.path.insert(0,str(W/'audit/nov01/r2scan_viscosity_diagnostics/deps'))
import numpy as np
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=read(W/'audit/oct29/pbe15k_ti_temperature/protocol.json');models=read(W/'audit/oct25/all_xc_audit/protocol.json')['models'];checkpoint=models['r2SCAN-23200'];assert sha(checkpoint['local_path'])==checkpoint['sha256']
frozen=dict(path='/home6/u3129/tial-testsets/model-export-20261022-02-cpu/r2SCAN/model_frozen.pt2',sha256='1a8c1ef801263c7cbaa8561b2dbf4af32d46fb4375a6cfafaad13bde29f91c0a')
prior=read(W/'audit/nov01/r2scan_viscosity_diagnostics/results.json');remote='/home6/u3129/tial-testsets/r2scan-ti-temperature-20261106-01';registry=dict(run_id='r2scan-ti-temperature-20261106-01',remote=remote,created_unix=time.time(),temperatures_K=[1873,1941,2000,2100],functional='r2SCAN',checkpoint=checkpoint,frozen_model=frozen,runs={})
seen=set();checks=[]
for T in registry['temperatures_K']:
 sid=f'Ti_{T}K';out=R/sid;assert not out.exists();(out/'legacy').mkdir(parents=True);e=prior['temperatures'][str(T)];arr=[];manifest=[]
 for item in e['manifest']:
  f=Path(item['pressure']);mol=f.parent/'mol.data';assert sha(f)==item['pressure_sha256'] and sha(mol)==item['mol_sha256'];assert item['pressure_sha256'] not in seen;seen.add(item['pressure_sha256'])
  data=np.loadtxt(f);assert data.shape==(20001,7) and np.isfinite(data).all() and np.array_equal(data[:,0],np.arange(20001))
  text=mol.read_text();bounds=re.findall(r'^\s*([-+.\deE]+)\s+([-+.\deE]+)\s+[xyz]lo [xyz]hi',text,re.M);assert len(bounds)==3 and '512 atoms' in text
  atoms=[l.split() for l in text.split('Atoms # atomic')[1].split('Velocities')[0].splitlines() if l.strip()];assert len(atoms)==512 and all(a[1]=='2' for a in atoms)
  vol=float(np.prod([float(b)-float(a) for a,b in bounds]));assert abs(vol/item['volume_A3']-1)<1e-12
  n=item['replica'];dest=out/f'legacy/mol_{n:04d}.data';shutil.copyfile(mol,dest);arr.append(data[:,1:]);manifest.append(dict(replica=n,source_pressure=str(f),source_pressure_sha256=sha(f),source_mol=str(mol),mol_file=dest.relative_to(out).as_posix(),mol_sha256=sha(mol),volume_A3=vol,first_pressure_bar=data[0,1:].tolist()))
 np.savez_compressed(out/'legacy/stress.npz',pressure_bar=np.asarray(arr),volume_A3=np.array([x['volume_A3'] for x in manifest]),dt_ps=.001,temperature_K=T)
 legacy=dict(temperature_K=T,n_replicas=len(arr),production_ps=20,raw_manifest=manifest,stress_file='legacy/stress.npz',stress_sha256=sha(out/'legacy/stress.npz'),published=e['published'],source_audit=str(W/'audit/nov01/r2scan_viscosity_diagnostics/results.json'),source_audit_sha256=sha(W/'audit/nov01/r2scan_viscosity_diagnostics/results.json'))
 (out/'legacy/manifest.json').write_text(json.dumps(legacy,indent=2)+'\n');shutil.copyfile(Path(manifest[0]['source_mol']),out/'initial.data')
 cfg=json.loads(json.dumps(base));cfg.update(run_id=f'r2scan-ti-{T}-20261106-01',remote=remote+'/'+sid,checkpoint=checkpoint,frozen_model=frozen,states=[dict(id=sid,temperature_K=T,x_Al=0.,c_Ti=1.,actual_x_Al=0.,counts={'Al':0,'Ti':512},initial_file='initial.data',initial_sha256=sha(out/'initial.data'),initial_source=manifest[0]['source_mol'],seed=611000000+T*100)],functional='r2SCAN')
 cfg.pop('inherited_job_id',None);cfg['allocation']['node']=None;cfg['scheduling'].update(priority_temperatures_K=[T],policy='Independent temperature job;analyze after5,10,and every5;four jobs submitted together,oneA100 each;actual concurrency limited by cluster GPUs.')
 cfg['restart_policy']='One initial guarded sbatch for this temperature;20h allocations continue only by same-ID requeue at a completed-replica boundary. PBE jobs must never be restarted.'
 cfg['duration_policy']='Primary refinement combines verified legacy20ps spectra with two20ps segments per new40ps replica; trajectories are never joined across boundaries. New40ps/first20/second20 estimates and new5 remain separately reported.'
 cfg['preparation_note']='New independent seeded reservoir from archived pureTi configuration,20psNPT then5psNPT+1psNVT+40psNVT per newreplica;legacy preparation remains separately identified.'
 cfg['legacy']=dict(manifest='legacy/manifest.json',sha256=sha(out/'legacy/manifest.json'),n_replicas=len(arr),duration_ps=20)
 cfg['pooling']=dict(version='r2scan-legacy20-new40-v1',primary='Pooled20ps spectrum;old20ps+two20pswindows from each new40ps',weight='Equal-length20ps channels pooled without temporal concatenation;ownV/(kB*T),fivechannels',uncertainty='max(formalSE,delete-one-whole-replica cluster jackknifeSE);both halves removed together for newreplicas',required_model_pressure_max_error_bar=20,additional_stop_gates=['pooled20cost/cutoffZ<2','pooled20conservativeSE<=3%','pooled20last3range<=2%'],new40_stop_rule='Retain full original heuristic includingNnew>=30,neff40>=60,two consecutive passes;old20ps do not substitute for40ps duration checks',legacy_model_validation='All archived zero-step pressures must match checkpoint23200 before submission')
 (out/'protocol.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for name in ['workflow.py','analysis_common.py','test_analysis.py']:
  shutil.copyfile(W/'audit/oct29/pbe15k_ti_temperature'/name,out/name)
 md=(W/'audit/oct29/pbe15k_ti_temperature/md_core.py').read_text(encoding='utf-8').replace("assert meta['train_infos']['step']==14999","assert meta['train_infos']['step']==CFG['checkpoint']['step']-1")
 (out/'md_core.py').write_text(md,encoding='utf-8');(out/'job.sbatch').write_bytes((W/'audit/nov05/pbe15k_ti_1941/job.sbatch').read_bytes())
 registry['runs'][sid]=dict(temperature_K=T,local=str(out),remote=cfg['remote'],protocol_sha256=sha(out/'protocol.json'),legacy_replicas=len(arr),job_id=None)
 checks.append(dict(T=T,legacy_replicas=len(arr),samples=20001,duplicate_pressure_files=False,all_512_Ti=True,raw_and_volume_checks='PASS'))
(R/'registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(R/'preparation-checks.json').write_text(json.dumps(dict(unix=time.time(),checks=checks,total_distinct_legacy_replicas=len(seen)),indent=2)+'\n')
print(json.dumps(checks))
