"""Finalize the technical CPU-analysis reserve before any initial sbatch."""
from pathlib import Path
import sys,json,base64,hashlib,time,shlex
sys.path.insert(0,str(Path(__file__).resolve().parent));import cluster as c
root=c.ROOT
files=['workflow.py','test_analysis.py']
payload={sid:{name:base64.b64encode((root/sid/name).read_bytes()).decode() for name in files} for sid in c.REG['runs']}
code="""import pathlib,json,hashlib,base64,time
p=pathlib.Path(REMOTE);payload=PAYLOAD;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for sid,items in payload.items():
 q=p/sid;assert not (q/'submission.json').exists() and not (q/'submit-intent.json').exists()
 check=json.loads((q/'prequeue-validation.json').read_text());assert check['status']=='PASS'
 manifest=json.loads((q/'staging-manifest.json').read_text());assert all(sha(q/n)==v for n,v in manifest.items())
 changes={}
 for name,b64 in items.items():
  old=sha(q/name);(q/name).write_bytes(base64.b64decode(b64));manifest[name]=sha(q/name);changes[name]={'before':old,'after':manifest[name]}
 (q/'staging-manifest.json').write_text(json.dumps(manifest,indent=2))
 (q/'technical-reserve-patch.json').write_text(json.dumps({'unix':time.time(),'changes':changes,'physics_changed':False},indent=2))
print('Technical reserve staged; all models/protocols/legacy checks retained')
""".replace('REMOTE',repr(c.REMOTE)).replace('PAYLOAD',repr(payload))
print(c.run('python3 -',content=code.encode(),timeout=180),flush=True)
for sid,entry in c.REG['runs'].items():
 q=entry['remote'];print('Testing final code',sid,flush=True)
 command='cd '+shlex.quote(q)+' && CUDA_VISIBLE_DEVICES= DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR='+shlex.quote(q+'/cache/matplotlib')+' '+c.PYTHON+' -B test_analysis.py'
 result=c.run(command,timeout=600);assert '"status": "PASS"' in result
 (root/sid/'final-cpu-tests.log').write_text(result,encoding='utf-8')
 code="""import pathlib,json,hashlib,time
p=pathlib.Path(REMOTE);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();h=json.loads((p/'staging-manifest.json').read_text());assert all(sha(p/n)==v for n,v in h.items())
r=json.loads((p/'prequeue-validation.json').read_text());r.update(final_cpu_tests='PASS including CPU analysis walltime deferral',validated_staging_manifest_sha256=sha(p/'staging-manifest.json'),technical_code_verified_unix=time.time());(p/'prequeue-validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
""".replace('REMOTE',repr(q))
 check=json.loads(c.py(code));c.save(root/sid/'prequeue-validation.json',check);print(sid,'final code PASS',flush=True)
# Keep the root deployment receipt consistent with the final technical sources.
receipt=json.loads((root/'staging-complete.json').read_text())
for sid in c.REG['runs']:
 for name in files:receipt['files'][sid+'/'+name]=c.sha(root/sid/name)
receipt['technical_patch_verified_unix']=time.time();c.save(root/'staging-complete.json',receipt)
