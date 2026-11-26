"""Read-only final checkpoint and input provenance verification on UMT."""
import argparse,json
from pathlib import Path
import manage_expansion as m
from prepare_grace_pilot import RUN,REMOTE

code=r'''from pathlib import Path
import hashlib,json,subprocess,datetime
p=Path(REMOTE)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
verified={}
for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json']:
 for name,h in json.loads((p/manifest).read_text()).items():
  assert sha(p/name)==h,name
  verified[name]=h
cases={}
for c in json.loads((p/'plan.json').read_text())['cases']:
 w=p/'cases'/c['name'];s=json.loads((w/'training-summary.json').read_text());assert s['status']=='PASS' and s['steps']==c['steps'] and s['epochs']==c['epochs']
 hashes={n:sha(w/'final-checkpoint'/n) for n in s['hashes']};assert hashes==s['hashes']
 seed_folder=w/'seed'/str(c['seed'])
 export={f.relative_to(seed_folder/'final_model').as_posix():sha(f) for f in (seed_folder/'final_model').rglob('*') if f.is_file()}
 assert export
 archive=None
 if (seed_folder/'checkpoints/ARCHIVE.json').exists():
  archive=json.loads((seed_folder/'checkpoints/ARCHIVE.json').read_text());assert archive['status']=='PASS'
  assert sha(seed_folder/'checkpoints/optimizer-final.tar.gz')==archive['archive_sha256']
 cases[c['name']]=dict(checkpoint=hashes,export=export,job_id=json.loads((w/'attempt.json').read_text())['job'],training_summary_sha256=sha(w/'training-summary.json'),model_yaml_sha256=sha(seed_folder/'model.yaml'),optimizer_archive=archive)
print(json.dumps(dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=verified,cases=cases,accounting=subprocess.check_output(['sacct','-X','-j',JOBS,'-P','--format=JobID,State,ElapsedRaw,ExitCode,NodeList'],text=True))))
'''
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run');a=ap.parse_args()
    if a.run:
        assert Path(a.run).name==a.run;RUN=RUN.parent/a.run;REMOTE=json.loads((RUN/'package.json').read_text())['remote_root']
    keys=[k for k in ['train','trained-harmonics','postprocess'] if (RUN/'submissions'/(k+'.json')).exists()]+sorted(p.stem for p in (RUN/'submissions').glob('train-*.json'))
    jobs=','.join(json.loads((RUN/'submissions'/(k+'.json')).read_text())['job_id'] for k in keys)
    d=json.loads(m.remote_python(code.replace('REMOTE',repr(REMOTE)).replace('JOBS',repr(jobs))));(RUN/'final-remote-verification.json').write_text(json.dumps(d,indent=2)+'\n')
    print(d['status'],len(d['cases']),'checkpoints verified;',len(d['inputs']),'input files');print(d['accounting'])
