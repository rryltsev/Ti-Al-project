"""Collect completed expansion results and final weights with SHA256 verification."""
from pathlib import Path
import argparse,hashlib,io,json,shlex,tarfile
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-replicates-full-20261106-02'

def main():
    global RUN
    ap=argparse.ArgumentParser();ap.add_argument('--run',default=RUN.name);args=ap.parse_args()
    RUN=ROOT/'runs'/args.run
    remote=json.loads((RUN/'package.json').read_text())['remote_root']
    code=r'''from pathlib import Path
import datetime,hashlib,io,json,subprocess,sys,tarfile
root=Path(REMOTE);plan=json.loads((root/'plan.json').read_text())
jobs=[json.loads(p.read_text())['job_id'] for p in (root/'submissions').glob('*.json')]
accounting=subprocess.check_output(['sacct','-j',','.join(jobs),'--format=JobID,State,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,ExitCode,AllocTRES','-P'],text=True)
rows=[line.split('|') for line in accounting.splitlines()[1:] if line and '.' not in line.split('|')[0]]
assert len(rows)==2*len(plan['cases'])+len(plan['groups'])
assert all(row[1]=='COMPLETED' and row[6]=='0:0' for row in rows),rows
files=[];finals={}
for case in plan['cases']:
    folder=root/'training-plan01/cases'/case['name'];step=case['expected_steps']
    for name in ['training-summary.json','parameter-update-check.json','evaluation/summary.json',
                 'evaluation/harmonics/summary.json','evaluation/numerical-QC/summary.json','evaluation/physical-diagnostics/summary.json']:
        assert json.loads((folder/name).read_text())['status']=='PASS',(case['name'],name)
    final=f'merged/{step:05d}.pt'
    for p in folder.rglob('*'):
        if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
            if p.suffix in ['.pt','.pt2'] and p.relative_to(folder).as_posix()!=final:continue
            files.append(p)
    finals[case['name']]={}
    for rel in [final,f'checkpoints/model.ckpt-{step}.pt','evaluation/final.pt2']:
        p=folder/rel;finals[case['name']][rel]=dict(remote_path=str(p),bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),collected=(rel==final))
for group in plan['groups']:
    assert json.loads((root/'aggregates'/group/'summary.json').read_text())['status']=='PASS'
for folder in [root/'aggregates',root/'submissions']:
    files.extend(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
for folder in [root,root/'training-plan01']:
    files.extend(p for p in folder.iterdir() if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.sbatch','.json','.log','.txt'])
files=sorted(set(files));manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_root=str(root),
    accounting=accounting,final_artifacts=finals,files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},
    scope='Completed cases; final merged weights, predictions, harmonic outputs, QC and logs. No test labels/predictions.')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
    payload=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('COLLECTION_MANIFEST.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
    for p in files:t.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
'''.replace('REMOTE',repr(remote))
    data=c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code))
    out=RUN/'collected';out.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
        manifest=json.load(t.extractfile('COLLECTION_MANIFEST.json'));members=t.getmembers()
        assert len({m.name for m in members})==len(members)
        assert {m.name for m in members}==set(manifest['files'])|{'COLLECTION_MANIFEST.json'}
        for m in members:
            p=(out/m.name).resolve();assert p.is_relative_to(out.resolve()) and m.isfile()
            payload=t.extractfile(m).read()
            if m.name in manifest['files']:
                e=manifest['files'][m.name];assert len(payload)==e['bytes'] and hashlib.sha256(payload).hexdigest()==e['sha256'],m.name
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    record=dict(status='PASS',utc=manifest['utc'],verified_files=len(manifest['files']),archive_bytes=len(data),archive_sha256=hashlib.sha256(data).hexdigest(),output=str(out))
    (RUN/'completed-collection.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
