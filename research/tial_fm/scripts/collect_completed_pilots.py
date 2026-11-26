"""Collect completed N200 results, with remote hashes and immutable local copies."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import shlex
import tarfile

import cluster_pilot as c

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/dpa4-n200-pilot-20261106-02'
CASES = ['PBE-lora16', 'PBE-full', 'r2SCAN-lora16', 'r2SCAN-full']


def main():
    remote = json.loads((RUN / 'package.json').read_text())['remote_root']
    code = r'''
from pathlib import Path
import datetime, hashlib, io, json, subprocess, sys, tarfile
root = Path(REMOTE)
cases = CASES
acct = subprocess.check_output(['sacct','-j','20988454,20988460,20988469',
    '--format=JobID,State,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,ExitCode,AllocTRES','-P'],text=True)
rows = [line.split('|') for line in acct.splitlines()[1:]]
for job in ['20988454', '20988460', '20988469']:
    for i in range(4):
        row = next(r for r in rows if r[0] == job+'_'+str(i))
        assert row[1] == 'COMPLETED' and row[6] == '0:0', row
files = []
final_artifacts = {}
for name in cases:
    folder = root / 'training-plan01/cases' / name
    for rel in ['training-summary.json','parameter-update-check.json',
                'evaluation/summary.json','evaluation/harmonics/summary.json',
                'evaluation/numerical-QC/summary.json']:
        assert json.loads((folder / rel).read_text())['status'] == 'PASS', (name, rel)
    for p in folder.rglob('*'):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:
            continue
        rel = p.relative_to(folder).as_posix()
        if p.suffix in ['.pt','.pt2'] and rel != 'merged/10000.pt':
            continue
        files.append(p)
    final_artifacts[name] = {}
    for rel in ['merged/10000.pt','checkpoints/model.ckpt-10000.pt','evaluation/final.pt2']:
        p = folder / rel
        final_artifacts[name][rel] = dict(remote_path=str(p), bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), collected=(rel=='merged/10000.pt'))
# Exact executed scripts, configurations and job logs are part of the result bundle.
for folder in [root, root / 'training-plan01']:
    files.extend(p for p in folder.iterdir() if p.is_file() and not p.is_symlink()
                 and p.suffix in ['.py','.sbatch','.json','.log','.txt'])
files = sorted(set(files))
manifest = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    remote_root=str(root), cases=cases, accounting=acct, final_artifacts=final_artifacts,
    files={p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},
    scope='Finished pilot outputs and final merged weights; no test predictions or test labels.')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as t:
    payload = (json.dumps(manifest, indent=2)+'\n').encode()
    m = tarfile.TarInfo('COLLECTION_MANIFEST.json'); m.size = len(payload)
    t.addfile(m, io.BytesIO(payload))
    for p in files:
        t.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
'''.replace('REMOTE', repr(remote)).replace('CASES', repr(CASES))
    data = c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c ' + shlex.quote(code))
    out = RUN / 'collected-pilots'
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        manifest = json.load(archive.extractfile('COLLECTION_MANIFEST.json'))
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == set(manifest['files']) | {'COLLECTION_MANIFEST.json'}
        for m in members:
            target = (out / m.name).resolve()
            assert target.is_relative_to(out.resolve()) and m.isfile()
            payload = archive.extractfile(m).read()
            if m.name in manifest['files']:
                expected = manifest['files'][m.name]
                assert len(payload) == expected['bytes']
                assert hashlib.sha256(payload).hexdigest() == expected['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == payload, ('immutable output changed', m.name)
            else:
                target.write_bytes(payload)
    record = dict(utc=manifest['utc'], archive_sha256=hashlib.sha256(data).hexdigest(),
                  archive_bytes=len(data), verified_files=len(manifest['files']),
                  output=str(out), status='PASS')
    (RUN / 'completed-pilot-collection.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
