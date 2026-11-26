"""Read-only collection of completed r2SCAN results with per-file SHA-256 verification."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'r2scan_enthalpy_extend30k'
DEST = OUT/'collected'
sys.path.insert(0, str(ROOT/'testset/dft'))
import cluster_test50 as c
cfg = json.loads((OUT/'protocol.json').read_text())
remote = cfg['remote']
ssh = c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt']

def run(command):
    result = subprocess.run(ssh+[command], capture_output=True, encoding='utf-8', timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout

job = json.loads((OUT/'submission.json').read_text())['job_id']
accounting = run('sacct -j '+job+' --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList -Pn')
assert any(line.startswith(job+'|') and '|COMPLETED|0:0|' in line for line in accounting.splitlines()), accounting
(OUT/'accounting-completed.txt').write_text(accounting)
code = '''from pathlib import Path
import hashlib,json
p=Path(REMOTE)
summary=json.loads((p/'summary.json').read_text())
assert summary['status']=='completed' and len(summary['completed'])==11 and not summary['failed']
files=[f for f in p.iterdir() if f.is_file() and f.suffix in ['.json','.npz','.py','.sbatch','.txt','.traj','.csv','.log']]
files += [f for f in (p/'results').rglob('*') if f.is_file()]
manifest={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(files)}
print(json.dumps(dict(summary=summary,files=manifest)))
'''.replace('REMOTE', repr(remote))
inventory = json.loads(run('python3 -c '+shlex.quote(code)))
assert not DEST.exists(), 'Collection already exists; verify it instead of overwriting'
(OUT/'remote-results-manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
names = list(inventory['files'])
assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
archive = OUT/'completed-results.tar.gz'
assert not archive.exists()
print('Downloading', len(names), 'files;', sum(x['bytes'] for x in inventory['files'].values()), 'uncompressed bytes', flush=True)
command = 'tar -czf - -C '+shlex.quote(remote)+' -- '+shlex.join(names)
with archive.open('xb') as fp:
    subprocess.run(ssh+[command], stdout=fp, check=True, timeout=600)
DEST.mkdir()
with tarfile.open(archive, 'r:gz') as tf:
    members = tf.getmembers()
    assert len(members) == len(names) and {m.name for m in members} == set(names)
    for member in members:
        assert member.isfile()
        target = DEST.joinpath(*PurePosixPath(member.name).parts).resolve()
        assert target.is_relative_to(DEST.resolve())
        content = tf.extractfile(member).read()
        expected = inventory['files'][member.name]
        assert len(content) == expected['bytes'] and hashlib.sha256(content).hexdigest() == expected['sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as fp:
            fp.write(content)
staged = json.loads((OUT/'staging-complete.json').read_text())['files']
assert all(hashlib.sha256((DEST/n).read_bytes()).hexdigest() == h for n,h in staged.items())
verification = dict(status='passed', job_id=job, files_verified=len(names),
                    staging_inputs_unchanged=True, collected_utc=datetime.now(timezone.utc).isoformat(),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(OUT/'collection-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(accounting, json.dumps(verification), flush=True)
