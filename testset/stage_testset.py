"""Upload a frozen geometry-only archive; do not submit calculations."""
from pathlib import Path
import hashlib, importlib.util, json, shlex, subprocess
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cluster',ROOT.parent/'phonons/scripts/cluster.py')
cluster=importlib.util.module_from_spec(spec);spec.loader.exec_module(cluster)
archive=ROOT/'TiAl-test50-v1.zip'
remote='/home6/u3129/tial-testsets/TiAl-test50-v1-20261021'
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
print(cluster.run('umask 077; mkdir -p /home6/u3129/tial-testsets; mkdir '+shlex.quote(remote)))
subprocess.run(cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','set -e; set -o noclobber; cat > '+shlex.quote(remote+'/TiAl-test50-v1.zip')],input=archive.read_bytes(),check=True)
code="""import hashlib,json,pathlib,zipfile
p=pathlib.Path(%r)
archive=p/'TiAl-test50-v1.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(archive) as z:
    assert all(not n.startswith('/') and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
    z.extractall(p)
root=p/'TiAl-test50-v1'
manifest=json.loads((root/'SHA256SUMS.json').read_text())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
assert len(list((root/'POSCARS').glob('POSCAR.*')))==50
print(json.dumps({'status':'verified','files':len(manifest),'poscars':50,'archive_sha256':%r,'remote_root':str(root),'dft_submitted':False}))
"""%(remote,sha,sha)
result=cluster.run('python3 -c '+shlex.quote(code))
report=json.loads(result);(ROOT/'cluster-staging.json').write_text(json.dumps(report,indent=2));print(result)
