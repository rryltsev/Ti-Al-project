"""Stage or collect one short CPU diagnostic on already computed harmonic FCs."""
from pathlib import Path
import argparse, hashlib, io, json, shlex, tarfile
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
NAME='pilot-physical-diagnostics-v1'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['submit','collect']);a=ap.parse_args()
    remote=json.loads((RUN/'package.json').read_text())['remote_root']
    if a.action=='submit':
        assert not (RUN/(NAME+'-submission.json')).exists()
        source=(ROOT/'scripts/analyze_pilot_physics.py').read_text()
        launcher=f'''#!/bin/bash
set -eo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd {shlex.quote(remote)}
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python analyze_pilot_physics_v1.py --root . --output {NAME}
'''
        payload={'analyze_pilot_physics_v1.py':source, NAME+'.sbatch':launcher}
        code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED')"
        print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
        command=f'cd {shlex.quote(remote)} && sbatch --parsable --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:05:00 --job-name=TiAl-pilot-physics --output={NAME}-%j.log {NAME}.sbatch'
        out=c.ssh(command).decode().strip();assert out.split(';')[0].isdigit()
        record=dict(job_id=out.split(';')[0],command=command,source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},scope='Saved FC postprocessing only; no new model forces or DFT.')
        (RUN/(NAME+'-submission.json')).write_text(json.dumps(record,indent=2)+'\n');print(out)
    else:
        code=r'''from pathlib import Path
import hashlib,io,json,sys,tarfile
p=Path(REMOTE);assert json.loads((p/'summary.json').read_text())['status']=='PASS'
files=sorted(x for x in p.iterdir() if x.is_file() and not x.is_symlink())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
    payload=json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2).encode()
    info=tarfile.TarInfo('SHA256.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
    for f in files:t.add(f,arcname=f.name,recursive=False)
'''.replace('REMOTE',repr(remote+'/'+NAME))
        data=c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code))
        out=RUN/NAME;out.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
            hashes=json.load(t.extractfile('SHA256.json'))
            for m in t.getmembers():
                path=(out/m.name).resolve();assert path.is_relative_to(out.resolve()) and m.isfile()
                payload=t.extractfile(m).read()
                if m.name!='SHA256.json':assert hashlib.sha256(payload).hexdigest()==hashes[m.name]
                if path.exists():assert path.read_bytes()==payload
                else:path.write_bytes(payload)
        print('COLLECTED',out)

if __name__=='__main__':main()
