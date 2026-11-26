"""Retrieve converged PBE forces with checksums, excluding licensed/large files."""
from datetime import datetime, timezone
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from cluster import RECORD, REMOTE, run, save, ssh

if __name__ == '__main__':
    remote = REMOTE+'/PBE/production-02/force-001-restart-05'
    target = RECORD/'collected/PBE/production-02/force-001-restart-05'
    names = ['OUTCAR','vasprun.xml','INCAR','POSCAR','KPOINTS','CONTCAR',
             'OSZICAR','vasp.stdout','status.json','input-hashes.json',
             'restart-provenance.json','validation.json','forces.npy',
             'output-manifest.json','potcar-provenance.json']
    scheduler = run('sacct -X -nP -j 20968908 --format=JobIDRaw,State,End,Elapsed,ExitCode')
    assert any(r.split('|')[:2]==['20968908','COMPLETED'] and r.split('|')[4]=='0:0' for r in scheduler.splitlines())
    hashes = {row.split()[1]:row.split()[0] for row in run('cd '+shlex.quote(remote)+'; sha256sum '+shlex.join(names)).splitlines()}
    payload = subprocess.check_output(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
        'tar -czf - -C '+shlex.quote(remote)+' '+shlex.join(names)])
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        assert set(archive.getnames())==set(names)
        for member in archive:
            assert member.isfile() and member.name in names
            data=archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==hashes[member.name]
            path=target/member.name
            if path.exists(): assert path.read_bytes()==data
            else: path.write_bytes(data)
    save('PBE-force001-result-collection-20261031.json',dict(
        utc=datetime.now(timezone.utc).isoformat(),remote=remote,scheduler=scheduler,files=hashes))
    print('PBE force-001: 15 result files downloaded and SHA256 verified.')
