"""Collect completed EDIFF-converged force results; exclude POTCAR/densities/WAVE."""
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
from cluster import RECORD, REMOTE, run, save, ssh, CFG

def main(xc,revision,step):
    assert xc in CFG['functionals']
    assert re.fullmatch(r'production-\d+',revision)
    assert re.fullmatch(r'(force-\d{3}|check-(cutoff|kmesh|smearing))(-restart-\d+)?',step)
    remote='/'.join([REMOTE,xc,revision,step])
    state=json.loads(run('cat '+shlex.quote(remote+'/status.json')))
    assert state['state']=='validated'
    job=state['job_id']
    assert job.isdigit()
    scheduler=run('sacct -X -nP -j '+job+' --format=JobIDRaw,State,End,Elapsed,ExitCode')
    assert any(r.split('|')[:2]==[job,'COMPLETED'] and r.split('|')[4]=='0:0' for r in scheduler.splitlines())
    names=['OUTCAR','vasprun.xml','INCAR','POSCAR','KPOINTS','CONTCAR','OSZICAR',
           'vasp.stdout','status.json','input-hashes.json','validation.json',
           'forces.npy','output-manifest.json','potcar-provenance.json']
    names.append('restart-provenance.json' if 'restart' in step or state.get('seed_source') else 'provenance.json')
    optional=['checkpoint-request.json','checkpoint-guard.json','STOPCAR']
    for name in optional:
        if run('test -f '+shlex.quote(remote+'/'+name)+' && echo yes || true').strip()=='yes': names.append(name)
    hashes={row.split()[1]:row.split()[0] for row in run('cd '+shlex.quote(remote)+'; sha256sum '+shlex.join(names)).splitlines()}
    payload=subprocess.check_output(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
        'tar -czf - -C '+shlex.quote(remote)+' '+shlex.join(names)])
    target=RECORD/'collected'/xc/revision/step
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
    text=(target/'OUTCAR').read_text(errors='replace')
    assert 'aborting loop because EDIFF is reached' in text
    assert 'General timing and accounting informations' in text
    save(xc+'-'+step+'-'+revision+'-collection-'+datetime.now(timezone.utc).strftime('%Y%m%d')+'.json',dict(
        utc=datetime.now(timezone.utc).isoformat(),remote=remote,scheduler=scheduler,files=hashes))
    print(xc,step,'collected:',len(hashes),'SHA256 verified files',flush=True)

if __name__=='__main__': main(*sys.argv[1:])
