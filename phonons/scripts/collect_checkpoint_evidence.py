"""Collect stopped attempts and densities, never POTCAR, with SHA256 verification."""
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from cluster import RECORD, REMOTE, run, save, ssh

def main():
    manifest={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        folder=RECORD/'collected'/xc/'production-02/force-001'
        folder.mkdir(parents=True,exist_ok=True)
        names=['CHGCAR','TAUCAR','OUTCAR','vasprun.xml','vasp.stdout','OSZICAR','INCAR','KPOINTS','POSCAR','CONTCAR',
               'input-hashes.json','status.json','checkpoint-request.json','validation.json','potcar-provenance.json']
        for name in names:
            remote=REMOTE+'/'+xc+'/production-02/force-001/'+name
            result=run('if test -f '+shlex.quote(remote)+'; then sha256sum '+shlex.quote(remote)+'; fi').strip()
            if not result: continue
            expected=result.split()[0]
            target=folder/name
            if not target.exists() or hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=expected:
                part=target.with_name(name+'.part')
                with part.open('wb') as out:
                    subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','cat '+shlex.quote(remote)],stdout=out,check=True)
                with part.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=expected: raise RuntimeError('Transfer mismatch: '+remote)
                part.replace(target)
            manifest[xc+'/'+name]=dict(sha256=expected,bytes=target.stat().st_size,remote=remote)
        save('stopped-checkpoints-collection.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=manifest))
        print(xc,'collected and checksum verified',flush=True)

if __name__=='__main__': main()
