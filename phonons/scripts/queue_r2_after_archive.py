"""Continue improved r2SCAN orbitals only after durable archival frees quota."""
from datetime import datetime, timezone
import hashlib
import json
import shlex
from cluster import ROOT, RECORD, REMOTE, run, save, CFG

NAME='r2SCAN-production02-force-001-wave-next-20261029'


if __name__=='__main__':
    if (RECORD/(NAME+'-intent.json')).exists():raise RuntimeError('Reconcile existing restart intent')
    archive=json.loads((RECORD/'r2SCAN-old-wavecar-archive-20261029-attempt02-result.json').read_text())
    assert archive['result']['state']=='archived_with_original_path_preserved'
    parent=run('sacct -X -nP -j 20964090 --format=JobIDRaw,State,End,ExitCode')
    assert '20964090|COMPLETED|' in parent
    queue=run('squeue -h -u u3129 -o "%i|%j|%T"')
    assert 'ti7-r2SCAN-f001' not in queue
    source=REMOTE+'/r2SCAN/production-02/force-001-restart-03'
    run('test ! -e '+REMOTE+'/r2SCAN/production-02/force-001-restart-04')
    # Small header/record checks here; full checksums and copying occur in Slurm.
    code='''import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,ROOT)
from wavecar_check import inspect_wavecar
p=Path(SOURCE)
lines=(p/'POSCAR').read_text().splitlines()
cell=[[float(x) for x in row.split()] for row in lines[2:5]]
wave=inspect_wavecar(p/'WAVECAR',cell)
q=subprocess.check_output(['quota','-w'],text=True)
row=next(s for s in q.splitlines() if ':/home6' in s).split()
free=(int(row[3].rstrip('*'))-int(row[1].rstrip('*')))*1024
required=wave['bytes']+3*((p/'CHGCAR').stat().st_size+(p/'TAUCAR').stat().st_size)+4*1024**3
assert free>required,(free,required)
print(json.dumps(dict(wavecar=wave,quota=q,free_bytes=free,required_bytes=required)))
'''.replace('ROOT',repr(REMOTE)).replace('SOURCE',repr(source))
    checked=json.loads(run(CFG['python_base']+' -c '+shlex.quote(code)))
    launcher=REMOTE+'/launch_v11.sbatch'
    source_hashes={}
    for name in ['restart_step_v7.py','wavecar_check.py','launch_v11.sbatch']:
        local=ROOT/'scripts'/name
        sha=hashlib.sha256(local.read_bytes()).hexdigest()
        assert run('sha256sum '+REMOTE+'/'+name).split()[0]==sha
        source_hashes[name]=sha
    args=['sbatch','--parsable','--partition=apollo','--exclude=tesla-a101,apollo22',
          '--account=imet-2026-2','--qos=normal','--nodes=2','--ntasks=64',
          '--ntasks-per-node=32','--cpus-per-task=1','--mem=240G','--time=20:00:00',
          '--job-name=ti7-r2SCAN-f001','--chdir='+REMOTE,'--output='+REMOTE+'/'+NAME+'-%j.out',
          launcher,'restart_step_v7.py','r2SCAN','force-001','20964090']
    intent=dict(utc=datetime.now(timezone.utc).isoformat(),command=args,parent_job='20964090',
                source_hashes=source_hashes,scientific_changes=[],source=str(source),
                storage_checks=checked,parent_scheduler=parent,archive=archive['result'])
    save(NAME+'-intent.json',intent)
    response=run('set -o pipefail; '+shlex.join(args)+' | tee '+shlex.quote(REMOTE+'/'+NAME+'.jobid')).strip()
    job=response.split(';')[0]
    if not job.isdigit():raise RuntimeError('Ambiguous scheduler response: '+response)
    save(NAME+'-submission.json',dict(intent,job_id=job,state='submitted'))
    print(job,flush=True)
