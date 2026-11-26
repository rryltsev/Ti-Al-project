"""Remove PartitionTimeLimit in place without resubmitting the phonon jobs."""
from datetime import datetime, timezone
import re
from cluster import RECORD, run, save

JOBS = ['20959753', '20959337', '20959338']
NAME = 'apollo-walltime-correction-20261025'

if __name__ == '__main__':
    if (RECORD/(NAME+'-intent.json')).exists():
        raise RuntimeError('Reconcile existing walltime correction intent before retry')
    before = {job: run('scontrol show job -o '+job) for job in JOBS}
    for job, info in before.items():
        if 'JobState=PENDING ' not in info or 'Partition=apollo ' not in info:
            raise RuntimeError('State changed; inspect before changing walltime: '+info)
    evidence = dict(utc=datetime.now(timezone.utc).isoformat(),before=before,
                    partition=run('scontrol show partition apollo'),
                    reason='48h requests blocked by a dynamically changing weekend partition limit; use 20h per allocation',
                    requested_walltime='20:00:00',scientific_changes=[],updates=[])
    save(NAME+'-intent.json',evidence)
    for job in JOBS:
        command='scontrol update JobId='+job+' TimeLimit=20:00:00'
        result=run(command)
        after=run('scontrol show job -o '+job)
        if 'TimeLimit=20:00:00 ' not in after:
            raise RuntimeError('Walltime update not confirmed: '+after)
        for key in ['JobId','SubmitTime','EligibleTime','Partition','NumTasks','MinMemoryNode']:
            old=re.search(r'\b'+key+r'=(\S+)',before[job]).group(1)
            new=re.search(r'\b'+key+r'=(\S+)',after).group(1)
            if old != new:
                raise RuntimeError('Unexpected scheduling/input allocation change: '+key)
        evidence['updates'].append(dict(job_id=job,command=command,result=result,after=after))
        save(NAME+'-result.json',evidence)
        print(job+' TimeLimit=20:00:00; submission and eligibility times preserved',flush=True)
    evidence['queue']=run('squeue --start -j '+','.join(JOBS))
    save(NAME+'-result.json',evidence)
    print(evidence['queue'])
