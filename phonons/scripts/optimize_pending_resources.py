"""Change scheduler resources only; preserve job IDs, inputs and task counts."""
from datetime import datetime, timezone
from cluster import run, save

raise SystemExit('Historical one-off optimizer disabled: CPU VASP must stay on Apollo. See scheduler-overrides-05.json; never rerun the A100 placement.')

plans=[('PBE','20959336','v100',32,1),('r2SCAN','20959337','apollo',32,1),('PBEsol','20959338','apollo',64,2)]
evidence=dict(utc=datetime.now(timezone.utc).isoformat(),
              partitions=run('scontrol show partition apollo; scontrol show partition v100'),
              previous_peak_RSS=run('sacct -nP -j 20958089.0,20958090.0,20958091.0 --format=JobIDRaw,MaxRSS'),
              rationale='240 GiB per node vs <=~215 GiB conservative 32*MaxRSS bound; 48h avoids premature SCF checkpoint; PBE v100 CPU-only early projection',
              scientific_changes=[],updates=[])
save('resource-optimization-20261023-intent.json',evidence)
for xc,job,partition,tasks,nodes in plans:
    before=run('scontrol show job -o '+job)
    if 'JobState=PENDING ' not in before: raise RuntimeError('Job is no longer pending: '+job)
    if 'NumTasks='+str(tasks)+' ' not in before: raise RuntimeError('Unexpected task count')
    command='scontrol update JobId='+job+' Partition='+partition+' MinMemoryNode=245760 TimeLimit=2-00:00:00'
    save('resource-optimization-'+job+'-intent.json',dict(before=before,command=command,scientific_changes=[]))
    output=run(command)
    after=run('scontrol show job -o '+job)
    if 'MinMemoryNode=240G ' not in after or 'TimeLimit=2-00:00:00 ' not in after:
        raise RuntimeError('Update did not apply as intended: '+after)
    update=dict(xc=xc,job_id=job,command=command,before=before,after=after,output=output)
    evidence['updates'].append(update)
    save('resource-optimization-20261023-applied.json',evidence)
    print(xc,job,partition,'240GiB/node','48h',flush=True)
evidence['queue']=run('squeue --start -j 20959336,20959337,20959338')
save('resource-optimization-20261023-applied.json',evidence)
print(evidence['queue'])
