from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/trained_numerical_qc.py').read_text();base=(RUN/'package/baseline.sbatch').read_text().split('python baseline_worker.py')[0]
launch=base+'CASES=(PBE-lora16 PBE-full r2SCAN-lora16 r2SCAN-full)\nCASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}\npython trained_numerical_qc.py --case "$CASE_NAME"\n'
early=base+'python trained_numerical_qc.py --case PBE-lora16 --step 1000\n'
payload={'trained_numerical_qc.py':source,'trained-QC.sbatch':launch,'checkpoint1000-QC.sbatch':early};payload['trained-QC-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_POST_ADAPTATION_QC')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
prefix=f'cd {shlex.quote(remote)} && sbatch --parsable --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=12G --time=00:10:00 '
records={}
for name,args in [('final','--dependency=aftercorr:20988454 --array=0-3 --job-name=TiAl-pilot-QC --output=pilot-QC-%A_%a.log trained-QC.sbatch'),('first_checkpoint','--job-name=TiAl-LoRA-1000-QC --output=checkpoint1000-QC-%j.log checkpoint1000-QC.sbatch')]:
    command=prefix+args;out=c.ssh(command).decode().strip();assert out.split(';')[0].isdigit();records[name]=dict(job_id=out.split(';')[0],command=command)
(RUN/'trained-QC-submission.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
