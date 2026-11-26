"""Stage or submit individual calculations under the uncapped user authorization."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import re
import shlex
import subprocess
from cluster import ROOT,RECORD,REMOTE,CFG,run,save,ssh

P=json.loads((ROOT/'config/execution-policy-02.json').read_text())

def stage():
    files=[ROOT/'config/execution-policy-02.json',ROOT/'scripts/production_step.py',ROOT/'scripts/production-step.sbatch']
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'policy':P,
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    save('production02-source-manifest.json',manifest)
    for p in files:
        subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
            'set -e; set -o noclobber; cat > '+shlex.quote(REMOTE+'/'+p.name)],input=p.read_bytes(),check=True)
    print(run('cd '+shlex.quote(REMOTE)+'; sha256sum execution-policy-02.json production_step.py production-step.sbatch'))

def submit(xc,step):
    placement=json.loads((ROOT/'config/vasp-placement.json').read_text())
    if P['partition'] != placement['allowed_partition']:
        raise RuntimeError('VASP must run on Apollo; A100 nodes are forbidden by the user')
    if xc not in CFG['functionals'] or step not in P['steps']: raise ValueError((xc,step))
    hold_path=ROOT/'config/workflow-holds-20261031.json'
    if hold_path.exists() and step.startswith('check-'):
        hold=json.loads(hold_path.read_text())
        if hold.get('defer_parameter_checks_until_all_functionals_base_forces_ready'):
            missing=[]
            for functional in hold['functionals']:
                for base_step in hold['required_base_steps']:
                    reports=(ROOT/'results').glob(functional+'-'+base_step.replace('-','')+'-validation-*.json')
                    accepted=False
                    for report in reports:
                        data=json.loads(report.read_text())
                        if data.get('accepted') and data.get('functional')==functional and data.get('step')==base_step:
                            accepted=True
                    if not accepted: missing.append(functional+'/'+base_step)
            if missing: raise RuntimeError('User deferred parameter checks until baseline phonons finish; missing accepted force sets: '+', '.join(missing))
    current=P
    policy_name='execution-policy-02.json'
    launcher_driver=None
    if xc == 'PBE':
        policy_name='execution-policy-09.json' if step=='force-000' else 'execution-policy-06.json'
        current=json.loads((ROOT/'config'/policy_name).read_text())
        if step not in current['steps']: raise ValueError('PBE step not reviewed in this policy')
        audit=json.loads((ROOT/'results'/('PBE-force002-validation-20261101.json' if step=='force-000' else 'PBE-force001-validation-20261031.json')).read_text())
        if not audit['accepted'] or audit['job_id']!=('20972819' if step=='force-000' else '20968908'):
            raise RuntimeError('Independent PBE force-001 audit has not passed')
        launcher_driver=['launch_v15.sbatch','production_step_v9.py'] if step=='force-000' else ['launch_v12.sbatch','production_step_v6.py']
    if xc == 'PBEsol':
        if step=='check-cutoff':
            policy_name='execution-policy-07.json'
            launcher_driver=['launch_v13.sbatch','production_step_v7.py']
            audit=json.loads((ROOT/'results/PBEsol-force000-validation-20261031.json').read_text())
            if not audit['accepted'] or audit['job_id']!='20968907':
                raise RuntimeError('Independent PBEsol reference audit has not passed')
        elif step=='force-000':
            policy_name='execution-policy-05.json'
            launcher_driver=['launch_v11.sbatch','production_step_v5.py']
        else: raise ValueError('Review memory/resource plan before later PBEsol diagnostics')
        current=json.loads((ROOT/'config'/policy_name).read_text())
        if step not in current['steps']: raise ValueError('PBEsol force-001 already completed; do not repeat')
        if step == 'force-000':
            audit=json.loads((ROOT/'results/PBEsol-force002-validation-20261029.json').read_text())
            if not audit['accepted'] or audit['job_id'] != '20962927':
                raise RuntimeError('Independent PBEsol force-002 audit has not passed')
    if xc=='r2SCAN':
        policy_name='execution-policy-10.json' if step=='force-000' else 'execution-policy-08.json'
        current=json.loads((ROOT/'config'/policy_name).read_text())
        if step not in current['steps']: raise ValueError('r2SCAN step not reviewed in this policy')
        launcher_driver=['launch_v15.sbatch','production_step_v10.py'] if step=='force-000' else ['launch_v13.sbatch','production_step_v8.py']
        audit=json.loads((ROOT/'results'/('r2SCAN-force002-validation-20261101.json' if step=='force-000' else 'r2SCAN-force001-validation-20261031.json')).read_text())
        if not audit['accepted'] or audit['job_id']!=('20972837' if step=='force-000' else '20968910'): raise RuntimeError('r2SCAN independent audit required')
        archive=json.loads((RECORD/('r2SCAN-first-wavecar-archive-20261101-result.json' if step=='force-000' else 'r2SCAN-old-wavecar-archive-20261031-attempt02-result.json')).read_text())
        if archive['result']['state']!='archived_with_original_path_preserved': raise RuntimeError('Archive not durable/verified')
    name=xc+'-'+current['revision'].replace('-','')+'-'+step
    if (RECORD/(name+'-intent.json')).exists(): raise RuntimeError('Existing intent: reconcile before retry')
    queue=run('squeue -h -u u3129 -o "%i|%j|%T|%Z"')
    for row in queue.splitlines():
        columns=row.split('|')
        if (len(columns)>3 and columns[3].startswith(REMOTE)
                and re.search(r'(^|-)'+re.escape(xc)+r'(-|$)',columns[1])):
            raise RuntimeError('Another job for this functional is still queued/active: '+row)
    run('test ! -e '+shlex.quote(REMOTE+'/'+xc+'/'+current['revision']+'/'+step))
    source_hashes={}
    if launcher_driver:
        dependencies=['config/'+policy_name,'scripts/'+launcher_driver[0],'scripts/'+launcher_driver[1]]
        if policy_name=='execution-policy-07.json': dependencies.append('scripts/checkpoint_guard_v5.py')
        if policy_name=='execution-policy-09.json': dependencies.append('scripts/checkpoint_guard_v2.py')
        if policy_name in ['execution-policy-08.json','execution-policy-10.json']:
            dependencies.extend('scripts/'+s for s in ['checkpoint_guard_v2.py','wavecar_check.py','restart_step_v7.py','restart_step_v2.py'])
        for relative in dependencies:
            path=ROOT/relative
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if run('sha256sum '+shlex.quote(REMOTE+'/'+path.name)).split()[0]!=sha:
                raise RuntimeError('Staged source mismatch: '+path.name)
            source_hashes[path.name]=sha
    resources=current['dense_kmesh' if step=='check-kmesh' else 'standard']
    index = P['steps'].index(step)
    if index:
        previous = P['steps'][index-1]
        # Old production drivers count SCF steps but do not distinguish STOPCAR.
        # Require positive EDIFF evidence before advancing to another displacement.
        code = '''
import json, subprocess
from pathlib import Path
root = Path(BASE)
valid = False
for p in [*sorted(root.glob('production-*/'+PREVIOUS)), *sorted(root.glob('production-*/'+PREVIOUS+'-restart-*'))]:
    if not (p/'status.json').exists(): continue
    status = json.loads((p/'status.json').read_text())
    if status.get('state') != 'validated': continue
    text = (p/'OUTCAR').read_text(errors='replace')
    if 'aborting loop because EDIFF is reached' not in text: continue
    if 'General timing and accounting informations' not in text: continue
    job = status['job_id']
    rows = subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobIDRaw,State'],text=True)
    if not any(row.split('|')[:2] == [job,'COMPLETED'] for row in rows.splitlines()): continue
    if (p/'validation.json').exists() and (p/'forces.npy').exists(): valid=True
if not valid: raise RuntimeError('Previous step is not scientifically validated and completed')
'''.replace('BASE',repr(REMOTE+'/'+xc)).replace('PREVIOUS',repr(previous))
        run(CFG['python_base']+' -c '+shlex.quote(code))
    # Validate every frozen source input before committing a scheduler submission.
    source=RECORD/'snapshot'/xc/'recovery-01'/step
    inputs=json.loads((source/'input-hashes.json').read_text())
    expected='\n'.join(sha+'  '+REMOTE+'/'+xc+'/recovery-01/'+step+'/'+file for file,sha in inputs.items())+'\n'
    result=subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','sha256sum -c -'],
                          input=expected.encode(),capture_output=True,check=True)
    args=['sbatch','--parsable','--partition='+P['partition'],'--exclude='+','.join(placement['excluded_nodes']),'--account='+P['account'],'--qos='+P['qos'],
          '--nodes='+str(resources['nodes']),'--ntasks='+str(resources['ranks']),
          '--ntasks-per-node='+str(resources['ranks_per_node']),'--cpus-per-task=1',
          '--mem='+str(resources['memory_GiB_per_node'])+'G','--time='+placement['default_walltime'],
          '--job-name=ti2-'+xc+'-'+step,'--chdir='+REMOTE,'--output='+REMOTE+'/'+name+'-%j.out',
          *([REMOTE+'/'+launcher_driver[0],launcher_driver[1],xc,step] if launcher_driver else [REMOTE+'/production-step.sbatch',xc,step])]
    intent={'utc':datetime.now(timezone.utc).isoformat(),'command':args,'state':'submission_intent',
            'source_input_hashes':inputs,'policy_sha256':hashlib.sha256((ROOT/'config'/policy_name).read_bytes()).hexdigest(),
            'source_hashes':source_hashes,
            'placement_policy_sha256':hashlib.sha256((ROOT/'config/vasp-placement.json').read_bytes()).hexdigest()}
    save(name+'-intent.json',intent)
    response=run('set -o pipefail; cd '+shlex.quote(REMOTE)+'; '+shlex.join(args)+' | tee '+shlex.quote(name+'.jobid')).strip()
    job=response.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous submission: '+response)
    save(name+'-submission.json',dict(intent,state='submitted',job_id=job))
    print(xc,step,job,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit']);p.add_argument('xc',nargs='?');p.add_argument('step',nargs='?');a=p.parse_args()
    if a.action=='stage': stage()
    else: submit(a.xc,a.step)
