"""Read-only monitoring of production-02; retain scheduler and raw progress evidence."""
from datetime import datetime, timezone
import json
import shlex
from cluster import RECORD, REMOTE, CFG, run

records = [json.loads(p.read_text()) for p in sorted(RECORD.glob('*production0[23456]*-submission.json'))]
job_ids = [r['job_id'] for r in records]
if not job_ids:
    raise RuntimeError('No recorded production-02 submissions')
ids = ','.join(job_ids)
accounting = run('sacct -X -nP -j ' + ids + ' --format=JobIDRaw,State,Elapsed,ExitCode,CPUTimeRAW')
failed_ids = {line.split('|')[0] for line in accounting.splitlines()
              if len(line.split('|')) > 1 and line.split('|')[1] in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']}
launcher_failures = {}
for r in records:
    if r['job_id'] not in failed_ids: continue
    output = next((arg.split('=',1)[1] for arg in r.get('command',[]) if arg.startswith('--output=')), None)
    if output:
        path = output.replace('%j', r['job_id'])
        launcher_failures[r['job_id']] = run('if test -f '+shlex.quote(path)+'; then tail -40 '+shlex.quote(path)+'; fi')
queue = run('squeue -h -j ' + ids + ' -o "%i|%T|%M|%R"')
remote_code = '''
import json, re, time
from pathlib import Path
root = Path(REMOTE_ROOT)
result = {}
for xc in ['PBE','PBEsol','r2SCAN']:
    result[xc] = {}
    for p in sorted((root / xc).glob('production-*/*')):
        if not p.is_dir(): continue
        d = {'files':{}}
        status = p/'status.json'
        if status.exists():
            try: d['status'] = json.loads(status.read_text())
            except (ValueError, OSError): d['status'] = 'temporarily_unreadable'
        for name in ['checkpoint-guard.json', 'checkpoint-request.json', 'restart-provenance.json']:
            if (p/name).exists():
                try: d[name] = json.loads((p/name).read_text())
                except (ValueError, OSError): d[name] = 'temporarily_unreadable'
        for name in ['CHGCAR','TAUCAR','WAVECAR']:
            if (p/name).exists(): d['files'][name] = {'bytes':(p/name).stat().st_size}
        for name in ['OUTCAR','vasp.stdout','OSZICAR','vasprun.xml']:
            f = p/name
            if not f.exists(): continue
            stat = f.stat()
            d['files'][name] = {'bytes':stat.st_size,'mtime_epoch':stat.st_mtime,
                               'age_seconds':time.time()-stat.st_mtime}
            if name == 'vasprun.xml': continue
            with f.open('rb') as stream:
                stream.seek(max(0,stat.st_size-524288))
                text = stream.read().decode(errors='replace')
            d[name+'_tail'] = text.splitlines()[-10:]
            if name == 'OUTCAR':
                d['recent_loop_wall_seconds'] = re.findall(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)',text)[-8:]
                d['ediff_reached'] = 'aborting loop because EDIFF is reached' in text
                d['normal_termination'] = 'General timing and accounting informations' in text
                if (p/'checkpoint-request.json').exists() and not d['ediff_reached']:
                    d['scientific_state'] = 'checkpoint_requested_or_saved_NOT_converged'
            if name == 'vasp.stdout':
                d['recent_scf_lines'] = [s for s in text.splitlines() if re.match(r'^(DAV|RMM|CG)\s*:',s)][-5:]
                d['fatal_indicators'] = [s for s in text.splitlines() if re.search(r'oom_kill|Out Of Memory|segmentation fault|MPI_ABORT|DUE TO TIME LIMIT|VERY BAD NEWS',s,re.I)][-10:]
        key = p.name if p.parent.name == 'production-02' else p.parent.name + '/' + p.name
        result[xc][key] = d
print(json.dumps(result))
'''.replace('REMOTE_ROOT', repr(REMOTE))
progress = json.loads(run(CFG['python_base'] + ' -c ' + shlex.quote(remote_code)))
# Preserve raw worker state, but expose authoritative scheduler state alongside it.
scheduler_states={row.split('|')[0]:row.split('|')[1] for row in accounting.splitlines() if len(row.split('|'))>1}
for xc,steps_data in progress.items():
    for step_data in steps_data.values():
        worker=step_data.get('status')
        if not isinstance(worker,dict): continue
        scheduler=scheduler_states.get(worker.get('job_id'))
        step_data['scheduler_state']=scheduler
        if scheduler and scheduler not in ('RUNNING','PENDING','COMPLETING','CONFIGURING'):
            step_data['effective_state']=scheduler
            if worker.get('state')=='running':step_data['worker_status_stale']=True
steps = ','.join(j + '.0' for j in job_ids)
memory = run('sstat -j ' + steps + ' --format=JobID,MaxRSS,AveCPU -P')
record = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'job_ids':job_ids,
          'accounting':accounting,'queue':queue,'progress':progress,'memory':memory,
          'launcher_failures':launcher_failures}
folder = RECORD/'monitor'
folder.mkdir(exist_ok=True)
(folder/'production-latest.json').write_text(json.dumps(record,indent=2)+'\n')
with (folder/'production-history.jsonl').open('a') as out:
    out.write(json.dumps(record)+'\n')
print(json.dumps(record,indent=2))
