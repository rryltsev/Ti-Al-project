from pathlib import Path
import json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
code='''import json,pathlib,re,subprocess
root=pathlib.Path(REMOTE).resolve()
assert json.loads((root/'harmonics/summary.json').read_text())['status']=='PASS'
assert (root/'harmonics/mode_diagnostics.json').is_file()
records=[]
for jid,name in [('20988442','TiAl-FM-H0'),('20988444','TiAl-H0-modes')]:
    text=subprocess.check_output(['scontrol','show','job','-o',jid],text=True)
    assert 'UserId=u3129(' in text and ('JobName='+name+' ') in text and 'JobState=PENDING ' in text,text
    wd=re.search(r'WorkDir=(\\S+)',text).group(1);assert pathlib.Path(wd).resolve()==root
    subprocess.run(['scancel',jid],check=True);records.append(dict(job_id=jid,name=name,reason='Equivalent H0 and mode diagnostics already completed in job20988449; cancel only redundant pending tasks.'))
print(json.dumps(records))
'''.replace('REMOTE',repr(remote))
out=c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code)).decode();(RUN/'redundant-H0-cancellation.json').write_text(out);print(out)
