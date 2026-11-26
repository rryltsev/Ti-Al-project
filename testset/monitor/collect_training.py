"""Collect the recorded training inference job with byte-level checksum checks."""
from pathlib import Path
import base64,hashlib,json,shlex,sys,time
ROOT=Path(__file__).resolve().parent
CAL=ROOT/'alignment/train-offsets-v1'
sys.path.insert(0,str(ROOT.parent/'dft'))
import cluster_test50 as c

submission=json.loads((CAL/'submission.json').read_text())
job=submission['job_id'].split(';')[0];remote=submission['remote']
accounting=c.cluster.run('sacct -j '+job+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW -Pn')
(CAL/'accounting.txt').write_text(accounting,encoding='utf-8')
rows={r.split('|')[0]:r.split('|') for r in accounting.splitlines()}
for task in range(8):
    r=rows[job+'_'+str(task)]
    assert r[1]=='COMPLETED' and r[2]=='0:0', r
names=[f'result-{i}.{ext}' for i in range(8) for ext in ['json','sha256']]
code="import pathlib,json,hashlib,base64; root=pathlib.Path("+repr(remote)+"); names="+repr(names)+"; print(json.dumps({n:{'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest(),'base64':base64.b64encode((root/n).read_bytes()).decode()} for n in names}))"
files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
assert set(files)==set(names)
for name,info in files.items():
    data=base64.b64decode(info['base64'],validate=True)
    assert hashlib.sha256(data).hexdigest()==info['sha256']
    path=CAL/name
    if path.exists(): assert path.read_bytes()==data
    else: path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
result={'status':'collected','job_id':job,'remote':remote,'unix':time.time(),'files_sha256':{n:v['sha256'] for n,v in files.items()}}
(CAL/'collection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'job_id':job,'verified_files':len(files)}))
