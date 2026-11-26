"""Read-only snapshot of complete frames from running coexistence MD."""
from pathlib import Path
import sys,json,shlex,zlib,base64,hashlib,datetime
sys.argv=['melting_cluster.py','inspect','attempt2']
import melting_cluster as c
code='''import pathlib,json,time,zlib,base64,hashlib
p=pathlib.Path(REMOTE);out={'unix':time.time(),'stages':[],'dumps':{}}
for folder in sorted(p.glob('T*K/*')):
 if not folder.is_dir():continue
 item={'path':str(folder.relative_to(p))}
 f=folder/'summary.json'
 if f.exists():item['summary']=json.loads(f.read_text())
 f=folder/'trajectory.lammpstrj'
 if f.exists():
  with f.open('rb') as fp:
   size=f.stat().st_size;data=fp.read(size)
  chunks=data.split(b'ITEM: TIMESTEP\\n')[1:];complete=[]
  for chunk in chunks:
   lines=chunk.splitlines(keepends=True)
   try:
    n=int(lines[2]);step=int(lines[0])
    if len(lines)>=8+n and lines[7+n].endswith(b'\\n'):complete.append((step,b'ITEM: TIMESTEP\\n'+b''.join(lines[:8+n])))
   except (ValueError,IndexError):pass
  item['last_complete_step']=complete[-1][0] if complete else None
  if folder.name=='melt' and complete:
   kept=[b for step,b in complete if step%500==0];snapshot=b''.join(kept);key=str(folder.relative_to(p))
   out['dumps'][key]={'zlib_base64':base64.b64encode(zlib.compress(snapshot)).decode(),'sha256':hashlib.sha256(snapshot).hexdigest(),'frames':len(kept),'last_sampled_step':max(s for s,b in complete if s%500==0)}
 out['stages'].append(item)
print(json.dumps(out))
'''.replace('REMOTE',repr(c.remote))
d=json.loads(c.run('python3 -c '+shlex.quote(code)))
stamp=datetime.datetime.fromtimestamp(d['unix'],datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=c.OUT/'snapshots'/stamp;out.mkdir(parents=True,exist_ok=False)
for key,v in d['dumps'].items():
 b=zlib.decompress(base64.b64decode(v.pop('zlib_base64')));assert hashlib.sha256(b).hexdigest()==v['sha256'];p=out/(key.replace('/','_')+'.lammpstrj');p.write_bytes(b);v['local_file']=str(p.resolve())
(out/'snapshot.json').write_text(json.dumps(d,indent=2));(c.OUT/'latest-snapshot.json').write_text(json.dumps({'directory':str(out.resolve()),'unix':d['unix']}))
print(json.dumps(d,indent=2))
