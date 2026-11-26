"""Move completed-run intermediate GRACE checkpoints to a verified local archive."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,tarfile
import cluster_pilot as c
import manage_expansion as m
from prepare_grace_mask_ablation import RUN,REMOTE,PARENT_REMOTE

OUT=RUN/'storage-recovery';OUT.mkdir(exist_ok=True)
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def file_sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','archive','release']);a=ap.parse_args()
    if a.action=='plan':
        code=r'''from pathlib import Path
import hashlib,json
roots=[Path(PARENT),Path(REMOTE)];result={}
for root in roots:
 for w in sorted((root/'cases').iterdir()):
  if not (w/'evaluation-summary.json').exists():continue
  t=json.loads((w/'training-summary.json').read_text());assert t['status']=='PASS' and t['steps']==10000
  assert json.loads((w/'evaluation-summary.json').read_text())['status']=='PASS'
  for n,h in t['hashes'].items():assert hashlib.sha256((w/'final-checkpoint'/n).read_bytes()).hexdigest()==h
  folder=w/'seed/10/checkpoints';files={}
  for f in sorted(folder.iterdir()):
   assert f.is_file() and not f.is_symlink() and f.name in ['checkpoint.index','checkpoint.data-00000-of-00001','checkpoint.best_test_loss.index','checkpoint.best_test_loss.data-00000-of-00001'],str(f)
   files[f.name]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  assert len(files)==4
  result[root.name+'__'+w.name]=dict(folder=str(folder),files=files,final_checkpoint_hashes=t['hashes'])
assert len(result)==5
print(json.dumps(result))
'''.replace('PARENT',repr(PARENT_REMOTE)).replace('REMOTE',repr(REMOTE))
        data=json.loads(m.remote_python(code));dump(OUT/'archive-plan.json',data);print('PLAN',len(data),'completed cases',sum(v['bytes'] for d in data.values() for v in d['files'].values()),'bytes');return
    plan=json.loads((OUT/'archive-plan.json').read_text())
    if a.action=='archive':
        verified={}
        for key,record in plan.items():
            arc=OUT/(key+'.tar.gz')
            if not arc.exists():
                code=r'''from pathlib import Path
import hashlib,json,sys,tarfile
d=RECORD;p=Path(d['folder'])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as t:
 for name,meta in d['files'].items():
  f=p/name;assert f.stat().st_size==meta['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'];t.add(f,arcname=name)
'''.replace('RECORD',repr(record))
                part=arc.with_suffix(arc.suffix+'.part')
                with part.open('wb') as dest:
                    process=subprocess.run(c.cluster_ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',m.PY+' -c '+shlex.quote(code)],stdout=dest,stderr=subprocess.PIPE,timeout=1800)
                assert process.returncode==0,process.stderr.decode(errors='replace')
                part.rename(arc)
            names=[]
            with tarfile.open(arc,'r:gz') as t:
                for item in t:
                    assert item.isfile() and item.name in record['files'];h=hashlib.sha256();stream=t.extractfile(item)
                    for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
                    assert item.size==record['files'][item.name]['bytes'] and h.hexdigest()==record['files'][item.name]['sha256'];names.append(item.name)
            assert set(names)==set(record['files'])
            verified[key]=dict(path=str(arc),archive_sha256=file_sha(arc),bytes=arc.stat().st_size,all_payload_hashes_verified=True)
            dump(OUT/'verified-archives.json',verified);print('VERIFIED',key,arc.stat().st_size,flush=True)
        return
    verified=json.loads((OUT/'verified-archives.json').read_text());assert verified and set(verified).issubset(plan)
    # Release only archives already verified locally; an ongoing transfer of
    # another case remains untouched.
    plan={k:plan[k] for k in verified}
    for v in verified.values():assert v['all_payload_hashes_verified'] and file_sha(Path(v['path']))==v['archive_sha256']
    code=r'''from pathlib import Path
import json,hashlib,subprocess
plan=PLAN;verified=VERIFIED;allowed=[Path(PARENT).resolve(),Path(REMOTE).resolve()];released=[]
queue=subprocess.check_output(['squeue','-h','-j','20990597,20990675','-o','%T'],text=True);assert not any(s in queue for s in ['RUNNING','COMPLETING','PENDING'])
for key,d in plan.items():
 folder=Path(d['folder']).resolve();assert any(folder.is_relative_to(root/'cases') for root in allowed) and folder.parts[-3:]==('seed','10','checkpoints')
 work=folder.parent.parent.parent;summary=json.loads((work/'training-summary.json').read_text());assert summary['status']=='PASS'
 for n,h in d['final_checkpoint_hashes'].items():assert hashlib.sha256((work/'final-checkpoint'/n).read_bytes()).hexdigest()==h
 for name,meta in d['files'].items():
  f=folder/name;assert f.parent==folder and not f.is_symlink()
  if f.exists():
   assert f.is_file() and f.stat().st_size==meta['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
   f.unlink();released.append(dict(path=str(f),bytes=meta['bytes']))
 (folder/'LOCAL_ARCHIVE.json').write_text(json.dumps(dict(local_archive=verified[key],files=d['files'],final_checkpoint_preserved=True),indent=2)+'\n')
print(json.dumps(dict(status='PASS',released=released,bytes_released=sum(f['bytes'] for f in released),quota=subprocess.check_output(['quota','-s'],text=True,timeout=15))))
'''.replace('PLAN',repr(plan)).replace('VERIFIED',repr(verified)).replace('PARENT',repr(PARENT_REMOTE)).replace('REMOTE',repr(REMOTE))
    result=json.loads(m.remote_python(code))
    previous=json.loads((OUT/'released.json').read_text()) if (OUT/'released.json').exists() else {'released':[]}
    combined={f['path']:f for f in previous['released']+result['released']};result['released']=list(combined.values());result['bytes_released']=sum(f['bytes'] for f in result['released'])
    dump(OUT/'released.json',result);print('RELEASED cumulative',result['bytes_released'],'bytes');print(result['quota'])

if __name__=='__main__':main()
