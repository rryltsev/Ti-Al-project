"""Read-only SCF diagnostics for the first three r2SCAN crystal tasks."""
import json
from pathlib import Path
import shlex
import cluster_test50 as c

CODE = r'''
import hashlib,json,pathlib,re,subprocess,time
root=pathlib.Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
now=time.time()
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def read(p):
 if not p.exists():return None
 data=p.read_bytes();st=p.stat()
 return {'text':data.decode('utf-8',errors='replace'),'size':len(data),'mtime':st.st_mtime,'age_seconds':now-st.st_mtime,'sha256':hashlib.sha256(data).hexdigest()}
def digest(xc,sid,details=False):
 p=root/'calculations'/xc/sid
 out=read(p/'OUTCAR');osz=read(p/'OSZICAR')
 if out is None:return {'missing':True,'path':str(p)}
 text=out['text']
 loops=[float(x) for x in re.findall(r'^\s*LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)',text,re.M)]
 iters=[int(x) for x in re.findall(r'Iteration\s+1\(\s*(\d+)\)',text)]
 elapsed=re.findall(r'Elapsed time \(sec\):\s*([\d.]+)',text)
 result={'structure_id':sid,'functional':xc,'path':str(p),'outcar_size':out['size'],'outcar_mtime':out['mtime'],'outcar_age_seconds':out['age_seconds'],'outcar_sha256':out['sha256'],'loop_seconds':loops,'iteration_headers':iters,'converged_marker':'aborting loop because EDIFF is reached' in text,'elapsed_seconds':float(elapsed[-1]) if elapsed else None,'oszicar':osz}
 if details:
  result['files']={n:read(p/n) for n in ('INCAR','status.json','input-manifest.json')}
  result['outcar_tail']=text[-14000:]
  result['warnings']=[line for line in text.splitlines() if re.search(r'WARNING|BRMIX|ZBRENT|VERY BAD NEWS|ERROR|Error EDDDAV|ZHEGV|ZPOTRF|FATAL',line)]
  result['settings_lines']=[line for line in text.splitlines() if any(k in line for k in ('EDIFF ','NELM ','NBANDS=','NKPTS =','NIONS =','METAGGA','IALGO','ISMEAR','SIGMA ','NSW ','IBRION'))][:50]
  result['slurm']=run(['scontrol','show','job','20957372_'+str(int(sid.rsplit('-',1)[1])-1),'-o'])
  sd=run(['scontrol','show','job','20957372_'+str(int(sid.rsplit('-',1)[1])-1),'-o'])
  for key in ('StdOut','StdErr'):
   m=re.search(r'\b'+key+r'=(\S+)',sd['stdout'])
   if m:
    fp=pathlib.Path(m.group(1));f=read(fp)
    if f:f['text']=f['text'][-16000:]
    result[key]=f
 return result
result={'unix':now,'clock':run(['date','--iso-8601=seconds']),'targets':{sid:digest('r2SCAN',sid,True) for sid in ('TiAl-test-026','TiAl-test-034','TiAl-test-043')},'completed_r2scan':[],'matching_GGA':[]}
for i in range(1,51):
 sid='TiAl-test-'+str(i).zfill(3)
 p=root/'calculations'/'r2SCAN'/sid/'status.json'
 if p.exists():
  st=json.loads(p.read_text())
  if st.get('state')=='validated' or st.get('status') in ('validated','complete','success','completed'):
   result['completed_r2scan'].append(digest('r2SCAN',sid))
for xc in ('PBE','PBEsol'):
 for sid in ('TiAl-test-026','TiAl-test-034','TiAl-test-043'):
  result['matching_GGA'].append(digest(xc,sid))
result['accounting']=run(['sacct','-X','-j','20957372_25,20957372_33,20957372_42','--format=JobID,State,ExitCode,Elapsed,Start,End,Timelimit','-P'])
print(json.dumps(result))
'''

if __name__ == '__main__':
    data=json.loads(c.cluster.run('python3 -c '+shlex.quote(CODE)))
    out=Path(__file__).resolve().parent/'scf-check-20261025'
    out.mkdir(exist_ok=True)
    path=out/('snapshot-'+str(int(data['unix']))+'.json')
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'saved':str(path),'unix':data['unix'],'targets':{sid:{'slurm':v['slurm'],'last_scf_lines':v['oszicar']['text'].splitlines()[-10:],'loop_seconds_last5':v['loop_seconds'][-5:],'converged_marker':v['converged_marker'],'outcar_age_seconds':v['outcar_age_seconds'],'warnings':v['warnings']} for sid,v in data['targets'].items()},'completed_controls':len(data['completed_r2scan'])},indent=2))
