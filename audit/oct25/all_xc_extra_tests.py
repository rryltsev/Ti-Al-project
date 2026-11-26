"""Freeze additional validated tests outside common20 and submit one bounded CPU array."""
from pathlib import Path
import sys,json,shlex,hashlib,time,base64,tarfile,io
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=HERE/'all_xc_audit';OUT=BASE/'extra_tests';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'audit/oct20/deps'),str(ROOT/'testset/dft')]
import numpy as np
import cluster_test50 as c
read=lambda p:json.loads(Path(p).read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2))
REMOTE=read(BASE/'protocol.json')['remote']+'-extra-tests';action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists()
    snap=read(BASE/'test_snapshot.json');manifest=snap['manifest'];records=[]
    requests=[(xc,r) for xc in ['PBE','PBEsol','r2SCAN'] for r in manifest
              if r['structure_id'] not in snap['common_ids'] and snap['statuses'][xc].get(r['structure_id'],{}).get('state')=='validated']
    code=('from pathlib import Path\nimport json,hashlib\np=Path("/home6/u3129/tial-testsets/test50-dft-20261021-01")\n'
      +'cfg=json.loads((p/"protocol.json").read_text()); source=Path(cfg["dataset_root"]); out=[]\n'
      +'for xc,r in '+repr(requests)+':\n'
      +' d=p/"calculations"/xc/r["structure_id"]; b=(d/"label.json").read_bytes(); lab=json.loads(b); outputs=json.loads((d/"output-manifest.json").read_text())\n'
      +' assert hashlib.sha256(b).hexdigest()==outputs["label.json"] and lab["scf_converged"] and lab["ionic_calculations"]==1\n'
      +' pos=(source/r["poscar"]).read_bytes(); assert hashlib.sha256(pos).hexdigest()==r["poscar_sha256"]==lab["input_sha256"]["POSCAR"]\n'
      +' out.append({"functional":xc,"metadata":r,"label":lab,"label_sha256":outputs["label.json"],"poscar":pos.decode()})\n'
      +'print(json.dumps(out))')
    records=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('snapshot.json',records)
    from ase.io import read as ase_read
    cfg=read(BASE/'protocol.json');cfg['remote']=REMOTE;cfg['datasets']={};tasks=[];files={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        rr=[r for r in records if r['functional']==xc]
        if not rr:continue
        g={};l={};rows=[]
        for i,r in enumerate(rr):
            at=ase_read(io.StringIO(r['poscar']),format='vasp');lab=r['label']
            g[f'coords_{i}']=at.positions;g[f'cell_{i}']=at.cell.array;g[f'types_{i}']=np.array([['Al','Ti'].index(s) for s in at.get_chemical_symbols()],dtype='i4')
            l[f'energy_{i}']=lab['energy_eV'];l[f'forces_{i}']=np.array(lab['forces_eV_A']);l[f'virial_{i}']=-np.array(lab['stress_eV_A3'])*lab['volume_A3']
            rows.append({'i':i,'id':r['metadata']['structure_id'],'family':r['metadata']['family'],'n_atoms':len(at),
                         'x_Ti':at.get_chemical_symbols().count('Ti')/len(at),'volume_A3_atom':at.get_volume()/len(at)})
        np.savez_compressed(OUT/(xc+'-geometries.npz'),**g);np.savez_compressed(OUT/(xc+'-labels.npz'),**l);save(xc+'-rows.json',rows)
        n=xc+'-geometries.npz';files[n]=(OUT/n).read_bytes();cfg['datasets'][xc]={'geometry_file':n,'geometry_sha256':sha(files[n]),'n':len(rows)}
        for key,m in cfg['models'].items():
            if m['functional']==xc:tasks.append({'id':len(tasks),'key':key,'kind':'inference','indices':list(range(len(rows))),'split':'extra_test'})
    save('tasks.json',tasks);cfg['tasks_sha256']=sha((OUT/'tasks.json').read_bytes());save('protocol.json',cfg)
    files.update({n:(OUT/n).read_bytes() for n in ['tasks.json','protocol.json']})
    files['all_xc_worker.py']=(HERE/'all_xc_worker.py').read_bytes()
    files['job.sbatch']=(HERE/'pbe_phonons_dft_geometry/job.sbatch').read_text().replace('python dft_geometry_phonons_remote.py','python all_xc_worker.py').encode()
    hashes={n:sha(b) for n,b in files.items()};save('staging-intent.json',{'remote':REMOTE,'files':hashes})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==s for n,s in h.items()); print("verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1',
          '--cpus-per-task=4','--mem=12G','--time=00:10:00','--array=0-'+str(len(tasks)-1)+'%2','--no-requeue',
          '--job-name=xc-extra-tests','--chdir='+REMOTE,'--output='+REMOTE+'/run-%A_%a.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit();save('submission.json',{'job_id':jid,'remote':REMOTE});print('Submitted',jid,'tests',len(records),'tasks',len(tasks))
else:
    jid=read(OUT/'submission.json')['job_id'];tasks=read(OUT/'tasks.json')
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn');(OUT/'accounting.txt').write_text(acct);print(acct)
    rec={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({f.name:f.read_text(errors="replace")[-3000:] for f in p.glob("run-*.out")}))'
    save('logs.json',json.loads(c.cluster.run('python3 -c '+shlex.quote(code))))
    done=[t['id'] for t in tasks if rec.get(jid+'_'+str(t['id']),['',''])[1:3]==['COMPLETED','0:0'] and not (OUT/'results'/str(t['id'])/'summary.json').exists()]
    if done:
        code=('import pathlib,hashlib,base64,json,io,tarfile; p=pathlib.Path('+repr(REMOTE+'/results')+'); files=[]\n'
             +'for i in '+repr(done)+': files.extend(f for f in (p/str(i)).glob("*") if f.is_file())\n'
             +'b=io.BytesIO()\nwith tarfile.open(fileobj=b,mode="w:gz") as tf:\n'
             +' for f in files: tf.add(f,arcname=str(f.relative_to(p)),recursive=False)\n'
             +'v=b.getvalue(); print(json.dumps({"data":base64.b64encode(v).decode(),"sha256":hashlib.sha256(v).hexdigest()}))')
        payload=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));b=base64.b64decode(payload['data']);assert sha(b)==payload['sha256']
        with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
            for m in tf.getmembers():
                assert m.isfile();p=OUT/'results'/m.name;assert OUT.resolve() in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
                data=tf.extractfile(m).read()
                if p.exists():assert p.read_bytes()==data
                else:p.write_bytes(data)
        save('collection-'+str(done[0])+'.json',{'ids':done,'archive_sha256':payload['sha256']});print('Collected',done)
