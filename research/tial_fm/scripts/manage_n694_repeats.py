"""Idempotent submission and status for the eight authorized N694 repetitions."""
from pathlib import Path
import argparse,json,sys
import manage_expansion as m
from prepare_n694_repeats import RUN,REMOTE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','submit','status']);a=ap.parse_args()
    m.RUN=RUN
    if a.action=='status':
        m.main();return
    if a.action=='preflight':
        code='''from pathlib import Path
import ast,hashlib,json,subprocess,sys
root=Path(REMOTE)
manifest=json.loads((root/'FILES_SHA256.json').read_text())
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
for p in root.rglob('*.py'):ast.parse(p.read_text())
for p in root.rglob('*.sbatch'):subprocess.run(['bash','-n',str(p)],check=True)
assert not any(p.name=='test' for p in (root/'data').rglob('*'))
plan=json.loads((root/'plan.json').read_text());assert len(plan['cases'])==8
assert {c['seed'] for c in plan['cases']}=={20,30} and {c['N'] for c in plan['cases']}=={694}
for case in plan['cases']:
    subprocess.run([sys.executable,str(root/'verify_case.py')],cwd=root/'training-plan01/cases'/case['name'],capture_output=True,check=True)
sys.path.insert(0,str(root));import aggregate_expansion as agg
parents=[]
for name,expected in plan['existing_seed10'].items():
    result,rows=agg.collect(Path(expected['remote_path']),expected)
    parents.append(dict(name=name,N=expected['N'],model_sha256=result['model_sha256'],structures=len(rows)))
for module,expected in [('deepmd.pt.train.training','2ee355877da3ba10bf213b5653b3b858caa7795d846528c47ab1d7881dc3d090'),('deepmd.pt.model.descriptor.sezm_nn.lora','03591f701e266b9db0e395c53765391183dda38e9c8786d1d0a304a14c5a1644')]:
    p=Path('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/lib/python3.12/site-packages')/(module.replace('.','/')+'.py')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print(json.dumps(dict(status='PASS',files=len(manifest),case_count=8,all_case_input_verifiers_executed=True,parent_checks=parents)))
'''.replace('REMOTE',repr(REMOTE))
        result=json.loads(m.remote_python(code));(RUN/'remote-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return
    assert json.loads((RUN/'remote-preflight.json').read_text())['status']=='PASS'
    common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1']
    j=m.submit(REMOTE,'N694-training',common+['--partition=v100','--gres=gpu:1','--cpus-per-task=4','--mem=40G','--time=04:00:00','--array=0-7%1','--job-name=TiAl-N694-seeds','--output=training-plan01/N694-%A_%a.log','training-plan01/train.sbatch','N694'])
    q=m.submit(REMOTE,'N694-QC',common+['--partition=apollo','--cpus-per-task=4','--mem=12G','--time=00:10:00','--array=0-7%4','--dependency=afterany:'+j,'--job-name=TiAl-N694-seeds-QC','--output=training-plan01/N694-QC-%A_%a.log','training-plan01/qc.sbatch','N694'])
    g=m.submit(REMOTE,'N694-aggregate',common+['--partition=apollo','--cpus-per-task=1','--mem=8G','--time=00:10:00','--dependency=afterany:'+q,'--job-name=TiAl-N694-3seeds','--output=N694-aggregate-%j.log','aggregate.sbatch','N694'])
    (RUN/'submission-summary.json').write_text(json.dumps(dict(N694_training=j,N694_QC=q,N694_aggregate=g,max_new_GPU_concurrency=1,automatic_followup='validation/harmonics/numerical QC/physical diagnostics/three-seed aggregate',dependency_policy='afterany preserves visibility of failed candidates'),indent=2)+'\n')

if __name__=='__main__':main()
