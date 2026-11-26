"""Prepare eight immutable N694 repetitions from the qualified seed10 package."""
from pathlib import Path
import ast, copy, hashlib, json, shutil, tarfile

ROOT=Path(__file__).resolve().parents[1]
ID='dpa4-n694-seeds-20261107-01'
RUN=ROOT/'runs'/ID; OUT=RUN/'package'
PARENT=ROOT/'runs/dpa4-replicates-full-20261106-02'; SRC=PARENT/'package'
REMOTE='/home6/u3129/tial-fm-research/'+ID

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')

def main():
    parent=load(SRC/'plan.json'); oldremote=parent['remote_root']
    OUT.mkdir(parents=True,exist_ok=False)
    # Verify immutable source inputs before reusing the successful pipeline.
    for name,digest in load(SRC/'FILES_SHA256.json').items():assert sha(SRC/name)==digest,name
    for folder in ['label-audit','baseline','harmonics','harmonic_inputs','runtime-mask-preflight']:
        shutil.copytree(SRC/folder,OUT/folder)
    shutil.copytree(SRC/'data/N694',OUT/'data/N694')
    for p in SRC.iterdir():
        if p.is_file() and p.name not in ['FILES_SHA256.json','plan.json','preflight.json','protocol.json','aggregate.sbatch','dataset-files-N200.json']:
            shutil.copyfile(p,OUT/p.name)
    assert load(OUT/'dataset-files-N694.json')=={p.relative_to(OUT).as_posix():sha(p) for p in (OUT/'data/N694').rglob('*') if p.is_file()}
    work=OUT/'training-plan01';work.mkdir()
    for p in (SRC/'training-plan01').glob('*.py'):shutil.copyfile(p,work/p.name)
    cases=[]
    for seed in [20,30]:
        for xc in ['PBE','r2SCAN']:
            for method in ['lora16','full']:
                original=SRC/'training-plan01/cases'/f'{xc}-{method}-N694-seed10'
                name=f'{xc}-{method}-N694-seed{seed}';dest=work/'cases'/name
                cfg=load(original/'input.json');previous=copy.deepcopy(cfg)
                cfg['training']['seed']=seed
                for key in ['training_data','validation_data']:
                    cfg['training'][key]['systems']=[s.replace(oldremote,REMOTE) for s in cfg['training'][key]['systems']]
                normalized=copy.deepcopy(cfg);normalized['training']['seed']=10
                for key in ['training_data','validation_data']:
                    normalized['training'][key]['systems']=[s.replace(REMOTE,oldremote) for s in normalized['training'][key]['systems']]
                assert normalized==previous,'Unexpected recipe drift'
                dump(dest/'input.json',cfg)
                case=load(original/'case.json');case.update(name=name,seed=seed,data_root=REMOTE+'/data/N694',configuration_sha256=sha(dest/'input.json'))
                dump(dest/'case.json',case);cases.append(case)
    picker='GROUP="$1"\ntest "$GROUP" = N694\nCASES=('+' '.join(c['name'] for c in cases)+')\n'
    for filename in ['train.sbatch','qc.sbatch']:
        s=(SRC/'training-plan01'/filename).read_text()
        begin=s.index('GROUP="$1"');end=s.index('CASE_NAME=',begin)
        s=(s[:begin]+picker+s[end:]).replace(oldremote,REMOTE)
        (work/filename).write_text(s,encoding='utf8',newline='\n')
    (OUT/'aggregate.sbatch').write_text((SRC/'aggregate.sbatch').read_text().replace(oldremote,REMOTE),encoding='utf8',newline='\n')
    # Retain both completed sizes at seed10, for a 12-case N694 aggregate and size control.
    existing=copy.deepcopy(parent['existing_seed10'])
    for c in parent['cases']:
        if c['N']!=694:continue
        t=load(PARENT/'collected/training-plan01/cases'/c['name']/'evaluation/summary.json')
        existing[c['name']]=dict(remote_path=oldremote+'/training-plan01/cases/'+c['name'],model_sha256=t['model_sha256'],seed=10,N=694)
    plan=copy.deepcopy(parent)
    plan.update(id=ID,remote_root=REMOTE,parent_run=str(PARENT),cases=cases,groups={'N694':[c['name'] for c in cases]},existing_seed10=existing,seeds_N694=[10,20,30],
        estimated_GPU_hours={'N694_8_additional':21.7083333333,'total':21.7083333333},
        estimate_qualification='Twice the measured allocated GPU time of the four completed N694 seed10 cases; excludes queue delay.',
        followup='Per-case final-checkpoint validation, harmonics, derivative/export/mask QC; three-seed N694 aggregate and matched seed10 size control after completion.')
    dump(OUT/'plan.json',plan)
    protocol=load(SRC/'protocol.json');protocol.update(id=ID,remote_root=REMOTE)
    protocol['pilot'].update(N=[694],seeds_N694=[10,20,30])
    protocol['authorization']['current_request']='2026-11-07: launch N694 seeds20/30 for LoRA/full PBE/r2SCAN; prepare GRACE-2L pilot'
    dump(OUT/'protocol.json',protocol)
    checks=load(SRC/'preflight.json');checks.update(configuration_count=8,recipe_identical_to_N694_seed10_except_seed_and_run_paths=True)
    checks['datasets']={'694':checks['datasets']['694']};dump(OUT/'preflight.json',checks)
    agg=(OUT/'aggregate_expansion.py').read_text(encoding='utf8')
    agg=agg.replace('Completed seed10 N200 is the common reference for both seed and size comparisons.','Completed seed10 cases at both sizes anchor replication and size comparisons.')
    agg=agg.replace('Same N200 membership across seeds','Same train membership within each size across seeds')
    agg=agg.replace('N694 has seed10 only, paired to N200 seed10; same50 epochs, different step counts.','N694 has seeds10/20/30; size control compares seed10 at the same50 epochs and different step counts.')
    agg=agg.replace('N694 seed10 — контроль размера данных; вывод о его разбросе между seeds пока невозможен.','N694 включает seeds 10/20/30; seed10 также служит парным контролем размера данных.')
    (OUT/'aggregate_expansion.py').write_text(agg,encoding='utf8',newline='\n')
    for p in OUT.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    assert len(cases)==8 and not any(p.name=='test' for p in (OUT/'data').rglob('*'))
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    archive=RUN/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
    print(json.dumps(dict(status='PASS',cases=len(cases),archive_bytes=archive.stat().st_size,sha256=sha(archive))))

if __name__=='__main__':main()
