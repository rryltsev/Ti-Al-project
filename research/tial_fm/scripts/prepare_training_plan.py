from pathlib import Path
import copy,hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';BASE=RUN/'package';OUT=RUN/'training-plan01'
SOURCE=Path('C:/Users/Roman Ryltsev/Desktop/STARTUPS/Ti-Al-X/TiAl-Project/DPA4-finetune/TiAl_tune_pbe_DPA4-shard-Neo-OMat24-v20260805.json')
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False);recipe=json.loads(SOURCE.read_text());proto=json.loads((BASE/'protocol.json').read_text());remote=proto['remote_root']
    shutil.copyfile(SOURCE,OUT/'historical-recipe.json')
    cases=[]
    for xc in ['PBE','r2SCAN']:
        for method in ['lora16','full']:
            name=xc+'-'+method;target=OUT/'cases'/name;target.mkdir(parents=True);cfg=copy.deepcopy(recipe)
            if method=='full':del cfg['model']['lora']
            cfg['model']['use_compile']=False
            cfg['training']['zero_stage']=0
            cfg['training']['stat_file']='statistics.hdf5'
            cfg['training']['stat_file_mode']='update'
            cfg['training']['save_dir']='checkpoints'
            cfg['training']['save_freq']=1000
            cfg['training']['max_ckpt_keep']=3
            cfg['training']['disp_freq']=100
            cfg['training']['num_epochs']=50
            for part,key in [('train','training_data'),('validation','validation_data')]:
                cfg['training'][key]['systems']=[remote+'/data/'+xc+'/'+part+'/'+p.name for p in sorted((BASE/'data'/xc/part).iterdir())]
                cfg['training'][key]['batch_size']='auto:128'
            cfg['validating']['save_best_dir']='checkpoints_best'
            cfg['validating']['compiled_infer']=False
            cfg['validating']['tf32_infer']=False
            cfg['validating']['amp_infer']=False
            dump(target/'input.json',cfg)
            case=dict(name=name,XC=xc,method=method,N=200,validation=149,seed=10,source=proto['model'],split_sha256=proto['split_sha256'],subsets_sha256=proto['subsets_sha256'],source_recipe_sha256=sha(SOURCE),configuration_sha256=sha(target/'input.json'),intent='Fixed 50 epochs; single GPU, no distributed sharding; native historical HybridMuon and loss recipe; N200-only statistics; test not supplied to trainer or checkpoint selection.')
            dump(target/'case.json',case);cases.append(case)
    for name in ['train_observer.py','evaluate_pilot.py','evaluate_pilot_harmonics.py']:shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    common=(BASE/'baseline.sbatch').read_text().split('export CUDA_VISIBLE_DEVICES=""')[0]
    script=common+f'''RUN={remote}
CASES=(PBE-lora16 PBE-full r2SCAN-lora16 r2SCAN-full)
CASE_NAME=${{CASES[$SLURM_ARRAY_TASK_ID]}}
cd "$RUN/training-plan01/cases/$CASE_NAME"
test ! -e attempt.json
python -c 'import json,os,socket; open("attempt.json","x").write(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],array=os.environ["SLURM_ARRAY_JOB_ID"],task=os.environ["SLURM_ARRAY_TASK_ID"],host=socket.gethostname())))'
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > hardware.txt
python ../../train_observer.py --pt train input.json --finetune {proto['model']['path']} > train.log 2>&1
python ../../evaluate_pilot.py > evaluate.log 2>&1
export PYTHONPATH=/home6/u3129/tial-phonons/envs/phonopy-4.8.2
/home6/u3129/mlip-research/envs/kplib_111/bin/python ../../evaluate_pilot_harmonics.py > harmonics.log 2>&1
'''
    (OUT/'train.sbatch').write_text(script,encoding='utf8',newline='\n')
    dump(OUT/'plan.json',dict(cases=cases,resource_request=dict(partition='v100',account='imet-2026-2',gpus=1,cpus=4,mem='40G',time='04:00:00',array='0-3%1'),epochs=50,estimated_steps_each=10000,estimate_note='Batch auto:128 implies one frame for 216-256 atoms; actual scheduler steps and frame/atom exposures recorded. Walltime estimate updated at first100steps. Four-hour allocation is a per-job scheduler ceiling, not expected duration.',gates=['B0 PASS','H0 PASS','zero-step source restoration up to composition energy bias','live vs merged equivalence','runtime mask and frozen weights check'],test_access=False))
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    archive=RUN/'training-plan01.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'training-package.json',dict(remote_root=remote+'/training-plan01',path=str(archive),sha256=sha(archive)))
    print('PREPARED',len(cases),'cases')
if __name__=='__main__':main()
