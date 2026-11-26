from pathlib import Path
import subprocess
import json
import argparse

ROOT=Path(__file__).resolve().parents[3]
MAN=ROOT/'manuscript'
parser=argparse.ArgumentParser()
parser.add_argument('--build-dir',default='build/revision-oct26-enthalpy2000')
args=parser.parse_args()
BUILD=MAN/args.build_dir
BUILD.mkdir(parents=True,exist_ok=True)
TEX=ROOT/'audit/latex-tools/tectonic.exe'
results={}
for name in ['main','supporting_materials']:
    commands=[([str(TEX),'-X','compile','--outdir',str(BUILD),'--outfmt','pdf','--untrusted','--keep-logs',name+'.tex'],MAN)]
    for i,(command,cwd) in enumerate(commands):
        p=subprocess.run(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=120)
        (BUILD/f'{name}-pass{i+1}.txt').write_text(p.stdout,encoding='utf-8')
        if p.returncode:
            print(p.stdout[-6000:]);raise SystemExit(p.returncode)
    log=(BUILD/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    results[name]={'pdf':str(BUILD/(name+'.pdf')),'warnings':[l for l in log.splitlines() if any(w in l for w in ['Warning','Overfull','Missing character','undefined'])]}
print(json.dumps(results,indent=2))
