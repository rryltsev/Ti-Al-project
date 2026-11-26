from pathlib import Path
import json, collections, csv
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
OUT=Path(__file__).parent
files=[]
for base in ['results-scripts-logs','MD-results/mixing_enthaphy','MD-results/density_vs_x']:
    for p in (ROOT/base).rglob('*'):
        if p.is_file():
            files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
(OUT/'inventory.json').write_text(json.dumps(files,indent=2),encoding='utf8')
groups=collections.defaultdict(list)
for f in files:
    parts=f['path'].split('/')
    group='/'.join(parts[:2])
    groups[group].append(f)
for g,fs in groups.items():
    print(g,len(fs),round(sum(f['bytes'] for f in fs)/1e9,3),'GB',collections.Counter(Path(f['path']).suffix for f in fs).most_common(8))
    if g.startswith('MD-results'):
        for f in fs:
            print(' ',f)
            if f['path'].endswith('.csv'):
                print((ROOT/f['path']).read_text(encoding='utf-8-sig')[:1200])
for category in ['stacie_pipeline_cluster','stacie_pipeline_wsl2']:
    fs=[f for f in files if category in f['path'] and f['path'].endswith('summary.json')]
    print('\nSUMMARY FILES',category,len(fs))
    for f in fs:
        d=json.loads((ROOT/f['path']).read_text(encoding='utf8'))
        print(f['path'],str(d)[:2000])
for prop in ['mixing_enthalpy','viscosity','density']:
    fs=[f for f in files if f'/results/{prop}/' in f['path'] and f['path'].endswith('summary.json')]
    print('\nPROPERTY',prop,len(fs))
    if fs:
        print(fs[0]['path'],(ROOT/fs[0]['path']).read_text(encoding='utf8')[:4500])
