from pathlib import Path
import json, csv, collections, importlib.util
from pypdf import PdfReader
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
OUT=Path(__file__).parent
inv=json.loads((OUT/'inventory.json').read_text())
st=[];tdm=[]
for f in inv:
    p=ROOT/f['path']
    if p.name!='summary.json': continue
    if 'stacie_pipeline' in str(p):
        d=json.loads(p.read_text())
        for T,ms in d['temperatures'].items():
            for m,rs in ms.items():
                for r in rs['points']:
                    st.append(dict(r,source=f['path']))
    elif '/results/viscosity/' in f['path'] and p.parent.name=='04_viscosity_analysis':
        d=json.loads(p.read_text())
        for r in d['concentrations']:
            tdm.append(dict(r,dft=d['dft_method'],source=f['path']))
# The aggregate TDM summaries omit several completed case results. Use each
# case summary as the authoritative detailed result and retain the source.
tdm=[]
for f in inv:
    p=ROOT/f['path']
    if p.name=='summary.json' and '/04_viscosity_analysis/c_Ti_' in f['path']:
        d=json.loads(p.read_text());m=p.parents[2].name
        tdm.append(dict(d,dft=m,source=f['path']))
st.sort(key=lambda r:(r['T'],r['dft'],r['c_Ti']))
for r in st:
    match=[t for t in tdm if t['T']==r['T'] and t['dft']==r['dft'] and t['c_Ti']==r['c_Ti']]
    if len(match)==1:
        r['tdm_mPa_s']=match[0]['viscosity_mPa_s']
        r['tdm_delta_pct']=100*(r['tdm_mPa_s']/r['eta_mPa_s']-1)
    print(f"ST {r['T']} {r['dft']:4} cTi={r['c_Ti']:.2f} eta={r['eta_mPa_s']:.5f} +- {r['eta_std_mPa_s']:.5f} neff={r['neff']:.1f} reps={r['n_replicas']} dt={r['timestep_ps']} steps={r['n_steps']} TDM={r.get('tdm_mPa_s',float('nan')):.4f} diff%={r.get('tdm_delta_pct',float('nan')):.1f}")
(OUT/'viscosity_merged.json').write_text(json.dumps(st,indent=2),encoding='utf8')
(OUT/'tdm_merged.json').write_text(json.dumps(tdm,indent=2),encoding='utf8')
stkeys={(r['T'],r['dft'],r['c_Ti']) for r in st}
print('TDM only',[(r['T'],r['dft'],r['c_Ti']) for r in tdm if (r['T'],r['dft'],r['c_Ti']) not in stkeys])
print('ST duplicate keys',[(k,n) for k,n in collections.Counter((r['T'],r['dft'],r['c_Ti']) for r in st).items() if n>1])
for host in ['cluster','wsl2']:
    fs=[f for f in inv if f'results-scripts-logs/models_check_{host}/results/' in f['path']]
    print('PROPERTIES',host,collections.Counter(f['path'].split('/')[3] for f in fs))
    pressures=[f for f in fs if Path(f['path']).name.startswith('pressure_')]
    print('pressure_files',host,len(pressures),'bytes',sum(f['bytes'] for f in pressures))
print('PACKAGES', {k:bool(importlib.util.find_spec(k)) for k in ['ase','matplotlib','scipy','pypdfium2','stacie']})
p=ROOT/'TiAl-papers/viscosity/STACIE.pdf'
doc=PdfReader(p)
texts=[page.extract_text() or '' for page in doc.pages]
(OUT/'STACIE.txt').write_text('\n'.join(f'\nPAGE {i+1}\n{s}' for i,s in enumerate(texts)),encoding='utf8')
print('STACIE PAPER',len(texts),'pages',texts[0][:3500])
