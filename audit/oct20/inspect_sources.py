from pathlib import Path
import json, re, hashlib, collections
import pypdfium2 as pdfium
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
OUT=Path(__file__).parent
for rel,page,label in [('TiAl-papers/density/2025 Thermophysical Properties of Al–Ti Binary Liquid Alloys.pdf',26,'density_table7'),('TiAl-papers/viscosity/2025_T-Al_Takeda.pdf',8,'viscosity_table3'),('TiAl-papers/mixing enhalpy/witusiewicz2008.pdf',7,'mixing_reference')]:
    doc=pdfium.PdfDocument(ROOT/rel)
    doc[page].render(scale=1.6).to_pil().save(OUT/f'{label}.png')
inv=json.loads((OUT/'inventory.json').read_text())
for host in ['cluster','wsl2']:
    base=ROOT/f'results-scripts-logs/models_check_{host}'
    print('\nHOST',host)
    for f in inv:
        if f'models_check_{host}/results/viscosity/' not in f['path'] or not f['path'].endswith('/03_stress_autocorr/checkpoint.json'):continue
        d=json.loads((ROOT/f['path']).read_text())
        print(f['path'].split('/AlTi/')[1],str(d)[:1100])
    fs=[f for f in inv if f'models_check_{host}/results/viscosity/' in f['path'] and Path(f['path']).name.startswith('pressure_')]
    counts=collections.Counter(f['path'].split('/03_stress_autocorr/')[0].split('/AlTi/')[1]+'/'+f['path'].split('/03_stress_autocorr/')[1].split('/')[0] for f in fs)
    print('RAW PRESSURE COVERAGE',dict(counts))
    if fs:
        p=ROOT/fs[0]['path'];print('SAMPLE PRESSURE',str(p),p.read_text().splitlines()[:5])
    fs=[f for f in inv if f'models_check_{host}/results/viscosity/' in f['path'] and Path(f['path']).suffix=='.lammps']
    if fs:
        p=ROOT/fs[0]['path'];print('SAMPLE INPUT',str(p),p.read_text()[:7000])
    for extra in ['elastic','point_defects','planar_defects','phonons','melting']:
        fs=[f for f in inv if f'models_check_{host}/results/{extra}/' in f['path'] and Path(f['path']).suffix=='.json']
        if fs: print(extra,'JSONS',len(fs),'SAMPLE',fs[0]['path'],(ROOT/fs[0]['path']).read_text()[:350])
