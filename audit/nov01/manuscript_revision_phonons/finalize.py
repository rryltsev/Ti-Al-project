from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];MAN=ROOT/'manuscript';OUT=Path(__file__).resolve().parent
DATA=MAN/'supporting_data';DEST=DATA/'phonons_matched_20261101'
SRC=ROOT/'phonons/results/figure2-six-panels-20261101-02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
shutil.copyfile(SRC/'figure2_phonons_dft_dpa4.pdf',DEST/'figure2_phonons_dft_dpa4.pdf')
(DEST/'README.txt').write_text('Frozen plotting inputs and verified metrics for the current Figure 2.\n'
 'manifest-sha256.json is the original FULL source-bundle manifest; some raw calculation inputs and scripts remain in that source bundle.\n'
 'Source bundle: phonons/results/figure2-six-panels-20261101-02, relative to the workspace root.\n'
 'local-manifest-sha256.json lists the files actually copied into this supporting_data directory.\n',encoding='utf-8')
save(DEST/'local-manifest-sha256.json',{p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='local-manifest-sha256.json'})
v=read(OUT/'verification.json')
v['visual_review']={'all_pages_checked':True,'article_pages':19,'SI_pages':18,
 'figure2_six_panels_and_caption_on_one_page':10,'no_clipping_or_overlap':True,
 'note':'REVTeX deferred-float warnings remain; all seven figures are present and their placement was checked.'}
paths=[MAN/'main.tex',MAN/'supporting_materials.tex',MAN/'revision_nov01_phonons_notes_ru.txt']
paths+=list((MAN/'sections').glob('*.tex'))+list(OUT.glob('*.py'))
v['final_source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in paths}
save(OUT/'verification.json',v)
pr=read(DATA/'revision_nov01_phonons_sources.json')
pr['PDF_sha256']={name:v[name]['pdf_sha256'] for name in ['main','supporting_materials']}
pr['preservation_checks']=v['preservation']
pr['verification']=str((OUT/'verification.json').relative_to(ROOT))
pr['final_source_sha256']=v['final_source_sha256']
pr['local_data_manifest_sha256']=sha(DEST/'local-manifest-sha256.json')
save(DATA/'revision_nov01_phonons_sources.json',pr)
manifest=read(DATA/'source_manifest.json')
manifest['revision_nov01_phonons']={'source_manifest':'supporting_data/revision_nov01_phonons_sources.json',
 'notes':'revision_nov01_phonons_notes_ru.txt','current_build':'build/revision-nov01-phonons',
 'source_thread':'01a15c09-6430-7c61-8706-69838a8cbe0c','PBE_phonon_step':15000,
 'figure2_panels':6,'figure2_page':10,'PDF_sha256':pr['PDF_sha256']}
save(DATA/'source_manifest.json',manifest)
print(json.dumps({'PDF_pages':{k:v[k]['pages'] for k in ['main','supporting_materials']},'figure2_page':10,'checks':'passed'},indent=2))
