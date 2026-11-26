"""Verify preservation, source identity, PDF references and page layout renders."""
from pathlib import Path
import hashlib,json,re,zipfile
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[3]
MAN=ROOT/'manuscript';OUT=Path(__file__).resolve().parent
BUILD=MAN/'build/revision-nov01-phonons'
SRC=ROOT/'phonons/results/figure2-six-panels-20261101-02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return p.read_text(encoding='utf-8').replace('\r\n','\n')
with zipfile.ZipFile(OUT/'before_revision.zip') as z:
    old={n.replace('\\','/'):z.read(n) for n in z.namelist()}
unchanged_figures=['testset_parity','density_ti','melting_ti','liquid_density','mixing_enthalpy','viscosity']
metadata_regenerated=[]
for f in unchanged_figures:
    p='figures/'+f+'.pdf';current=(MAN/p).read_bytes()
    if old[p]!=current:
        # The viscosity heartbeat can regenerate an identical PDF while this
        # manuscript revision is running. Accept only a creation-date change.
        norm=lambda b:re.sub(rb'/CreationDate\s*\(D:[^)]+\)',b'/CreationDate (normalized)',b)
        assert norm(old[p])==norm(current),p
        metadata_regenerated.append(p)
for p in ['sections/testset_metrics_table.tex','sections/introduction.tex']:
    assert old[p].decode('utf-8-sig').replace('\r\n','\n')==read(MAN/p),p
results=read(MAN/'sections/results.tex')
old_results=old['sections/results.tex'].decode('utf-8-sig').replace('\r\n','\n')
boundary='The temperature-dependent density of elemental Ti'
assert results.split(boundary)[1]==old_results.split(boundary)[1]
assert results.split('Figure~\\ref{fig:phonons} compares')[0]==old_results.split('The harmonic phonon dispersions')[0]
assert sha(MAN/'figures/phonons_alpha_ti.pdf')==sha(SRC/'figure2_phonons_dft_dpa4.pdf')
phonon_text=results.split('Figure~\\ref{fig:phonons} compares')[1].split(boundary)[0]
assert not re.search(r'checkpoint|\b(?:5k|15k|20k)\b',phonon_text,re.I)
assert r'\begin{figure*}[t]' in phonon_text

report={'preservation':{'six_other_figures_content_identical':True,'PDF_creation_date_only_changes':metadata_regenerated,'other_result_sections_identical':True,'test_metrics_table_identical':True},
 'source_figure_sha256':sha(SRC/'figure2_phonons_dft_dpa4.pdf')}
for name in ['main','supporting_materials']:
    path=BUILD/(name+'.pdf'); reader=PdfReader(path)
    texts=[p.extract_text() or '' for p in reader.pages]
    log=read(BUILD/(name+'.log'))
    assert not any(t in log for t in ['Overfull','Missing character','undefined references','undefined citations'])
    assert not any('??' in t for t in texts)
    dest=OUT/'rendered'/name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
    if name=='main':
        for i in range(1,8):assert any(f'FIG. {i}.' in t for t in texts),i
        fig_page=next(i for i,t in enumerate(texts) if 'FIG. 2.' in t)
        for tag in ['(a) PBE','(b) PBE','(c) PBEsol','(d) PBEsol','(e)','(f)']:
            assert tag in texts[fig_page],tag
        assert 'DPA4-finetuned' in texts[fig_page]
        report['figure2_page']=fig_page+1
    doc=pdfium.PdfDocument(str(path));thumbs=[]
    for i,page in enumerate(doc):
        im=page.render(scale=1.6).to_pil().convert('RGB');im.save(dest/f'page_{i+1}.png')
        im.thumbnail((360,510));tile=Image.new('RGB',(380,550),'white');tile.paste(im,((380-im.width)//2,26))
        ImageDraw.Draw(tile).text((12,7),f'{name} -- {i+1}',fill='black');thumbs.append(tile);page.close()
    doc.close()
    for start in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1140,1100),'#cccccc')
        for j,im in enumerate(thumbs[start:start+6]):sheet.paste(im,((j%3)*380,(j//3)*550))
        sheet.save(dest/f'contact_{start//6+1}.png')
    report[name]={'pages':len(texts),'pdf_sha256':sha(path),'undefined_references':False,'overfull_boxes':False,
        'warnings':[l for l in log.splitlines() if 'Warning' in l],
        'page_starts':[t[:100].replace('\n',' ') for t in texts]}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
