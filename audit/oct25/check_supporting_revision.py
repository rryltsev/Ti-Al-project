"""Render both deliverables and check supporting references and generated tables."""
from pathlib import Path
import json
import re
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / 'manuscript'
BUILD = MAN / 'build/revision-oct25-checkpoint-supporting'
OUT = ROOT / 'audit/oct25/supporting_revision_qa'
tex = (MAN / 'supporting_materials.tex').read_text(encoding='utf-8')
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex))
for table in ['checkpoint_error_rows.tex', 'checkpoint_stress_rows.tex']:
    for line in (MAN / 'supporting_data' / table).read_text().splitlines():
        assert line in tex, (table, line)

reports = {}
for name in ['main', 'supporting_materials']:
    pdf = BUILD / f'{name}.pdf'
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf)
    pages_text = [p.extract_text() or '' for p in reader.pages]
    (out / 'text.txt').write_text('\n\n'.join(pages_text), encoding='utf-8')
    log = (BUILD / f'{name}.log').read_text(encoding='utf-8')
    report = {'pages': len(reader.pages), 'unresolved': sum(p.count('??') for p in pages_text),
              'overfull': 'Overfull' in log, 'missing_characters': 'Missing character' in log,
              'missing_labels': sorted(refs-labels) if name == 'supporting_materials' else [],
              'page_starts': [p[:110].replace('\n', ' ') for p in pages_text]}
    doc = pdfium.PdfDocument(str(pdf))
    thumbs = []
    for i in range(len(doc)):
        page = doc[i]
        im = page.render(scale=1.5).to_pil().convert('RGB')
        im.save(out / f'page_{i+1}.png')
        im.thumbnail((360, 510))
        tile = Image.new('RGB', (380, 550), 'white')
        tile.paste(im, ((380-im.width)//2, 26))
        ImageDraw.Draw(tile).text((12, 7), f'{name} -- {i+1}', fill='black')
        thumbs.append(tile)
        page.close()
    doc.close()
    for start in range(0, len(thumbs), 6):
        sheet = Image.new('RGB', (1140, 1100), '#cccccc')
        for j, im in enumerate(thumbs[start:start+6]):
            sheet.paste(im, ((j%3)*380, (j//3)*550))
        sheet.save(out / f'contact_{start//6+1}.png')
    reports[name] = report
(OUT / 'report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
print(json.dumps(reports, indent=2))
