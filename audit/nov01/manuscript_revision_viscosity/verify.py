"""Check scoped manuscript changes and render every page of the revised main PDF."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[3]
MAN = ROOT / 'manuscript'
OUT = Path(__file__).resolve().parent
BUILD = MAN / 'build/revision-nov01-viscosity'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def text(data):
    return data.decode('utf-8-sig').replace('\r\n', '\n')

with zipfile.ZipFile(OUT / 'before_revision.zip') as archive:
    before = {name: archive.read(name) for name in archive.namelist()}

sections = {f'sections/{s}.tex' for s in ['methods', 'results', 'abstract', 'discussion', 'conclusions']}
allowed = sections | {'figures/viscosity.pdf'}
changed = []
for name, data in before.items():
    current = (MAN / name).read_bytes()
    if data != current:
        assert name in allowed, f'Unrequested change: {name}'
        changed.append(name)
assert sections <= set(changed)

boundary = '\\subsection{Liquid viscosity}\n'
results = (MAN / 'sections/results.tex').read_text(encoding='utf-8')
old_results = text(before['sections/results.tex'])
assert results.split(boundary)[0] == old_results.split(boundary)[0]
closing = 'Taken together, the present benchmarks'
assert results.split(closing)[1] == old_results.split(closing)[1]
methods = (MAN / 'sections/methods.tex').read_text(encoding='utf-8')
old_methods = text(before['sections/methods.tex'])
old_methods = old_methods.replace(
    'the complete Ti temperature-dependent density series, Ti coexistence, and all viscosity points shown in the main article.',
    'the complete Ti temperature-dependent density series, and Ti coexistence.')
assert methods.split(boundary)[0] == old_methods.split(boundary)[0]
viscosity_text = results.split(boundary)[1] + methods.split(boundary)[1]
for term in [r'checkpoint', r'\b15k\b', r'\b5k\b', r'\b15000\b', r'\b5000\b', r'cutoff', r'TDM', r'LorentzModel']:
    assert not re.search(term, viscosity_text, re.IGNORECASE), term
for section in ['abstract', 'conclusions', 'discussion']:
    content = (MAN / f'sections/{section}.tex').read_text(encoding='utf-8')
    assert not re.search(r'checkpoint|\b(?:5k|15k|20k)\b', content, re.IGNORECASE)

source = ROOT / 'audit/nov01/viscosity_arrhenius_article/viscosity_arrhenius_eta_T.pdf'
# The neighbouring viscosity updater rewrites this asset periodically with the
# same selected figure. Allow only its PDF creation timestamp to differ.
normalize_pdf_date = lambda value: re.sub(rb'/CreationDate\s*\(D:[^)]+\)', b'/CreationDate (normalized)', value)
assert normalize_pdf_date((MAN / 'figures/viscosity.pdf').read_bytes()) == normalize_pdf_date(source.read_bytes())
manifest = json.loads((source.parent / 'figure_manifest.json').read_text(encoding='utf-8'))
fit_checks = {}
for name, supplied in manifest['fits'].items():
    points = supplied['input_points']
    temperatures = np.array([row['T'] for row in points], dtype=float)
    eta = np.array([row['eta_mPa_s'] for row in points])
    se = np.array([row['eta_std_mPa_s'] for row in points])
    design = np.column_stack([np.ones(len(points)), 1000 / temperatures])
    sigma_log = se / eta
    parameters = np.linalg.lstsq(design / sigma_log[:, None], np.log(eta) / sigma_log, rcond=None)[0]
    residuals = (np.log(eta) - design @ parameters) / sigma_log
    assert np.allclose(parameters, [supplied['a'], supplied['B_K'] / 1000], atol=1e-10)
    assert np.max(np.abs(residuals)) < 0.5
    fit_checks[name] = {'chi2': float(residuals @ residuals), 'dof': len(points)-2,
        'maximum_standardized_residual': float(np.max(np.abs(residuals)))}

path = BUILD / 'main.pdf'
pages = [page.extract_text() or '' for page in PdfReader(path).pages]
log = (BUILD / 'main.log').read_text(encoding='utf-8')
assert not re.search(r'Overfull|Missing character|undefined references|undefined citations', log)
assert not any('??' in page for page in pages)
figure_pages = {}
for i in range(1, 8):
    matching = [j+1 for j, page in enumerate(pages) if f'FIG. {i}.' in page]
    assert len(matching) == 1, (i, matching)
    figure_pages[i] = matching[0]
figure7 = pages[figure_pages[7]-1]
for label in ['(a)', '(b)', 'Arrhenius', 'Takeda', 'Ishikawa']:
    assert label in figure7, label

dest = OUT / 'rendered'
dest.mkdir(exist_ok=True)
(dest / 'text.txt').write_text('\n\n'.join(pages), encoding='utf-8')
doc = pdfium.PdfDocument(str(path))
thumbs = []
for i, page in enumerate(doc):
    im = page.render(scale=1.6).to_pil().convert('RGB')
    im.save(dest / f'page_{i+1}.png')
    im.thumbnail((360, 510))
    tile = Image.new('RGB', (380, 550), 'white')
    tile.paste(im, ((380-im.width)//2, 26))
    ImageDraw.Draw(tile).text((12, 7), f'Main manuscript -- {i+1}', fill='black')
    thumbs.append(tile)
    page.close()
doc.close()
for start in range(0, len(thumbs), 6):
    sheet = Image.new('RGB', (1140, 1100), '#cccccc')
    for j, im in enumerate(thumbs[start:start+6]):
        sheet.paste(im, ((j % 3)*380, (j // 3)*550))
    sheet.save(dest / f'contact_{start//6+1}.png')

report = {
    'changed_files': changed,
    'all_other_source_files_and_assets_unchanged': True,
    'SI_unchanged': True,
    'other_result_sections_unchanged': True,
    'figure7_matches_selected_source_excluding_creation_date': True,
    'figure_sha256': sha(source),
    'arrhenius_fit_checks': fit_checks,
    'pages': len(pages),
    'figure_pages': figure_pages,
    'main_pdf_sha256': sha(path),
    'undefined_references': False,
    'overfull_boxes': False,
    'warnings': [line for line in log.splitlines() if 'Warning' in line],
}
(OUT / 'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
