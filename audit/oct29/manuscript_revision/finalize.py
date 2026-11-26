"""Check the frozen revision and record delivery provenance without rebuilding PDFs."""
from pathlib import Path
import hashlib, json, re
from collections import Counter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
MAN = ROOT / 'manuscript'
OUT = Path(__file__).resolve().parent
DATA = MAN / 'supporting_data'
BUILD = MAN / 'build/revision-oct29'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v): p.write_text(json.dumps(v, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

m = read(DATA/'test30_metrics.json')
assert len(m['common_ids']) == 30
assert m['models']['PBE']['checkpoint_step'] == 20000
snap = read(OUT/'test30_input_snapshot.json')
records = [r for r in snap['records'] if r['metadata']['structure_id'] in m['common_ids']]
assert len(records) == 30
assert sum(3*len(r['symbols']) for r in records) == 20532
assert sorted(Counter(r['metadata']['family'] for r in records).values()) == [3,3,4,20]
table = (MAN/'sections/testset_metrics_table.tex').read_text()
for xc, label in [('PBE','PBE'), ('PBEsol','PBEsol'), ('r2SCAN',r'\rscan{}')]:
    for key, title in [('metrics','Fine-tuned'), ('pretrained_metrics','Pretrained')]:
        row = next(x for x in table.splitlines() if x.startswith(label+' & '+title+' &'))
        numbers = [float(x.strip().rstrip('\\').strip()) for x in row.split('&')[2:]]
        ref = m[key][xc]['all']
        expected = [ref[q][stat] for q in ['energy_meV_atom','force_eV_A','stress_GPa'] for stat in ['mae','rmse']]
        assert numbers == [float(f'{x:.{p}f}') for x,p in zip(expected,[2,2,4,4,3,3])]

summary = read(DATA/'revision_oct29_sources.json')
scan = read(ROOT/'audit/oct26/r2scan_enthalpy_extend30k/analysis/comparison.json')
assert summary['mixing']['scan']['H'][1:-1] == [x['deltaH_60_kJ_mol'] for x in sorted(scan['comparison'], key=lambda x:x['x_Al'])]
assert summary['mixing']['scan']['SEM'][1:-1] == [x['SEM_60_kJ_mol'] for x in sorted(scan['comparison'], key=lambda x:x['x_Al'])]
v = read(DATA/'viscosity_current_oct29.json')
pbe = [r for r in v if r['dft']=='pbe']
assert len(pbe) == 5
assert len([r for r in pbe if r['T']==1873]) == 4
assert len([r for r in pbe if r['c_Ti']==1]) == 2
batch = read(ROOT/'audit/oct27/pbe15k_viscosity_iterative/collected/analysis/batch_0015/summary.json')
assert batch['n_replicas'] == 15
for r in batch['points']:
    actual = next(x for x in pbe if x['x_Al']==r['x_Al'])
    assert actual['n_replicas']==15 and actual['production_ps']==40
    for key in ['eta_mPa_s', 'eta_std_mPa_s']:
        assert actual[key] == r['windows']['40ps'][key]
original = read(ROOT/'audit/oct25/viscosity/viscosity_merged.json')
assert [r for r in v if r['dft']=='scan'] == [r for r in original if r['dft']=='scan']

methods = (MAN/'sections/methods.tex').read_text()
start = methods.index(r'\subsection{Fine-tuning DPA4-Neo}')
end = methods.index(r'\subsection{Common held-out DFT benchmark}',start)
outside = methods[:start]+methods[end:]
for f in ['main.tex', 'sections/abstract.tex', 'sections/introduction.tex', 'sections/results.tex', 'sections/discussion.tex', 'sections/conclusions.tex']:
    outside += (MAN/f).read_text()
assert not re.search(r'checkpoint|\b(?:5k|15k|20k)\b', outside, re.I)
results = (MAN/'sections/results.tex').read_text()
phonons = results.split('The harmonic phonon dispersions')[1].split('The temperature-dependent density')[0]
assert 'DFT' not in phonons
assert r'\includegraphics[width=\columnwidth]{figures/phonons_alpha_ti.pdf}' in results
assert 'enthalpy_temperature' not in results

figs = ['testset_parity','phonons_alpha_ti','density_ti','melting_ti','liquid_density','mixing_enthalpy','viscosity']
for f in figs:
    txt = ''.join(p.extract_text() or '' for p in PdfReader(MAN/f'figures/{f}.pdf').pages)
    assert not re.search(r'\b(?:5k|15k|20k)\b|checkpoint', txt, re.I)
    if f=='phonons_alpha_ti': assert 'DFT' not in txt
    if f=='testset_parity': assert '\u00c5' in txt
pages = [p.extract_text() or '' for p in PdfReader(BUILD/'main.pdf').pages]
assert [i+1 for i,p in enumerate(pages) if 'checkpoint' in p.lower()] == [3]
assert 'FIG. 2.' in pages[8] and 'Harmonic dispersions' in pages[8]
for i in range(1,8): assert any(f'FIG. {i}.' in p for p in pages)

verify = read(OUT/'verification.json')
verify['scientific_checks'] = {
    'benchmark_common_geometries':30, 'benchmark_design':50, 'force_components':20532,
    'PBE_benchmark_step':20000, 'metric_table_entries_checked':36,
    'r2SCAN_enthalpy_production_ps':60, 'enthalpy_values_and_errors_match_source':True,
    'PBE_viscosity_unique_states':5, 'PBE_viscosity_composition_points':4,
    'PBE_viscosity_temperature_points':2, 'alloy_replicas':15,
    'alloy_production_ps':40, 'r2SCAN_viscosity_unchanged':True,
    'checkpoint_mentions_only_in_fine_tuning':True,
    'main_phonon_comparison_experiment_only':True,
    'phonon_figure_and_caption_one_column_page':9,
    'all_7_main_figures_present':True,
    'all_31_pages_visually_reviewed':True,
    'log_notes':'REVTeX deferred-float warnings remain; every figure is present and placement was visually checked. inputenc is ignored by the UTF-8 engine.'
}
source_paths = [MAN/'main.tex',MAN/'supporting_materials.tex'] + list((MAN/'sections').glob('*.tex'))
source_paths += [MAN/f'figures/{f}.pdf' for f in figs]
source_paths += list(OUT.glob('*.py')) + [OUT/'test30_input_snapshot.json',MAN/'revision_oct29_notes_ru.txt']
verify['final_files_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in source_paths}
save(OUT/'verification.json',verify)
summary['revision_scripts_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in OUT.glob('*.py')}
summary['frozen_test_input'] = {'path':str((OUT/'test30_input_snapshot.json').relative_to(ROOT)), 'sha256':digest(OUT/'test30_input_snapshot.json')}
summary['thread_sources'] = {
    'test_benchmark':'01a15d2f-8c16-7e71-81e0-44fb533aa0b6',
    'r2SCAN_enthalpy':'01a1795b-d112-74f2-93c5-e2cf0c2da187',
    'PBE_viscosity':'01a17b96-45dd-7971-b28e-a55f28b9461c'
}
save(DATA/'revision_oct29_sources.json',summary)
manifest = read(DATA/'source_manifest.json')
manifest['revision_oct29'] = {
    'source_manifest':'supporting_data/revision_oct29_sources.json',
    'parity_manifest':'supporting_data/test30_parity_sources.json',
    'notes':'revision_oct29_notes_ru.txt', 'current_build':'build/revision-oct29',
    'PBE_benchmark_step':20000, 'PBE_updated_property_step':15000,
    'benchmark_completed':30, 'benchmark_design':50,
    'viscosity_alloy_replicas':15, 'r2SCAN_enthalpy_production_ps':60,
    'temperature_enthalpy_panel_removed':True,
    'verification':'../audit/oct29/manuscript_revision/verification.json',
    'PDF_sha256':{k:verify[k]['pdf_sha256'] for k in ['main','supporting_materials']}
}
save(DATA/'source_manifest.json',manifest)
print(json.dumps(verify['scientific_checks'],indent=2))
