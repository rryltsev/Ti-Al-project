"""Check data preservation, source hashes, exported points, and article contents."""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
OUT = WORK/'deliverables/2026-11-06/Fig07_viscosity_KRP'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
before = read(HERE/'before/figure_profile.json')
after = read(WORK/'audit/nov01/viscosity_article_combined/figure_profile.json')
manifest = read(OUT/'figure_manifest.json')
model = read(OUT/'krp_data.json')

checks = {}
for key in ['composition', 'temperature', 'references', 'selection', 'rendering']:
    assert before[key] == after[key], key
    checks[key+'_unchanged'] = True
assert read(HERE/'before/figure_manifest.json')['arrhenius_fits'] == manifest['arrhenius_fits']
checks['panel_b_arrhenius_fits_unchanged'] = True
assert manifest['composition'] == after['composition'] and manifest['temperature'] == after['temperature']
assert manifest['krp']['data_sha256'] == hashlib.sha256((OUT/'krp_data.json').read_bytes()).hexdigest()
for name, expected in model['sources_sha256'].items():
    assert hashlib.sha256((WORK/name).read_bytes()).hexdigest() == expected, name
checks['source_hashes_valid'] = True

with (OUT/'KRP_curves_1873K.csv').open(encoding='utf-8') as f:
    curves = list(csv.DictReader(f))
assert len(curves) == 3003
with (OUT/'enthalpy_inputs_and_KRP_nodes.csv').open(encoding='utf-8') as f:
    nodes = list(csv.DictReader(f))
assert len(nodes) == 33
eta_ti = model['pure_ti']['eta_mPa_s']
eta_al = model['pure_al']['eta_mPa_s']
for row in curves:
    x, h, eta = [float(row[k]) for k in ['x_Al', 'deltaH_kJ_mol', 'eta_KRP_mPa_s']]
    if x == 0:
        assert math.isclose(eta, eta_ti, rel_tol=1e-12)
    elif x == 1:
        assert math.isclose(eta, eta_al, rel_tol=1e-12)
    ideal = eta_ti**(1-x)*eta_al**x
    assert eta >= ideal*(1-1e-12)
    # Recover the input enthalpy from the exported viscosity, independently of interpolation.
    recovered = -3*model['R_J_mol_K']*model['viscosity_temperature_K']*math.log(eta/ideal)/1000
    assert math.isclose(recovered, h, abs_tol=1e-10)
checks['all_3003_exported_values_pass_inverse_enthalpy_check'] = True
checks['all_three_curves_have_correct_pure_component_limits'] = True

text = (HERE/'article_text.txt').read_text(encoding='utf-8')
compact_text = ''.join(text.split())
for token in ['KRP', '2.61', '2.54', '2.83', '0.497', '3.457', 'Assael', 'Kozlov', 'FIG. 7.']:
    assert ''.join(token.split()) in compact_text, token
assert '??' not in text
checks['article_contains_figure_method_results_and_references'] = True
checks['latex_undefined_references'] = False
report = dict(date='2026-11-06', checks=checks,
              figure_pdf_sha256=hashlib.sha256((OUT/'viscosity_combined.pdf').read_bytes()).hexdigest(),
              standalone_panel_pdf_sha256=hashlib.sha256((OUT/'figure7a_krp.pdf').read_bytes()).hexdigest())
(HERE/'verification.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
print(json.dumps(report, indent=2))
