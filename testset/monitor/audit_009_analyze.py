"""Independent XML/OUTCAR, immutable cache and training-coverage audit of Test009."""
from pathlib import Path
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEST = HERE / 'audits/TiAl-test-009-20261023'
SID = 'TiAl-test-009'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def array(node):
    return np.array([list(map(float, x.text.split())) for x in node.findall('v')])

bundle = read(HERE / 'data/latest.json')
metadata = {r['structure_id']: r for r in bundle['manifest']}
row = metadata[SID]
cache_manifest = read(HERE / 'prediction-manifest.json')
registry = read(HERE / 'model_registry.json')
report = {'created_unix': time.time(), 'structure_id': SID, 'metadata': row,
          'policy': 'Diagnostic only. No model, DFT output, cached prediction, offset or monitor metric is changed.',
          'dft': {}, 'predictions': {}, 'PBE_comparison_rows': []}
source_poscar = HERE.parent / 'TiAl-test50-v1/POSCARS' / ('POSCAR.' + SID)
assert sha(source_poscar) == row['poscar_sha256']
lines = source_poscar.read_text().splitlines()
cell = np.array([list(map(float, x.split())) for x in lines[2:5]])
frac = np.array([list(map(float, x.split()[:3])) for x in lines[8:8 + row['n_atoms']]])
symbols = [s for s, n in zip(lines[5].split(), map(int, lines[6].split())) for _ in range(n)]
for xc in ['PBE', 'PBEsol', 'r2SCAN']:
    d = DEST / xc
    label = read(d / 'label.json')
    inp = read(d / 'input-manifest.json')['input_sha256']
    out = read(d / 'output-manifest.json')
    for name, expected in inp.items():
        actual = read(d / 'POTCAR-metadata.json')['sha256'] if name == 'POTCAR' else sha(d / name)
        assert actual == expected == label['input_sha256'][name], (xc, name)
    for name, expected in out.items():
        assert sha(d / name) == expected, (xc, name)
    assert sha(d / 'POSCAR') == row['poscar_sha256']
    tree = ET.parse(d / 'vasprun.xml').getroot()
    calculations = tree.findall('calculation')
    assert len(calculations) == 1
    calc = calculations[0]
    energies = {x.attrib['name']: float(x.text) for x in calc.findall('energy/i')}
    forces = array(calc.find("varray[@name='forces']"))
    xml_symbols = [rc.find('c').text.strip() for rc in tree.findall("atominfo/array[@name='atoms']/set/rc")]
    assert xml_symbols == symbols
    deltas = []
    for struct in tree.findall('structure') + calc.findall('structure'):
        c = array(struct.find("crystal/varray[@name='basis']"))
        f = array(struct.find("varray[@name='positions']"))
        df = f - frac
        df -= np.rint(df)
        deltas.append({'cell_max_A': float(np.max(np.abs(c - cell))),
                       'position_max_A': float(np.max(np.abs(df @ cell)))})
    assert max(x['cell_max_A'] for x in deltas) < 1e-7
    assert max(x['position_max_A'] for x in deltas) < 1e-6
    assert abs(energies['e_fr_energy'] - label['energy_eV']) < 1e-8
    force_delta = float(np.max(np.abs(forces - np.array(label['forces_eV_A']))))
    assert force_delta < 1e-6
    text = (d / 'OUTCAR').read_text()
    final_toten = float(re.findall(r'free\s+energy\s+TOTEN\s*=\s*([-\d.]+)', text)[-1])
    assert abs(final_toten - label['energy_eV']) < 1e-8
    assert 'aborting loop because EDIFF is reached' in text
    assert 'General timing and accounting informations' in text
    scf = calc.findall('scstep')
    last_dE = float(scf[-1].find("energy/i[@name='e_fr_energy']").text) - float(scf[-2].find("energy/i[@name='e_fr_energy']").text)
    nk = len(tree.findall("kpoints/varray[@name='kpointlist']/v"))
    occ = np.array([float(x.text.split()[1]) for x in calc.findall('.//eigenvalues/array/set/set/set/r')]).reshape(nk, -1)
    report['dft'][xc] = {'energy_TOTEN_eV': final_toten, 'energy_sigma0_eV': energies['e_0_energy'],
        'sigma0_minus_TOTEN_meV_atom': (energies['e_0_energy'] - final_toten) * 1000 / row['n_atoms'],
        'electronic_iterations': len(scf), 'last_dE_eV': last_dE,
        'EDIFF_reached': True, 'normal_termination': True, 'ionic_calculations': len(calculations),
        'geometry_differences': deltas, 'label_vs_XML_force_max_eV_A': force_delta,
        'n_irreducible_kpoints': nk, 'n_bands': occ.shape[1], 'last_band_max_abs_occupation': float(np.max(np.abs(occ[:, -1]))),
        'PAW': read(d / 'POTCAR-metadata.json'), 'input_hashes': inp,
        'diagnostic_keyword_matches': [s.strip() for s in text.splitlines() if re.search(r'warning|VERY BAD|ZBRENT|BRMIX:|ERROR', s, re.I)]}
    for role in [xc, 'pretrained']:
        rel = 'predictions/' + role + '/' + SID + '.json'
        assert sha(HERE / rel) == cache_manifest[rel]
        pred = read(HERE / rel)
        assert pred['poscar_sha256'] == row['poscar_sha256'] and pred['symbols'] == symbols
        if role == xc:
            assert pred['model_sha256'] == registry['models'][xc]['sha256']
        df = np.array(pred['forces_eV_A']) - forces
        report['predictions'][xc + '/' + role] = {'energy_eV': pred['energy_eV'],
            'signed_energy_error_meV_atom': (pred['energy_eV'] - final_toten) * 1000 / row['n_atoms'],
            'force_mae_eV_A': float(np.mean(np.abs(df))), 'force_rmse_eV_A': float(np.sqrt(np.mean(df ** 2))),
            'model_sha256': pred['model_sha256'], 'prediction_sha256': sha(HERE / rel)}

for f in sorted((DEST / 'PBE-labels').glob('*.json')):
    label, meta = read(f), metadata[f.stem]
    pred = read(HERE / 'predictions/PBE' / f.name)
    base = read(HERE / 'predictions/pretrained' / f.name)
    for role in ['PBE', 'pretrained']:
        rel = 'predictions/' + role + '/' + f.name
        assert sha(HERE / rel) == cache_manifest[rel]
    report['PBE_comparison_rows'].append({'structure_id': f.stem, 'family': meta['family'],
        'x_Al': meta['x_Al'], 'volume_per_atom_A3': meta['volume_per_atom_A3'],
        'fine_error_meV_atom': (pred['energy_eV'] - label['energy_eV']) * 1000 / meta['n_atoms'],
        'pretrained_error_meV_atom': (base['energy_eV'] - label['energy_eV']) * 1000 / meta['n_atoms']})

alignment = HERE / 'alignment/train-offsets-v1'
training = [r for r in read(alignment / 'training-labels.json') if r['functional'] == 'PBE']
z = np.load(alignment / 'training-geometries.npz')
train_xv = [(r['n_Al'] / r['n_atoms'], abs(float(np.linalg.det(z['cell_' + str(r['geometry_id'])]))) / r['n_atoms']) for r in training]
coverage = {}
for name, subset in [('all_PBE_train', train_xv), ('Al_fraction_at_most_0.1', [r for r in train_xv if r[0] <= .1]),
                      ('same_composition', [r for r in train_xv if abs(r[0] - row['x_Al']) < 1e-8])]:
    coverage[name] = {'n_frames': len(subset), 'volume_min_A3atom': min(v for x, v in subset),
                      'volume_max_A3atom': max(v for x, v in subset)}
report['training_coverage'] = coverage
report['training_input_hashes'] = {p.name: sha(p) for p in [alignment / 'training-labels.json', alignment / 'training-geometries.npz']}
report['model_inference_audit'] = read(DEST / 'inference/model-result.json')
assert report['model_inference_audit']['status'] == 'passed'
assert report['model_inference_audit']['cached_sha256'] == sha(HERE / 'predictions/PBE' / (SID + '.json'))
report['interpretation'] = {
    'established': 'The +17.926 meV/atom residual is reproduced by the unchanged original PBE checkpoint. Raw DFT parsing, SCF completion, geometry, functional, PAW hashes and prediction identity pass audit.',
    'supported_inference': 'A positive PBE fine-tuned energy bias occurs on several Ti-rich structures; Test009 also lies above the training volume range at the same composition. This supports composition/environment-dependent generalization error, not an isolated cache/export error.',
    'limits': 'No independent DFT convergence sweep was run. The exact contribution of training numerical settings is unresolved: the archived PBE template has ENCUT 520 whereas Test50 uses user-authorized 550; historical PBE OUTCAR is unavailable, and VASP/grid-generator versions differ. Correlation with composition/volume does not prove a causal mechanism. No Test50 offset was fitted.'}
(DEST / 'audit-summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
rows = report['PBE_comparison_rows']
for family, marker in [('liquid', 'o'), ('POSCAR_alpha-Ti-hcp_orthogonal', '^')]:
    rs = [r for r in rows if r['family'] == family]
    if not rs:
        continue
    axes[0].scatter([r['x_Al'] for r in rs], [r['fine_error_meV_atom'] for r in rs], marker=marker,
                    color='#2673c4', label='Fine-tuned, ' + ('liquid' if family == 'liquid' else 'hcp'), s=45)
axes[0].scatter([r['x_Al'] for r in rows], [r['pretrained_error_meV_atom'] for r in rows],
                facecolors='none', edgecolors='#ed8615', label='Pretrained, raw', s=45)
for sid in ['009', '015', '025', '005', '028']:
    r = next(r for r in rows if r['structure_id'].endswith(sid))
    axes[0].annotate(sid, (r['x_Al'], r['fine_error_meV_atom']), xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[0].axhline(0, color='0.4', lw=.8)
axes[0].set(xlabel='Al atomic fraction', ylabel='PBE model - DFT energy (meV/atom)', title='All available PBE labels (29 structures)')
axes[0].legend(fontsize=8)
axes[1].scatter([x for x, v in train_xv], [v for x, v in train_xv], s=10, color='0.65', alpha=.5, label='PBE train, 850 frames')
axes[1].scatter([row['x_Al']], [row['volume_per_atom_A3']], marker='*', s=200, color='#b22b2b', label='Test009: 11 Al + 205 Ti')
axes[1].annotate('009', (row['x_Al'], row['volume_per_atom_A3']), xytext=(7, -15), textcoords='offset points')
axes[1].set(xlabel='Al atomic fraction', ylabel='Volume per atom (angstrom^3)', title='Training composition and volume coverage')
axes[1].legend(fontsize=8, loc='lower right')
fig.suptitle('Test009 diagnostic: raw energies; no fitted shifts', fontsize=13)
for suffix in ['png', 'pdf']:
    fig.savefig(DEST / ('diagnostic.' + suffix), dpi=180)
plt.close(fig)
print(json.dumps({'checks': 'passed', 'PBE_residual_meV_atom': report['predictions']['PBE/PBE']['signed_energy_error_meV_atom'],
                  'training_coverage': coverage, 'output': str(DEST / 'audit-summary.json')}, indent=2))
