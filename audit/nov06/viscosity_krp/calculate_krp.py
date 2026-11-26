"""Reproduce the three KRP curves added to article Fig. 7a on 2026-11-06.

Uses the archived 1873 K enthalpies of the models in panel (a), including
the original PBE 5k parameter snapshot. Experimental enthalpy is the same
provisional 2000 K calorimetric digitization used in article Fig. 6.
No viscosity data are fitted and no new molecular dynamics is performed.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path.insert(0, str(WORK / 'audit/oct27/viscosity_5k_vs_15k/python_deps'))
import numpy as np

T_K = 1873.0
R_J_MOL_K = 8.31446261815324
ETA_TI = 0.61 * math.exp(27000.0 / (8.31 * T_K))
ETA_AL = 10.0 ** (-0.7324 + 803.49 / T_K)
OUT = WORK / 'deliverables/2026-11-06/Fig07_viscosity_KRP'
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def krp(x_al, h_kj_mol):
    return np.exp((1-x_al)*np.log(ETA_TI) + x_al*np.log(ETA_AL)
                  - 1000.0*h_kj_mol/(3.0*R_J_MOL_K*T_K))


thermo_path = WORK / 'audit/oct20/thermo_reprocessed.json'
refs_path = WORK / 'audit/oct20/reference_values.json'
profile_path = WORK / 'audit/nov01/viscosity_article_combined/figure_profile.json'
identity_path = WORK / 'manuscript/supporting_data/checkpoint_identity.json'
thermo, refs, profile = read(thermo_path), read(refs_path), read(profile_path)
assert profile['selection']['composition_PBE_checkpoint'] == 5000
assert all(r['T'] == T_K for rows in profile['composition'].values() for r in rows)

sources = {}
input_rows = []
for key in ['experiment', 'pbe', 'scan']:
    if key == 'experiment':
        points = sorted((float(x), h) for x, h in refs['mixing_digitized_kJmol'].items())
        points = [(0.0, 0.0)] + points + [(1.0, 0.0)]
        x, h = np.array(points, dtype=float).T
        se = [None] * len(x)
        temperature = 2000
        note = 'Calorimetry reproduced in Witusiewicz2008 Fig. 7; provisional digitization. Assumed unchanged from 2000 to 1873 K.'
        trajectory_sources = []
    else:
        rows = sorted([r for r in thermo if r['T'] == T_K and r['dft'] == key], key=lambda r: r['x_Al'])
        assert len(rows) == 11
        x = np.array([r['x_Al'] for r in rows])
        h = np.array([r['dH_external_pressure'] for r in rows])
        se = [r['dH_external_sem_5ps'] for r in rows]
        temperature = T_K
        note = 'Archived 1873 K MD; total energy plus external-pressure pV, pure-liquid references at the same T and model.'
        trajectory_sources = [r['source'] for r in rows]
        # Independently check the mixing-enthalpy subtraction and endmember ordering.
        h_absolute = np.array([r['H_external_pressure'] for r in rows])
        assert np.allclose(h, h_absolute - (1-x)*h_absolute[0] - x*h_absolute[-1], atol=1e-10)
    assert x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) > 0)
    assert h[0] == 0 and h[-1] == 0
    sources[key] = dict(x_Al=x.tolist(), deltaH_kJ_mol=h.tolist(),
                        deltaH_SEM_kJ_mol=se, enthalpy_temperature_K=temperature,
                        note=note, trajectory_sources=trajectory_sources)
    for xi, hi, sei in zip(x, h, se):
        eta = float(krp(xi, hi))
        input_rows.append(dict(source=key, x_Al=xi, enthalpy_temperature_K=temperature,
                               deltaH_kJ_mol=hi, deltaH_SEM_kJ_mol=sei,
                               viscosity_temperature_K=T_K, eta_KRP_mPa_s=eta,
                               eta_SEM_from_enthalpy_only_mPa_s=None if sei is None else eta*1000*sei/(3*R_J_MOL_K*T_K)))

grid = np.linspace(0, 1, 1001)
curves = {}
curve_rows = []
summaries = {}
for key, data in sources.items():
    h = np.interp(grid, data['x_Al'], data['deltaH_kJ_mol'])
    eta = krp(grid, h)
    assert np.all(np.isfinite(eta)) and np.all(eta > 0)
    assert np.allclose(eta[[0,-1]], [ETA_TI, ETA_AL], rtol=1e-13)
    # Independent multiplicative form catches sign/unit mistakes.
    assert np.allclose(eta, ETA_TI**(1-grid)*ETA_AL**grid*np.exp(-h/(3*(R_J_MOL_K/1000)*T_K)), rtol=1e-13)
    curves[key] = dict(x_Al=grid.tolist(), deltaH_kJ_mol=h.tolist(), eta_mPa_s=eta.tolist())
    peak = int(np.argmax(eta))
    summaries[key] = dict(maximum_eta_mPa_s=float(eta[peak]), x_Al_at_maximum=float(grid[peak]),
                          eta_at_xAl_045_mPa_s=float(krp(.45, np.interp(.45, data['x_Al'], data['deltaH_kJ_mol']))))
    for xi, hi, etai in zip(grid, h, eta):
        curve_rows.append(dict(source=key, x_Al=xi, deltaH_kJ_mol=hi, eta_KRP_mPa_s=etai))

comparisons = []
for c_ti, (a, b, lo, hi) in refs['takeda_table3'].items():
    x = 1-float(c_ti)
    reference = 10**(a+b/T_K)
    row = dict(x_Al=x, Takeda_mPa_s=reference)
    for key, data in sources.items():
        row[key+'_KRP_mPa_s'] = float(krp(x, np.interp(x, data['x_Al'], data['deltaH_kJ_mol'])))
    comparisons.append(row)

# Estimate the impact of substituting 2000 K MD enthalpy, holding viscosity T fixed.
temperature_sensitivity = {}
for key in ['pbe', 'scan']:
    old2000 = sorted([r for r in thermo if r['T'] == 2000 and r['dft'] == key], key=lambda r:r['x_Al'])
    h2000 = np.array([r['dH_external_pressure'] for r in old2000])
    h1873 = np.array(sources[key]['deltaH_kJ_mol'])
    relative_change = np.expm1(-(h2000-h1873)*1000/(3*R_J_MOL_K*T_K))
    temperature_sensitivity[key] = dict(max_abs_eta_change_percent=float(100*np.max(np.abs(relative_change))),
                                       note='Same original model, 2000 K versus 1873 K archived MD; not an uncertainty bound for experiment.')

manifest = dict(
    date='2026-11-06', viscosity_temperature_K=T_K,
    equation='ln(eta/eta_ref)=(1-x_Al)*ln(eta_Ti/eta_ref)+x_Al*ln(eta_Al/eta_ref)-deltaH_mix/(3*R*T)',
    eta_ref='1 mPa s', enthalpy_units='kJ per mol of atoms; multiplied by 1000 in the equation',
    R_J_mol_K=R_J_MOL_K,
    pure_ti=dict(eta_mPa_s=ETA_TI, formula_mPa_s='0.61*exp(27000/(8.31*T))',
                 source='Ishikawa2012', doi='10.1088/0957-0233/23/2/025305',
                 note='Same provisional transcription as existing Fig. 7b; 1873 K is below the equilibrium melting point of Ti.'),
    pure_al=dict(eta_mPa_s=ETA_AL, formula_mPa_s='10**(-0.7324+803.49/T)',
                source='Assael2006, Eq. (19)', doi='10.1063/1.2149380',
                fit_temperature_range_K=[933,1270], extrapolated=True,
                reported_uncertainty_percent_in_validated_range=14,
                note='Critically evaluated experimental reference correlation, not an arithmetic average. Its 14% uncertainty is not validated at 1873 K.',
                equation_crosscheck_url='https://link.springer.com/article/10.1007/s11661-017-4053-6',
                original_paper_mirror='https://www.scribd.com/document/655453439/285-1-online-viscosity'),
    interpolation='Piecewise-linear interpolation of mixing enthalpy, then evaluate KRP; endpoints constrained to zero; no fit to viscosity.',
    sources=sources, curves=curves, summaries=summaries,
    md_temperature_sensitivity=temperature_sensitivity,
    model_identity=read(identity_path)['models'],
    composition_PBE_checkpoint=profile['selection']['composition_PBE_checkpoint'],
    limitation_notes=[refs['mixing_digitization_note'],
        'Only experimental enthalpy is transferred from 2000 to 1873 K; both MD enthalpies are evaluated at 1873 K.',
        'PBE input is the 5k model used in Fig. 7a, not the PBE 15k / 2000 K data used in Fig. 6.',
        'Conditional enthalpy SEM is exported at nodes, but no total confidence band is claimed for KRP.',
        'Pure-component viscosity inputs are identical for all three KRP curves.'],
    literature=[dict(key='Kozlov1983', citation='L. Ya. Kozlov, L. M. Romanov, N. N. Petrov, Prediction of multicomponent metallic melt viscosity, Izv. Vuz. Chern. Metallurg. 3, 7-11 (1983).',
                     equation_and_bibliography_crosscheck='https://link.springer.com/article/10.1007/s11665-016-2297-8'),
                dict(key='Assael2006', doi='10.1063/1.2149380'),
                dict(key='Witusiewicz2008', doi='10.1016/j.jallcom.2007.10.061')],
    sources_sha256={str(path.relative_to(WORK)):sha(path) for path in [thermo_path,refs_path,identity_path,Path(__file__)]},
)
for path in [HERE/'krp_data.json', WORK/'manuscript/supporting_data/viscosity_krp_1873K.json', OUT/'krp_data.json']:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
write_csv(OUT/'enthalpy_inputs_and_KRP_nodes.csv', input_rows)
write_csv(OUT/'KRP_curves_1873K.csv', curve_rows)
write_csv(OUT/'KRP_vs_Takeda.csv', comparisons)

profile['krp'] = dict(data_path='manuscript/supporting_data/viscosity_krp_1873K.json',
                      data_sha256=sha(WORK/'manuscript/supporting_data/viscosity_krp_1873K.json'))
profile['version'] = '2026-11-06-krp-1873K'
profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(eta_Ti_mPa_s=ETA_TI, eta_Al_mPa_s=ETA_AL,
                      summaries=summaries, temperature_sensitivity=temperature_sensitivity), indent=2))
