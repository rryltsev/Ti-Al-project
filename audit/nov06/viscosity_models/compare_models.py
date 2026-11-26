"""Compare published enthalpy-based viscosity models for liquid Ti-Al at 1873 K.

No parameters are fitted to alloy viscosities. This exploratory calculation
reuses the inputs of the KRP figure and leaves the manuscript files unchanged.
Run from the workspace: python audit/nov06/viscosity_models/compare_models.py
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path.insert(0, str(WORK / 'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = WORK / 'deliverables/2026-11-06/Fig07_viscosity_models'
OUT.mkdir(parents=True, exist_ok=True)
KRP_PATH = WORK / 'manuscript/supporting_data/viscosity_krp_1873K.json'
PROFILE_PATH = WORK / 'audit/nov01/viscosity_article_combined/figure_profile.json'
FIGURE_MANIFEST_PATH = PROFILE_PATH.with_name('figure_manifest.json')
BRILLO_TEXT = WORK / 'audit/pdf_text/2025 Thermophysical Properties of Al–Ti Binary Liquid Alloys.txt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


krp_data = read(KRP_PATH)
profile = read(PROFILE_PATH)
figure_manifest = read(FIGURE_MANIFEST_PATH)
assert profile['selection']['composition_PBE_checkpoint'] == 5000
assert profile['krp']['data_sha256'] == sha(KRP_PATH)
assert figure_manifest['temperature'] == profile['temperature']
assert figure_manifest['composition'] == profile['composition']
protected = [KRP_PATH, PROFILE_PATH, FIGURE_MANIFEST_PATH,
             WORK / 'manuscript/figures/viscosity.pdf',
             WORK / 'manuscript/sections/methods.tex',
             WORK / 'manuscript/sections/results.tex']
protected_hashes = {str(p.relative_to(WORK)): sha(p) for p in protected}
T = float(krp_data['viscosity_temperature_K'])
R = float(krp_data['R_J_mol_K'])
ETA_TI = float(krp_data['pure_ti']['eta_mPa_s'])
ETA_AL = float(krp_data['pure_al']['eta_mPa_s'])
V_TI, V_AL = 11.45, 12.94555  # cm3/mol, Brillo 2025 Table 7, 1873 K.
V0, V1 = -2.66, -2.831  # Table 8: specifically the isothermal 1873 K column.
ALPHA = 0.155
SOURCES = ['experiment', 'pbe', 'scan']
MODELS = ['KRP', 'Moelwyn-Hughes', 'Schick', 'Kaptay']
GRID = np.linspace(0, 1, 1001)


def h_at(source, x):
    data = krp_data['sources'][source]
    return np.interp(x, data['x_Al'], data['deltaH_kJ_mol']) * 1000.0


def ideal_entropy_term(x):
    # With 0 ln(0) defined as zero, the pure-component limits remain exact.
    x = np.asarray(x, dtype=float)
    return x * np.log(np.maximum(x, np.finfo(float).tiny)) + (1-x) * np.log(np.maximum(1-x, np.finfo(float).tiny))


def molar_volume(x, excess=True):
    x = np.asarray(x, dtype=float)
    v = (1-x) * V_TI + x * V_AL
    if excess:
        # Brillo Eq. (7), A=Al, B=Ti, so x_A-x_B = 2*x_Al-1.
        v = v + x * (1-x) * (V0 + V1 * (2*x-1))
    return v


def eta(model, x, h, alpha=ALPHA, excess=True):
    """x: mole fraction Al; h: integral mixing enthalpy J/mol atoms; eta: mPa s."""
    x, h = np.asarray(x, dtype=float), np.asarray(h, dtype=float)
    log_g = (1-x) * math.log(ETA_TI) + x * math.log(ETA_AL)
    if model == 'KRP':
        return np.exp(log_g - h / (3*R*T))
    if model == 'Moelwyn-Hughes':
        # The integral-enthalpy convention in Schick et al. (2012), Eq. (1).
        return ((1-x)*ETA_TI + x*ETA_AL) * (1 - 2*h/(R*T))
    if model == 'Schick':
        # Eqs. (8), (11), (12), retaining the configurational entropy factor.
        return np.exp(log_g + ideal_entropy_term(x) - h/(R*T))
    if model == 'Kaptay':
        # Equivalent to the original Eyring expression, with volumes in any
        # common unit; the geometric-volume/arithmetic-volume ratio is unitless.
        volume_ratio = V_TI**(1-x) * V_AL**x / molar_volume(x, excess)
        return np.exp(log_g - alpha*h/(R*T)) * volume_ratio
    raise ValueError(model)


experiment = []
for c_ti, (a, b, lo, hi) in profile['references']['takeda_table3'].items():
    experiment.append(dict(x_Al=1-float(c_ti), eta_Takeda_mPa_s=10**(a+b/T),
                           fit_min_T_K=lo, fit_max_T_K=hi,
                           extrapolation_K=max(lo-T, T-hi, 0)))
experiment.sort(key=lambda row: row['x_Al'])
EX = np.array([r['x_Al'] for r in experiment])
EY = np.array([r['eta_Takeda_mPa_s'] for r in experiment])
curves, curve_rows, metrics, point_rows = {}, [], [], []
checks = []
for model in MODELS:
    curves[model] = {}
    for source in SOURCES:
        h = h_at(source, GRID)
        y = eta(model, GRID, h)
        assert np.all(np.isfinite(y)) and np.all(y > 0)
        assert np.allclose(y[[0, -1]], [ETA_TI, ETA_AL], rtol=1e-13)
        if model == 'KRP':
            assert np.allclose(y, krp_data['curves'][source]['eta_mPa_s'], rtol=1e-13)
        curves[model][source] = y
        peak = int(np.argmax(y))
        yp = eta(model, EX, h_at(source, EX))
        error = yp - EY
        row = dict(model=model, enthalpy_source=source, n=5,
                   MAPE_percent=float(100*np.mean(np.abs(error/EY))),
                   RMSE_mPa_s=float(np.sqrt(np.mean(error**2))),
                   mean_signed_relative_error_percent=float(100*np.mean(error/EY)),
                   maximum_eta_mPa_s=float(y[peak]), x_Al_at_maximum=float(GRID[peak]),
                   eta_xAl_045_mPa_s=float(eta(model, .45, h_at(source, .45))))
        metrics.append(row)
        for xi, hi, yi in zip(GRID, h, y):
            curve_rows.append(dict(model=model, enthalpy_source=source, x_Al=xi,
                                   deltaH_J_mol=hi, eta_mPa_s=yi))
        for exp, yi, ei in zip(experiment, yp, error):
            point_rows.append(dict(model=model, enthalpy_source=source, **exp,
                                   eta_model_mPa_s=float(yi), residual_mPa_s=float(ei),
                                   relative_error_percent=float(100*ei/exp['eta_Takeda_mPa_s'])))
checks.append('All 12 curves positive and finite; pure limits equal the prescribed Ti/Al viscosities.')
checks.append('All three KRP curves agree with the previous published calculation to rtol=1e-13.')

# Verify Schick with its unreduced Arrhenius form, independently of eta().
pre_ti, pre_al = .61, 10**(-.7324)
ea_ti, ea_al = R*27000/8.31, R*math.log(10)*803.49
for source in SOURCES:
    pre = np.exp((1-GRID)*math.log(pre_ti) + GRID*math.log(pre_al) + ideal_entropy_term(GRID))
    ea = (1-GRID)*ea_ti + GRID*ea_al - h_at(source, GRID)
    assert np.allclose(pre*np.exp(ea/(R*T)), curves['Schick'][source], rtol=1e-13)
checks.append('Schick: independent Arrhenius evaluation of Eqs. (11)-(12) agrees to rtol=1e-13.')

# Verify Kaptay through Eyring activation free energies, now in SI units.
PLANCK, AVOGADRO = 6.62607015e-34, 6.02214076e23
DG_TI = R*T*math.log((ETA_TI*1e-3)*(V_TI*1e-6)/(PLANCK*AVOGADRO))
DG_AL = R*T*math.log((ETA_AL*1e-3)*(V_AL*1e-6)/(PLANCK*AVOGADRO))
for source in SOURCES:
    eyring_pa_s = PLANCK*AVOGADRO/(molar_volume(GRID)*1e-6) * np.exp(((1-GRID)*DG_TI+GRID*DG_AL-ALPHA*h_at(source, GRID))/(R*T))
    assert np.allclose(eyring_pa_s*1000, curves['Kaptay'][source], rtol=1e-13)
checks.append('Kaptay: original Eyring expression in Pa s and m3/mol agrees to rtol=1e-13.')
assert np.allclose(eta('Moelwyn-Hughes', GRID, 0), (1-GRID)*ETA_TI+GRID*ETA_AL)
assert np.all(molar_volume(GRID) > 0)
checks.append('Moelwyn-Hughes: zero-enthalpy limit is the arithmetic mixture.')

md_metrics = []
for model in MODELS:
    for source in ['pbe', 'scan']:
        rows = profile['composition'][source]
        xx = np.array([1-r['c_Ti'] for r in rows])
        yy = np.array([r['eta_mPa_s'] for r in rows])
        error = eta(model, xx, h_at(source, xx)) - yy
        md_metrics.append(dict(model=model, MD_model=source, enthalpy_source=source,
                               n=len(rows), MAPE_percent=float(100*np.mean(np.abs(error/yy))),
                               RMSE_mPa_s=float(np.sqrt(np.mean(error**2)))))

sensitivity = {}
for source in SOURCES:
    standard = curves['Kaptay'][source]
    ideal = eta('Kaptay', GRID, h_at(source, GRID), excess=False)
    sensitivity[source] = dict(
        alpha_014_max_relative_change_percent=float(100*np.max(np.abs(eta('Kaptay', GRID, h_at(source, GRID), alpha=.14)/standard-1))),
        alpha_017_max_relative_change_percent=float(100*np.max(np.abs(eta('Kaptay', GRID, h_at(source, GRID), alpha=.17)/standard-1))),
        neglecting_excess_volume_max_relative_change_percent=float(100*np.max(np.abs(ideal/standard-1))),
    )

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                     'axes.labelsize': 10, 'axes.titlesize': 10,
                     'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
COLORS = {'pbe': '#2878B5', 'scan': '#D55E00', 'experiment': '#20836B'}
LABELS = {'pbe': 'PBE', 'scan': r'r$^2$SCAN', 'experiment': 'experiment'}
LINESTYLES = {'experiment': '-.', 'pbe': '--', 'scan': ':'}
MODEL_LABELS = {'KRP': 'KRP (reference)', 'Moelwyn-Hughes': 'Moelwyn-Hughes',
                'Schick': 'Schick et al.', 'Kaptay': r'Kaptay ($\alpha=0.155$)'}


def setup_axes(ax):
    ax.grid(axis='y', alpha=.15, lw=.6)
    ax.spines[['top', 'right']].set_visible(False)


def draw_composition(ax, model):
    for source in ['pbe', 'scan']:
        rows = sorted(profile['composition'][source], key=lambda row: 1-row['c_Ti'])
        ax.errorbar([1-r['c_Ti'] for r in rows], [r['eta_mPa_s'] for r in rows],
                    yerr=[r['eta_std_mPa_s'] for r in rows], fmt='o-',
                    color=COLORS[source], lw=1.1, ms=3.8, capsize=2.5,
                    label='MD: '+LABELS[source], zorder=3)
    ax.scatter(EX, EY, s=25, marker='s', c='black', zorder=6, label='Takeda fit')
    for source in SOURCES:
        ax.plot(GRID, curves[model][source], color=COLORS[source],
                ls=LINESTYLES[source], lw=1.7, zorder=4,
                label=r'Model: $\Delta H_{\rm '+('r^2SCAN' if source == 'scan' else 'PBE' if source == 'pbe' else 'exp')+'}$')
    ax.set(xlabel=r'$x_{\mathrm{Al}}$', ylabel=r'$\eta$ (mPa s)', xlim=(-.025, 1.025))
    ax.set_xticks(np.arange(0, 1.01, .2))
    setup_axes(ax)


def save_figure(fig, basename):
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'{basename}.{ext}', dpi=300)


fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True, sharey=True)
fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.075, hspace=.30, wspace=.16)
for letter, ax, model in zip('abcd', axes.flat, MODELS):
    draw_composition(ax, model)
    ax.set_title(f'({letter}) '+MODEL_LABELS[model], loc='left', pad=9)
    ax.set_ylim(0, 12)
    ax.set_yticks(np.arange(0, 12.1, 2))
    ax.tick_params(labelbottom=True, labelleft=True)
    score = next(r['MAPE_percent'] for r in metrics if r['model']==model and r['enthalpy_source']=='experiment')
    ax.text(.97, .96, rf'$\Delta H_{{\rm exp}}$: MAPE {score:.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=8.3)
fig.suptitle('Ti-Al at 1873 K: published viscosity models', y=.985, fontsize=12)
handles, labels = axes.flat[0].get_legend_handles_labels()
lookup = dict(zip(labels, handles))
order = ['MD: PBE', r'MD: r$^2$SCAN', 'Takeda fit',
         r'Model: $\Delta H_{\rm exp}$', r'Model: $\Delta H_{\rm PBE}$', r'Model: $\Delta H_{\rm r^2SCAN}$']
legend_order = [order[i] for i in [0, 3, 1, 4, 2, 5]]
fig.legend([lookup[s] for s in legend_order], legend_order, loc='upper center', bbox_to_anchor=(.53, .95),
           ncol=3, frameon=False, columnspacing=2.5, handlelength=3)
save_figure(fig, 'viscosity_models_comparison')
plt.close(fig)

# A directly reviewable alternative to Fig. 7: Schick panel (a), original panel (b).
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65), layout='constrained')
draw_composition(axes[0], 'Schick')
axes[0].set(title='(a) Ti-Al, 1873 K: Schick model', ylim=(0, 8.2))
handles, labels = axes[0].get_legend_handles_labels()
lookup = dict(zip(labels, handles))
axes[0].legend([lookup[s] for s in order], order, loc='lower left', fontsize=7.2, frameon=False)
for source in ['pbe', 'scan']:
    rows = sorted(profile['temperature'][source], key=lambda r: r['T'])
    pars = figure_manifest['arrhenius_fits'][source]
    tt = np.linspace(rows[0]['T'], rows[-1]['T'], 400)
    axes[1].plot(tt, np.exp(pars['a']+pars['B_K']/tt), color=COLORS[source], lw=1.35)
    axes[1].errorbar([r['T'] for r in rows], [r['eta_mPa_s'] for r in rows],
                     yerr=[r['eta_std_mPa_s'] for r in rows], fmt='o', color=COLORS[source],
                     ms=4.5, capsize=2.7, lw=1.2, zorder=4)
tt = np.linspace(1850, 2120, 200)
ti = .61*np.exp(27000/(8.31*tt))
axes[1].plot(tt, ti, 'k--', lw=1)
axes[1].fill_between(tt, .897*ti, 1.103*ti, color='.65', alpha=.20, zorder=0)
axes[1].set(title='(b) Pure Ti', xlabel='Temperature (K)', ylabel=r'$\eta$ (mPa s)',
            xlim=(1840, 2130), ylim=(2.2, 4.35))
axes[1].set_xticks([1873, 1941, 2000, 2100])
setup_axes(axes[1])
axes[1].legend([Line2D([], [], color=COLORS[s], marker='o', lw=1.35, ms=4.5) for s in ['pbe', 'scan']]
               +[Line2D([], [], color='black', ls='--', lw=1)],
               ['PBE', r'r$^2$SCAN', 'Ishikawa fit*'], loc='upper right', frameon=False)
save_figure(fig, 'figure7_schick_candidate')
plt.close(fig)

write_csv('model_curves_1873K.csv', curve_rows)
write_csv('model_metrics_vs_Takeda.csv', metrics)
write_csv('comparison_at_Takeda_compositions.csv', point_rows)
write_csv('model_metrics_vs_matching_MD.csv', md_metrics)
write_csv('molar_volume_1873K.csv', [dict(x_Al=float(x), Vm_cm3_mol=float(molar_volume(x)),
                                          excess_V_cm3_mol=float(molar_volume(x)-molar_volume(x, False))) for x in GRID])

manifest = dict(
    date='2026-11-06', temperature_K=T, R_J_mol_K=R,
    pure_ti=krp_data['pure_ti'], pure_al=krp_data['pure_al'],
    enthalpy_sources=krp_data['sources'], model_identity=krp_data['model_identity'],
    composition_PBE_checkpoint=5000, no_parameters_fitted_to_alloy_viscosity=True,
    convention='x=x_Al, H=integral mixing enthalpy in J/mol atoms, eta in mPa s; 0 ln(0)=0',
    equations=dict(
        KRP='eta=eta_Ti**(1-x)*eta_Al**x*exp(-H/(3*R*T))',
        Moelwyn_Hughes='eta=((1-x)*eta_Ti+x*eta_Al)*(1-2*H/(R*T))',
        Schick='eta=eta_Ti**(1-x)*eta_Al**x*exp((1-x)*ln(1-x)+x*ln(x)-H/(R*T))',
        Kaptay='eta=eta_Ti**(1-x)*eta_Al**x*(V_Ti**(1-x)*V_Al**x/Vm)*exp(-0.155*H/(R*T))'),
    volume=dict(source='Brillo et al. 2025, Tables 7 and 8, Eq. (7)', doi='10.1007/s10765-025-03651-0',
                V_Ti_cm3_mol=V_TI, V_Al_cm3_mol=V_AL, v0_cm3_mol=V0, v1_cm3_mol=V1,
                equation='Vm=(1-x)*V_Ti+x*V_Al+x*(1-x)*(v0+v1*(2*x-1))',
                note='Common experimental volume curve for all enthalpy sources. Use the explicit 1873 K column of Table 8; the separate temperature-fit column does not reproduce those coefficients exactly.'),
    literature=[dict(key='Schick2012', title='Viscosity of Al-Cu liquid alloys: measurement and thermodynamic description',
                     doi='10.1007/s10853-012-6710-x', url='https://elib.dlr.de/77683/1/AlCuViscosity.pdf',
                     equations='MH Eq. (1), KRP Eq. (2), Kaptay Eqs. (5)-(6), Schick Eqs. (8),(11)-(12)'),
                dict(key='Kaptay2003', title='A new equation to estimate the concentration dependence of the viscosity of liquid metallic alloys from the heat of mixing data',
                     publication='microCAD 2003, Metallurgy, University of Miskolc, pp. 23-28',
                     author_summary='https://calphad.org/meetings/2004/a/12/12-5.htm'),
                dict(key='Dobosz2021', doi='10.1016/j.molliq.2021.117646',
                     url='https://iris.cnr.it/retrieve/ff429fe0-662b-47ab-963c-0d4c8bb2a6ac/21Dobosz%20Ga-Sn-Zn.pdf',
                     note='Independent confirmation of MH factor 2 in Eq. (6) and Schick Eqs. (8)-(9).'),
                dict(key='Mohr2020', doi='10.1002/adem.202000169',
                     note='Ti-based alloy application. Its Eq. (17) uses a different omega-parameter convention; not used here and not equated to integral H.')],
    metrics_definition='Unweighted descriptive residuals against five Takeda Table 3 fits at 1873 K; MAPE=100*mean(abs((prediction-reference)/reference)). No statistical significance or uncertainty interval claimed.',
    comparison_points=experiment, metrics=metrics, matching_MD_metrics=md_metrics,
    sensitivity=sensitivity, checks=checks,
    limitations=krp_data['limitation_notes']+[
        'Al reference correlation is extrapolated beyond its 933-1270 K validated range.',
        'Experimental viscosity comparison covers only x_Al=0.45-0.90; maxima on the Ti-rich side cannot be located from those data.',
        'Takeda fits at x_Al=0.45 and 0.60 are extrapolated upward by 3 K.',
        'Model spread is not a statistical confidence interval. Alpha sensitivity is not a total Kaptay error estimate.',
        'MH is specifically the integral-H form in Schick2012 Eq. (1); other conventions in Ti-Al literature must not be silently interchanged.',
        'Alloy liquid-state expressions may represent undercooled liquids in parts of the composition range at 1873 K.'],
    interpolation='Piecewise linear interpolation of H, then evaluate the nonlinear viscosity equation.',
    manuscript_status='Unchanged; figure7_schick_candidate is a separate reviewable alternative.',
    panel_b=dict(source_manifest=str(FIGURE_MANIFEST_PATH.relative_to(WORK)),
                 arrhenius_fits=figure_manifest['arrhenius_fits'], temperature=profile['temperature']),
    sources_sha256={str(p.relative_to(WORK)): sha(p) for p in [KRP_PATH, PROFILE_PATH, FIGURE_MANIFEST_PATH, BRILLO_TEXT, Path(__file__)]},
)
assert protected_hashes == {str(p.relative_to(WORK)): sha(p) for p in protected}
checks.append('Canonical manuscript figure, methods, results, profile and KRP inputs unchanged (SHA-256).')
manifest['protected_files_sha256'] = protected_hashes
manifest['output_sha256'] = {p.name: sha(p) for p in OUT.iterdir() if p.suffix in ['.csv', '.pdf', '.png', '.svg']}
(OUT / 'model_comparison_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(HERE / 'verification.json').write_text(json.dumps(dict(checks=checks, all_passed=True), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(pure_Ti_mPa_s=ETA_TI, pure_Al_mPa_s=ETA_AL, metrics=metrics,
                     matching_MD_metrics=md_metrics, sensitivity=sensitivity, checks=checks), indent=2))
