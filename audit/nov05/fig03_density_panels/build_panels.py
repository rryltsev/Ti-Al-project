"""Export the revised Fig. 3 and standalone panels without changing the article.

Run with the project's Python 3.11 environment. Data are the CSV export of
the current article figure, density_ti_6e45e69b79dc.pdf.
Use --combined-only to add the side-by-side figure without rewriting panels.
"""

from pathlib import Path
import csv
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'deliverables/2026-11-03/figure_data/Fig03_density_T'
OUT = ROOT / 'deliverables/2026-11-05/Fig03_density_T'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'matplotlib-cache'))
sys.path.insert(0, str(ROOT / 'audit/oct27/viscosity_5k_vs_15k/python_deps'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name):
    with (SOURCE / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


# Guard all manuscript sources and PDFs, including the compiled article.
protected = sorted(
    p for p in (ROOT / 'manuscript').rglob('*')
    if p.is_file() and p.suffix.lower() in {'.tex', '.bib', '.pdf', '.png'}
)
protected_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}

models_all = read_csv('models.csv')
models = [
    r for r in models_all
    if not (r['functional'] == 'pbe' and r['series'] == 'archived')
]
assert len(models_all) == 67 and len(models) == 51
assert all(r['series'] == 'updated' for r in models if r['functional'] == 'pbe')

solid_ref = read_csv('solid_reference.csv')
liquid_ref = read_csv('liquid_correlation.csv')
experiment = read_csv('liquid_experiment.csv')
assert len(solid_ref) == 15 and len(experiment) == 25

# Verify that the packaged CSV contains precisely the current plotted means
# and SEMs, including the extended PBEsol/r2SCAN liquid series.
registry = json.loads((ROOT / 'manuscript/supporting_data/pbe15k_density_temperature.json').read_text(encoding='utf-8'))
pbe_points = [p for p in registry['points'] if p['kind'] == 'homogeneous_NPT' and p['composition'] == 'Ti']
assert len(pbe_points) == 17
for point in pbe_points:
    matches = [r for r in models if r['functional'] == 'pbe'
               and r['starting_phase'] == point['phase_initial']
               and float(r['temperature_K']) == point['T_K']]
    assert len(matches) == 1
    assert float(matches[0]['density_g_cm3']) == point['rho']['mean']
    assert float(matches[0]['SEM_g_cm3']) == point['rho']['sem']
extended = json.loads((ROOT / 'manuscript/supporting_data/xc_liquid_density_extended.json').read_text(encoding='utf-8'))
for point in extended['points']:
    matches = [r for r in models if r['functional'] == point['key']
               and r['starting_phase'] == 'liquid'
               and float(r['temperature_K']) == point['T_K']]
    assert len(matches) == 1
    assert float(matches[0]['density_g_cm3']) == point['rho']['mean']
    assert float(matches[0]['SEM_g_cm3']) == point['rho']['sem']

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
styles = {
    'pbe': ('#2878B5', 'PBE'),
    'sol': ('#39835B', 'PBEsol'),
    'scan': ('#D55E00', r'r$^2$SCAN'),
}
phases = [('alpha', 'o'), ('beta', '^'), ('liquid', 's')]


def rows_for(functional, phase):
    return sorted(
        (r for r in models if r['functional'] == functional and r['starting_phase'] == phase),
        key=lambda r: float(r['temperature_K']),
    )


def coordinates(rows):
    return ([float(r['temperature_K']) for r in rows],
            [float(r['density_g_cm3']) for r in rows])


def main_curves(axes):
    # Retain the original drawing order, colors, markers, and line widths.
    for functional in ['sol', 'scan']:
        color, label = styles[functional]
        for phase, marker in phases:
            rows = rows_for(functional, phase)
            ax = axes[1] if phase == 'liquid' else axes[0]
            x, y = coordinates(rows)
            if phase == 'liquid':
                ax.errorbar(x, y, yerr=[float(r['SEM_g_cm3']) for r in rows],
                            color=color, marker=marker, ms=3.4, lw=1,
                            capsize=2, label=label)
            else:
                ax.plot(x, y, c=color, marker=marker, ms=3.4, lw=1,
                        label=label if phase == 'alpha' else None)
    for phase, marker in phases:
        rows = rows_for('pbe', phase)
        ax = axes[1] if phase == 'liquid' else axes[0]
        x, y = coordinates(rows)
        for i in range(len(rows) - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=styles['pbe'][0], lw=1.3)
        ax.errorbar(x, y, yerr=[float(r['SEM_g_cm3']) for r in rows],
                    color=styles['pbe'][0], fmt=marker, ms=4.4,
                    capsize=2, lw=.9, zorder=5,
                    label='PBE' if phase in ['alpha', 'liquid'] else None)


# Use the original two-panel layout so that standalone exports preserve
# the dimensions and typography of the current article panels.
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), layout='constrained')
main_curves(axes)
for phase in ['alpha', 'beta']:
    rows = [r for r in solid_ref if r['phase'] == phase]
    x, y = coordinates(rows)
    axes[0].plot(x, y, c='black', marker='D', ms=3, ls=':', lw=.9,
                 label="Zinov'ev (1989)" if phase == 'alpha' else None)

tt, rr = coordinates(liquid_ref)
axes[1].fill_between(tt, [float(r['lower_g_cm3']) for r in liquid_ref],
                     [float(r['upper_g_cm3']) for r in liquid_ref],
                     color='.88', zorder=0, label=r'$\pm2\%$ (95% coverage)')
axes[1].plot(tt, rr, c='black', lw=1.1, label='Ntonti et al. (2024)')
xt, yt = coordinates(experiment)
axes[1].errorbar(
    xt, yt,
    yerr=[[float(r['density_g_cm3']) - float(r['lower_g_cm3']) for r in experiment],
          [float(r['upper_g_cm3']) - float(r['density_g_cm3']) for r in experiment]],
    fmt='o', ms=2.6, mfc='white', mec='black', mew=.6,
    ecolor='.35', elinewidth=.55, capsize=1.2, zorder=4,
    label='Reiplinger & Brillo (2022)',
)
for temperature, label in [(1155, r'$\alpha$-$\beta$: 1155 K'), (1943, r'$T_m$: 1943 K')]:
    axes[0].axvline(temperature, color='.45', ls='-.', lw=.7, zorder=0)
    axes[0].text(temperature - 25, 4.82, label, rotation=90,
                 ha='right', va='top', fontsize=7, color='.3')

axes[0].set(title='(a) Ti crystal', xlim=(220, 2010), ylim=(4.20, 4.85))
axes[1].set(title='(b) Ti liquid', xlim=(1835, 2530), ylim=(3.87, 4.85))
axes[0].legend(frameon=True, facecolor='white', edgecolor='none',
               framealpha=1, ncol=2, fontsize=7, loc='lower left')
axes[1].legend(frameon=False, fontsize=6.7, loc='upper right',
               labelspacing=.28, handlelength=1.5)
for ax in axes:
    ax.set(xlabel='Temperature (K)', ylabel=r'Density (g cm$^{-3}$)')
    labels = ax.get_legend_handles_labels()[1]
    assert 'PBE' in labels and not any('5k' in s or '15k' in s for s in labels)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
# Fix layout before cropping so savefig does not rearrange either panel.
fig.set_layout_engine('none')
outputs = []
for extension in ['pdf', 'png']:
    output = OUT / f'Fig3_Ti_density_combined.{extension}'
    fig.savefig(output, bbox_inches='tight', dpi=600, facecolor='white')
    outputs.append(output)
if '--combined-only' not in sys.argv:
    for ax, basename in zip(axes, ['Fig3a_Ti_crystal', 'Fig3b_Ti_liquid']):
        # Hide the neighboring panel so its outer labels cannot enter the crop.
        for other in axes:
            other.set_visible(other is ax)
        bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(.10)
        for extension in ['pdf', 'png']:
            output = OUT / f'{basename}.{extension}'
            fig.savefig(output, bbox_inches=bounds, dpi=600, facecolor='white')
            outputs.append(output)
plt.close(fig)

assert protected_hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected}, 'Article files changed'
source_files = [SOURCE / name for name in ['models.csv', 'solid_reference.csv',
                                         'liquid_correlation.csv', 'liquid_experiment.csv']]
source_files.append(ROOT / 'manuscript/figures/density_ti_6e45e69b79dc.pdf')
manifest = {
    'source_figure': 'manuscript/figures/density_ti_6e45e69b79dc.pdf',
    'changes': ['Remove the PBE 5k series and legend entries from both panels.',
                'Rename PBE 15k to PBE.',
                'Use titles (a) Ti crystal and (b) Ti liquid.',
                'Export each panel independently.',
                'Also export both panels side by side as a combined figure.'],
    'model_points_plotted': {'a': 30, 'b': 21},
    'PBE_points_plotted': {'a': 10, 'b': 7},
    'article_rebuilt': False,
    'manuscript_files_unchanged': True,
    'sources_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source_files},
    'outputs_sha256': {p.name: sha(p) for p in sorted(OUT.glob('Fig3*.*'))
                      if p.suffix in {'.pdf', '.png'}},
    'generator': str(Path(__file__).resolve().relative_to(ROOT)),
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'outputs': [str(p) for p in outputs],
                  'model_points_plotted': manifest['model_points_plotted'],
                  'manuscript_files_unchanged': True}, indent=2))
