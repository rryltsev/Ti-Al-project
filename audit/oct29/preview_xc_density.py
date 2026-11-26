"""Create a clearly provisional Fig. 3 without publishing incomplete article data."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'audit/oct29/xc_liquid_density_extended'
sys.path.insert(0, str(ROOT / 'audit/oct20/deps'))
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student

load = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
folder = Path(load(RUN / 'previews/latest.json')['path'])
snapshot = load(folder / 'snapshot.json')
cfg = load(RUN / 'protocol.json')
completed = load(RUN / 'analysis/statistics.json')
assert snapshot['protocol_sha256'] == completed['protocol_sha256'] == sha(RUN / 'protocol.json')
assert snapshot['job_id'] == completed['job_id'] == '20968916'
manifest = load(RUN / 'collection-manifest.json')
assert all(sha(RUN / 'collected' / name) == meta['sha256'] for name, meta in manifest.items())
assert completed['completed_points'] == load(RUN / 'collection.json')['completed_points']
for name, meta in snapshot['files'].items():
    assert sha(folder / name) == meta['sha256']
for p in completed['points']:
    assert p['production_ps'] == 90 and p['qc']['MSD_10ps_A2'] >= 5
    assert p['qc']['solid_like_fraction_last10'] <= .2

state = snapshot['state']
eq = [json.loads(x) for x in (folder / 'equilibration-thermo.jsonl').read_text().splitlines()]
rows = [json.loads(x) for x in (folder / 'production-thermo.jsonl').read_text().splitlines()]
assert [r['step'] for r in eq] == list(range(0, 10001, 10))
assert [r['step'] for r in rows] == list(range(0, rows[-1]['step'] + 1, 10))
assert 0 < rows[-1]['step'] < state['production_steps']
data = rows[1:]
rho = np.array([r['rho_g_cm3'] for r in data])
np.testing.assert_allclose(rho, [47.867 * 1.66053906660 / r['V_A3_atom'] for r in data], atol=1e-10, rtol=0)

def partial_stats(values):
    values = np.asarray(values)
    assert np.isfinite(values).all()
    checks = {}
    for ps in [3, 5, 6, 9, 10, 15]:
        n = int(ps * 100)
        blocks = len(values) // n
        if blocks < 6:
            continue
        means = values[:blocks*n].reshape(blocks, n).mean(axis=1)
        checks[str(ps)] = dict(sem=float(means.std(ddof=1) / np.sqrt(blocks)), blocks=blocks,
                               used_samples=blocks*n, unused_tail_samples=len(values)-blocks*n)
    result = dict(mean=float(values.mean()), samples=len(values), sem=None, ci95=None, block_checks=checks)
    if checks:
        key = max(checks, key=lambda k: checks[k]['sem'])
        v = checks[key]
        half = float(student.ppf(.975, v['blocks']-1) * v['sem'])
        result.update(sem=v['sem'], ci95=[result['mean']-half, result['mean']+half], selected_block_ps=float(key))
    return result

partial = dict(id=state['id'], functional=state['functional'], key=state['key'], T_K=state['T_K'],
               checkpoint=state['checkpoint'], model_sha256=cfg['models'][str(state['checkpoint'])]['sha256'],
               production_ps=rows[-1]['time_ps'], status='partial', rho=partial_stats(rho),
               qc=dict(T_3N_minus_3_K=float(np.mean([r['T_K'] for r in data])*state['N']/(state['N']-1)),
                       P_GPa=partial_stats([r['P_GPa'] for r in data]),
                       rho_second_half_minus_first=float(rho[len(rho)//2:].mean()-rho[:len(rho)//2].mean()),
                       phase_validated=False,
                       warnings=['Incomplete production window; scalar snapshot only; phase not yet independently validated.']))
if partial['rho']['sem'] is None:
    partial['qc']['warnings'].append('Too short for six 3 ps blocks; no SEM or confidence interval reported.')
all_points = [dict(p, status='completed') for p in completed['points']] + [partial]
timestamp = dt.datetime.fromtimestamp(snapshot['unix'], dt.timezone(dt.timedelta(hours=5)))
result = dict(job_id='20968916', snapshot_local=timestamp.isoformat(), points=all_points,
              completed_points=completed['completed_points'], partial_points=1, planned_points=14,
              notes=['Completed values: 90 ps, 10 fs sampling, maximum block SEM; not total uncertainty.',
                     'Partial mean uses every fully written production record, excluding t=0. Equilibration is excluded.',
                     'Partial scalar density checked against mass/volume. Partial point is not promoted to the completed registry.',
                     'Archived liquid curves remain visible as separate previous 30 ps/100 fs results.',
                     'No interpolation across missing new-series temperatures and no connection to the partial estimate.'])
(folder / 'current_statistics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')

# Read the same source values and references as the manuscript figure.
pbe = load(ROOT / 'audit/oct29/pbe15k_ti_density_temperature/analysis/density_registry_15k.json')
pure = [p for p in pbe['points'] if p.get('composition') == 'Ti' and p.get('kind') == 'homogeneous_NPT']
assert len(pure) == 17
PROJECT = Path('C:/Users/Roman Ryltsev/Desktop/STARTUPS/Ti-Al-X/TiAl-Project')
styles = {'pbe': ('#2878B5', 'PBE 5k'), 'sol': ('#39835B', 'PBEsol'), 'scan': ('#D55E00', r'r$^2$SCAN')}
old = {}
for key in styles:
    with (PROJECT / f'MD-results/density_DPA4-neo/density_scan_{key}.csv').open() as f:
        old[key] = [{'T': float(r['T_target_K']), 'rho': float(r['density_g_cm3'])} for r in csv.DictReader(f, delimiter=';')]
assert len(old['pbe']) == 16 and any(p['T'] == 2000 for p in old['pbe'])
refs = load(ROOT / 'audit/density_reference/references.json')
experiment = load(ROOT / 'audit/density_reference/reiplinger2022_points.json')['points']
phases = [('alpha', 300, 1100, 'o'), ('beta', 1200, 1900, '^'), ('liquid', 2000, 2500, 's')]
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 7,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.9))
fig.subplots_adjust(left=.075, right=.99, bottom=.28, top=.85, wspace=.25)
for key, (color, label) in styles.items():
    for phase, lo, hi, marker in phases:
        ax = axs[1] if phase == 'liquid' else axs[0]
        g = [r for r in old[key] if lo <= r['T'] <= hi]
        archive = phase == 'liquid' and key != 'pbe'
        ax.plot([r['T'] for r in g], [r['rho'] for r in g], color=color, marker=marker,
                ms=3.4, lw=1, ls=':' if archive else '--' if key == 'pbe' else '-',
                mfc='white' if archive or key == 'pbe' else color, alpha=.45 if archive else 1,
                label=label if phase in ['alpha', 'liquid'] and not archive else None)
for phase, lo, hi, marker in phases:
    g = sorted([p for p in pure if p['phase_initial'] == phase], key=lambda p: p['T_K'])
    ax = axs[1] if phase == 'liquid' else axs[0]
    ax.errorbar([p['T_K'] for p in g], [p['rho']['mean'] for p in g], yerr=[p['rho']['sem'] for p in g],
                color=styles['pbe'][0], fmt=marker+'-', ms=4.1, capsize=2, lw=1.1, zorder=5,
                label='PBE 15k' if phase in ['alpha', 'liquid'] else None)
grid = [1873, 2000, 2100, 2200, 2300, 2400, 2500]
for key in ['sol', 'scan']:
    color, label = styles[key]
    g = sorted([p for p in completed['points'] if p['key'] == key], key=lambda p: p['T_K'])
    for a, b in zip(g[:-1], g[1:]):
        if grid.index(b['T_K']) == grid.index(a['T_K'])+1:
            axs[1].plot([a['T_K'], b['T_K']], [a['rho']['mean'], b['rho']['mean']], color=color, lw=1.3)
    axs[1].errorbar([p['T_K'] for p in g], [p['rho']['mean'] for p in g], yerr=[p['rho']['sem'] for p in g],
                    color=color, fmt='s', ms=4.6, capsize=2, lw=1, zorder=7, label=label)
color = styles[partial['key']][0]
axs[1].errorbar([partial['T_K']], [partial['rho']['mean']], yerr=partial['rho']['sem'],
                color=color, fmt='D', mfc='white', mew=1.3, ms=5.5, capsize=2, zorder=9)
for phase in ['alpha', 'beta']:
    a = np.asarray(refs['solid'][phase])
    axs[0].plot(a[:, 0], a[:, 1], c='black', marker='D', ms=3, ls=':', lw=.9,
                label="Zinov'ev (1989)" if phase == 'alpha' else None)
liquid = refs['liquid']
tt = np.linspace(*liquid['plotted_range_K'], 200)
rr = (liquid['c1_kg_m3']+liquid['c2_kg_m3_K']*(tt-liquid['Tm_K']))/1000
u = liquid['expanded_relative_uncertainty']
axs[1].fill_between(tt, rr*(1-u), rr*(1+u), color='.88', zorder=0, label=r'$\pm2\%$ (95% coverage)')
axs[1].plot(tt, rr, c='black', lw=1.1, label='Ntonti et al. (2024)')
axs[1].errorbar([p['T_K'] for p in experiment], [p['rho_g_cm3'] for p in experiment],
                yerr=[[p['rho_g_cm3']-p['lower_g_cm3'] for p in experiment],
                      [p['upper_g_cm3']-p['rho_g_cm3'] for p in experiment]],
                fmt='o', ms=2.6, mfc='white', mec='black', mew=.6, ecolor='.35', elinewidth=.55,
                capsize=1.2, zorder=4, label='Reiplinger & Brillo (2022)')
for T, label in [(1155, r'$\alpha$-$\beta$: 1155 K'), (1943, r'$T_m$: 1943 K')]:
    axs[0].axvline(T, color='.45', ls='-.', lw=.7, zorder=0)
    axs[0].text(T-25, 4.82, label, rotation=90, ha='right', va='top', fontsize=7, color='.3')
axs[0].set(title=r'(a) $\alpha$- and $\beta$-Ti starting cells', xlim=(220, 2010), ylim=(4.20, 4.85))
axs[1].set(title='(b) Liquid-start Ti: current statistics', xlim=(1835, 2530), ylim=(3.87, 4.85))
axs[0].legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, ncol=2, fontsize=7, loc='lower left')
axs[1].legend(frameon=False, fontsize=6.7, loc='upper right', labelspacing=.28, handlelength=1.5)
for ax in axs:
    ax.set(xlabel='Temperature (K)', ylabel=r'Density (g cm$^{-3}$)')
fig.suptitle('Figure 3 - interim results | '+timestamp.strftime('%d %b %Y, %H:%M')+' (UTC+5)', fontsize=11, y=.975)
keys = [Line2D([], [], color='.45', ls=':', marker='s', mfc='white', alpha=.65, label='Previous liquid: 30 ps'),
        Line2D([], [], color='.3', ls='-', marker='s', label='Completed liquid: 90 ps'),
        Line2D([], [], color=styles[partial['key']][0], ls='none', marker='D', mfc='white',
               label=f"Partial {partial['functional']} {partial['T_K']} K: {partial['production_ps']:.2f} ps")]
fig.legend(handles=keys, loc='lower center', bbox_to_anchor=(.5, .115), ncol=3, frameon=False, fontsize=8)
fig.text(.075, .078, 'Liquid key applies to PBEsol and r$^2$SCAN. Error bars: block SEM, not total uncertainty.', fontsize=8)
partial_note = ('Partial point: phase unverified; too short for the planned block SEM.'
                if partial['rho']['sem'] is None else
                'Partial point: mean and block SEM are preliminary; phase not yet verified.')
fig.text(.075, .037, partial_note+' Starting cells do not establish final phase.', fontsize=7.5)
for ext in ['pdf', 'png']:
    fig.savefig(folder / ('density_ti_current.'+ext), dpi=240, bbox_inches='tight')
plt.close(fig)
pressures = [p['qc']['P_GPa']['mean']*1000 for p in completed['points']]
pressure_failures = sum(not p['qc']['P_GPa']['ci95'][0] <= .0001 <= p['qc']['P_GPa']['ci95'][1]
                        for p in completed['points'])
report = ['Текущая статистика плотности Ti: '+timestamp.strftime('%d.%m.%Y %H:%M')+' UTC+5',
          f"Полностью завершено {completed['completed_points']}/14 точек по 90 пс; отдельная предварительная оценка по текущему окну.",
          f'SEM не включает систематическое смещение давления. При 90 пс давление {min(pressures):.1f}-{max(pressures):.1f} МПа; целевое 0.1 МПа исключено блочными 95% интервалами в {pressure_failures} точках.',
          'Архивные и новые данные не объединены в одно среднее. Промежуточный рисунок не заменяет финальную версию статьи.', '']
for p in sorted(all_points, key=lambda p:(p['T_K'], p['functional'])):
    error = f" +/- {p['rho']['sem']:.7f} (SEM)" if p['rho']['sem'] is not None else ' (SEM пока не определён)'
    report.append(f"{p['functional']} {p['T_K']} K: {p['rho']['mean']:.7f}{error} г/см3; {p['production_ps']:.2f} пс; {p['status']}")
(folder / 'CURRENT_REPORT_RU.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
sources = [RUN/'analysis/statistics.json', RUN/'protocol.json', folder/'snapshot.json', folder/'production-thermo.jsonl',
           ROOT/'audit/oct29/pbe15k_ti_density_temperature/analysis/density_registry_15k.json']
(folder/'preview_manifest.json').write_text(json.dumps({'script_sha256':sha(__file__),
    'sources':{str(p):sha(p) for p in sources},'pdf_sha256':sha(folder/'density_ti_current.pdf'),
    'png_sha256':sha(folder/'density_ti_current.png'),'article_files_modified':False},indent=2)+'\n',encoding='utf-8')
(folder/'preview_generator.py').write_bytes(Path(__file__).read_bytes())
sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(report))
print('PDF:',folder/'density_ti_current.pdf')
