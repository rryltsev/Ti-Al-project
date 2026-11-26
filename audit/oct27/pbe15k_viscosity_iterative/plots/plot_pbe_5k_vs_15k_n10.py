"""Plot the archived 5k baseline and cumulative 15k N=10, without pooling."""
from pathlib import Path
import hashlib
import json
import os
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT.parent / 'viscosity_5k_vs_15k' / 'python_deps'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

baseline_path = ROOT / 'baseline_5k.json'
batch_path = ROOT / 'collected/analysis/batch_0010/summary.json'
baseline = json.loads(baseline_path.read_text(encoding='utf-8-sig'))
batch = json.loads(batch_path.read_text(encoding='utf-8-sig'))
assert batch['n_replicas'] == 10 and batch['temperature_K'] == 1873
old = {round(1 - r['c_Ti'], 2): r for r in baseline}
points = sorted(batch['points'], key=lambda r: r['x_Al'])
assert [r['x_Al'] for r in points] == [0.20, 0.45, 0.80]
rows = []
for point in points:
    x = point['x_Al']
    ref = old[x]
    new = point['windows']['40ps']
    assert ref['T'] == 1873 and ref['n_replicas'] == 50
    assert point['n_replicas'] == len(point['sources']) == 10
    rows.append({
        'x_Al': x,
        'PBE_5k_20ps_N50': {k: ref[k] for k in ('eta_mPa_s', 'eta_std_mPa_s')},
        'PBE_15k_40ps_N10': {k: new[k] for k in ('eta_mPa_s', 'eta_std_mPa_s', 'criterion_z')},
        'delta_percent': 100 * (new['eta_mPa_s'] / ref['eta_mPa_s'] - 1),
    })

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 13, 'axes.edgecolor': '#526071',
    'xtick.color': '#354052', 'ytick.color': '#354052',
    'svg.fonttype': 'none',
})
fig, ax = plt.subplots(figsize=(10, 6.6))
fig.subplots_adjust(left=0.105, right=0.96, bottom=0.22, top=0.84)
fig.text(0.105, 0.94, 'Вязкость Ti–Al: PBE 5k и 15k', fontsize=20, weight='bold', color='#17253a')
fig.text(0.105, 0.885, '1873 K  •  Три общих состава  •  15k: накопленные 10 реплик', fontsize=12, color='#526071')
x = [r['x_Al'] for r in rows]
for key, label, color, marker in [
    ('PBE_5k_20ps_N50', 'PBE 5k — 50 реплик × 20 ps', '#2673aa', 'o'),
    ('PBE_15k_40ps_N10', 'PBE 15k — 10 реплик × 40 ps', '#cf6429', 's'),
]:
    ax.errorbar(x, [r[key]['eta_mPa_s'] for r in rows],
                yerr=[r[key]['eta_std_mPa_s'] for r in rows],
                color=color, marker=marker, markersize=8, markerfacecolor='white',
                markeredgewidth=1.8, linewidth=1.8, elinewidth=1.7,
                capsize=5, capthick=1.7, label=label, zorder=3)
ax.set(xlim=(0.1, 0.90), ylim=(0, 6.1),
       xlabel=r'Атомная доля алюминия, $x_{\mathrm{Al}}$',
       ylabel=r'Вязкость $\eta$, мПа·с')
ax.set_xticks(x, ['0,20', '0,45', '0,80'])
ax.grid(axis='y', color='#dfe5ec', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=False, fontsize=11)
ax.annotate('15k: cutoff z = 2,20 > 2', xy=(0.8, rows[-1]['PBE_15k_40ps_N10']['eta_mPa_s']),
            xytext=(0.56, 0.85), color='#a54920', fontsize=10.5,
            arrowprops={'arrowstyle': '->', 'color': '#a54920', 'lw': 1.1},
            ha='left', va='center')
fig.text(0.105, 0.115, 'Погрешности: ±1σ STACIE. Линии соединяют точки для наглядности.', fontsize=10, color='#526071')
fig.text(0.105, 0.075, '15k: предварительные оценки; при x_Al = 0,80 критерий cutoff ещё не выполнен.', fontsize=10, color='#526071')
stem = OUT / 'pbe_5k_vs_15k_N10_1873K'
fig.savefig(stem.with_suffix('.png'), dpi=220, facecolor='white')
fig.savefig(stem.with_suffix('.svg'), facecolor='white')
plt.close(fig)
payload = {
    'temperature_K': 1873, 'error_bars': 'plus/minus one STACIE standard error',
    'rows': rows,
    'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (baseline_path, batch_path)},
    'note': 'Different preparation and trajectory duration; differences cannot be attributed exclusively to checkpoint. Historical trajectories are not pooled.',
}
stem.with_suffix('.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'png': str(stem.with_suffix('.png')), 'svg': str(stem.with_suffix('.svg')), 'rows': rows}, ensure_ascii=True))
