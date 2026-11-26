"""Prepare manuscript assets from the completed checkpoint audit; no model refitting."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'manuscript/figures/supporting'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'manuscript/supporting_data'
DATA.mkdir(parents=True, exist_ok=True)
sources = {
    'checkpoint_comparison.png': 'audit/oct25/pbe_checkpoint_sweep/checkpoint_comparison.png',
    'fixed_composition_volume.png': 'audit/oct25/pbe_checkpoint_sweep/fixed_composition_volume.png',
    'train_test_residuals.png': 'audit/oct25/pbe_train_comparison/train_test_residuals.png',
    'checkpoint_phonons.pdf': 'audit/oct25/pbe_checkpoint_phonons/checkpoint_phonons.pdf',
    'checkpoint_phonon_dos.pdf': 'audit/oct25/pbe_checkpoint_phonons/checkpoint_phonon_dos.pdf',
    'checkpoint_dos_vs_dft.pdf': 'audit/oct25/pbe_checkpoint_phonons/dft_dos_comparison/checkpoint_dos_vs_dft.pdf',
    'dos_at_dft_geometry.pdf': 'audit/oct25/pbe_phonons_dft_geometry/dos_at_dft_geometry.pdf',
    'dos_geometry_effect.pdf': 'audit/oct25/pbe_phonons_dft_geometry/dos_geometry_effect.pdf',
}
manifest = {'figures': [], 'analyses': []}
for name, rel in sources.items():
    src = ROOT / rel
    shutil.copy2(src, OUT / name)
    manifest['figures'].append({'output': f'figures/supporting/{name}', 'source': rel,
                               'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
for name, rel in {
    'checkpoint_errors.json': 'audit/oct25/pbe_checkpoint_sweep/analysis.json',
    'training_bias.json': 'audit/oct25/pbe_train_comparison/analysis.json',
    'phonon_comparisons.json': 'audit/oct25/pbe_checkpoint_phonons/analysis.json',
    'phonon_convergence.json': 'audit/oct25/pbe_checkpoint_phonons/convergence-of-comparisons.json',
    'dft_dos_own_cells.json': 'audit/oct25/pbe_checkpoint_phonons/dft_dos_comparison/analysis.json',
    'dft_dos_identical_geometry.json': 'audit/oct25/pbe_phonons_dft_geometry/analysis.json',
    'checkpoint_identity.json': 'audit/oct25/pbe_train_comparison/checkpoint_stage_audit.json',
    'test009_sensitivity.json': 'audit/oct25/pbe_energy_diagnostic/sensitivity.json',
    'POSCAR-DFT': 'audit/oct25/pbe_phonons_dft_geometry/POSCAR-DFT',
    'POSCAR-test009': 'audit/oct25/pbe_energy_diagnostic/TiAl-test-009/POSCAR',
}.items():
    src = ROOT / rel
    shutil.copy2(src, DATA / name)
    manifest['analyses'].append({'output': f'supporting_data/{name}', 'source': rel,
                                'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})

log = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\DPA4-finetune\lcurve_pbe.out')
shutil.copy2(log, DATA / 'lcurve_pbe.out')
manifest['learning_curve'] = {'source': str(log), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
a = np.loadtxt(log)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
for ax, cols, factor, title, ylabel in [
    (axes[0], (2, 1), 1000, '(a) Energy learning curves', 'Logged energy MAE (meV/atom)'),
    (axes[1], (4, 3), 1, '(b) Force learning curves', 'Logged force MAE (eV/Å)'),
]:
    for col, color, label in zip(cols, ['#246db5', '#d17a13'], ['Training', 'Validation']):
        y = a[:, col] * factor
        ax.plot(a[:, 0] / 1000, y, color=color, lw=.6, alpha=.22)
        # Centered 11-record moving mean; no extrapolation at either end.
        yy = np.convolve(y, np.ones(11) / 11, mode='valid')
        ax.plot(a[5:-5, 0] / 1000, yy, color=color, lw=1.7, label=label)
    ax.axvline(5, color='0.3', ls='--', lw=1, label='Selected 5k')
    ax.set(xlabel='Optimization steps (thousands)', ylabel=ylabel, title=title, xlim=(0, 24))
    ax.grid(alpha=.15)
axes[0].set_ylim(0, 12)
axes[1].set_ylim(.08, .135)
axes[0].legend(frameon=False, fontsize=9)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'learning_curves.{ext}', dpi=220)
plt.close(fig)

groups = json.loads((DATA / 'checkpoint_errors.json').read_text(encoding='utf-8-sig'))['groups']
rows = []
for step, splits in groups.items():
    vals = [splits[s]['energy_rmse_meV_atom'] for s in ['train', 'validation', 'test20']]
    vals += [splits[s]['force_rmse_eV_A'] for s in ['train', 'validation', 'test20']]
    rows.append(f'{int(step)/1000:g} & ' + ' & '.join(f'{v:.3f}' if i < 3 else f'{v:.5f}' for i, v in enumerate(vals)) + r' \\')
(DATA / 'checkpoint_error_rows.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
rows = []
for step, splits in groups.items():
    vals = [splits[s]['stress_rmse_GPa'] for s in ['train', 'validation', 'test20']]
    rows.append(f'{int(step)/1000:g} & ' + ' & '.join(f'{v:.5f}' for v in vals) + r' \\')
(DATA / 'checkpoint_stress_rows.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
(DATA / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('Prepared supporting figures, exact source tables, and provenance in', DATA)
