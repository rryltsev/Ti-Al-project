"""Combine original 30 ps with the new 30 ps without duplicating the boundary."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path.cwd()
cfg = json.loads((ROOT/'protocol.json').read_text())
base = json.loads((ROOT/'baseline.json').read_text())
EV_TO_KJMOL = 96.48533212331

def block_sem(values):
    values = np.asarray(values)
    assert len(values) % 50 == 0
    means = values.reshape(-1, 50).mean(axis=1)
    return float(means.std(ddof=1)/np.sqrt(len(means)))

def statistics(baseline, continuation):
    assert len(baseline) == 301 and len(continuation) == 300
    old = np.array([r['H_external_eV_atom'] for r in baseline])*EV_TO_KJMOL
    new = np.array([r['H_external_eV_atom'] for r in continuation])*EV_TO_KJMOL
    combined = np.concatenate([old, new])
    return dict(H_kJ_mol=float(combined.mean()), H_old_kJ_mol=float(old.mean()),
                H_new_kJ_mol=float(new.mean()), frames_combined=len(combined),
                diagnostic_block_sem_kJ_mol=block_sem(np.concatenate([old[1:], new])),
                old_block_sem_kJ_mol=block_sem(old[1:]), new_block_sem_kJ_mol=block_sem(new),
                new_minus_old_kJ_mol=float(new.mean()-old[1:].mean()),
                restart_sensitivity_H_kJ_mol={str(ps): float(np.concatenate([old, new[ps*10:]]).mean())
                                              for ps in [0, 1, 2, 5]},
                T_mean_K=float(np.mean([r['T_K'] for r in baseline+continuation])),
                rho_mean_g_cm3=float(np.mean([r['rho_g_cm3'] for r in baseline+continuation])))

def main():
    result = {}
    for state in cfg['states']:
        name = state['composition']
        folder = ROOT/'results'/name
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['status'] == 'completed' and summary['steps'] == 30000
        thermo = [json.loads(line) for line in (folder/'production-thermo.jsonl').read_text().splitlines()]
        assert [r['step'] for r in thermo] == list(range(0, 30001, 10))
        new = [r for r in thermo if r['step'] > 0 and r['step'] % 100 == 0]
        result[name] = dict(x_Al=state['x_Al'], **statistics(base[name]['rows'], new))
    al, ti = result['Al'], result['Ti']
    for name, row in result.items():
        x = row['x_Al']
        for h, label in [('H_kJ_mol', 'deltaH_kJ_mol'), ('H_old_kJ_mol', 'deltaH_old_kJ_mol'),
                         ('H_new_kJ_mol', 'deltaH_new_kJ_mol')]:
            row[label] = row[h]-x*al[h]-(1-x)*ti[h] if 0 < x < 1 else 0.
        sem = 'diagnostic_block_sem_kJ_mol'
        row['deltaH_diagnostic_SEM_kJ_mol'] = float(np.sqrt(row[sem]**2+x*x*al[sem]**2+(1-x)**2*ti[sem]**2)) if 0 < x < 1 else 0.
        row['restart_sensitivity_deltaH_kJ_mol'] = {
            ps: (value-x*al['restart_sensitivity_H_kJ_mol'][ps]-(1-x)*ti['restart_sensitivity_H_kJ_mol'][ps]) if 0 < x < 1 else 0.
            for ps, value in row['restart_sensitivity_H_kJ_mol'].items()}
    report = dict(status='completed', functional='r2SCAN', checkpoint=23200, T_K=2000, P_bar=1,
                  old_steps=30000, added_steps=30000, total_steps_per_composition=60000,
                  all_steps_included=True, duplicate_boundary_excluded=True,
                  definition=cfg['enthalpy'], initialization=cfg['initialization'],
                  uncertainty='Diagnostic SEM from twelve 5 ps blocks (original t=0 excluded only for blocks); composition runs assumed independent. Shared pure-element references correlate errors between concentrations. Restart sensitivity is separate and does not trim the primary estimate.',
                  protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
                  compositions=result)
    (ROOT/'enthalpy_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    fields = ['composition', 'x_Al', 'deltaH_old_kJ_mol', 'deltaH_new_kJ_mol', 'deltaH_kJ_mol',
              'deltaH_diagnostic_SEM_kJ_mol', 'T_mean_K', 'rho_mean_g_cm3', 'frames_combined']
    with (ROOT/'mixing_enthalpy_60ps.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dict(composition=name, **row) for name, row in result.items())
    print('STATISTICS_COMPLETE', json.dumps(result['Al0.5Ti0.5']), flush=True)

if __name__ == '__main__':
    main()
