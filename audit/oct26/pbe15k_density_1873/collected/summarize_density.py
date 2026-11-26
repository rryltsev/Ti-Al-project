"""Summarize completed density trajectories without treating samples as independent."""
import csv
import json
from pathlib import Path
import numpy as np

root = Path.cwd()
cfg = json.loads((root/'protocol.json').read_text())
tasks = json.loads((root/'tasks.json').read_text())
rows = []
for task in tasks:
    state = cfg['states'][task['state']]
    folder = root/'15000/results'/str(task['id'])
    assert json.loads((folder/'summary.json').read_text())['status'] == 'completed'
    data = [json.loads(line) for line in (folder/'production-thermo.jsonl').read_text().splitlines()]
    # Exclude t=0, leaving exactly 3000 equally spaced samples over 30 ps.
    data = [r for r in data if 0 < r['step'] <= cfg['production_steps']]
    interval = cfg['thermo_interval']
    assert [r['step'] for r in data] == list(range(interval, cfg['production_steps']+1, interval))
    rho = np.array([r['rho_g_cm3'] for r in data])
    sems = {}
    for ps in [3, 5, 6]:
        n = round(ps*1000/(cfg['dt_fs']*interval))
        block_means = rho.reshape(-1, n).mean(axis=1)
        sems[str(ps)] = float(block_means.std(ddof=1)/np.sqrt(len(block_means)))
    rows.append(dict(x_Al=state['x_Al'], composition=state['composition'],
                     rho_g_cm3=float(rho.mean()), sem_g_cm3=max(sems.values()),
                     half_drift_g_cm3=float(rho[len(rho)//2:].mean()-rho[:len(rho)//2].mean()),
                     mean_T_K=float(np.mean([r['T_K'] for r in data])),
                     mean_P_GPa=float(np.mean([r['P_GPa'] for r in data])),
                     samples=len(data), block_sem_g_cm3=sems))
fields = [k for k in rows[0] if k != 'block_sem_g_cm3']
with (root/'density_1873K.csv').open('x', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
(root/'density_summary.json').write_text(json.dumps(dict(
    status='completed', T_K=cfg['T_K'], checkpoint=15000, rows=rows,
    note='Block SEM and half-window drift are diagnostics; liquid-state and convergence assessment requires trajectories.'), indent=2))
print(json.dumps(rows, indent=2))
