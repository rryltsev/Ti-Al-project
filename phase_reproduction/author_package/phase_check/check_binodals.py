from pathlib import Path
from collections import defaultdict
import csv
from reproduce_C46_phase_contacts import tangent

here = Path(__file__).resolve().parent
groups = defaultdict(list)
with (here / 'MC_window_means.csv').open() as f:
    for row in csv.DictReader(f):
        key = (row['model'], int(row['T_K']), int(row['N']))
        groups[key].append((float(row['mean_c_Al']), float(row['mean_delta_mu_eV'])))
with (here / 'C46_phase_boundary_points.csv').open() as f:
    expected = {(r['energetic_model'], int(r['T_K']), int(r['MC_sites'])): r
                for r in csv.DictReader(f) if r['statistics'] == 'MC'}
assert len(groups) == len(expected) == 45
for (label, T, N), windows in groups.items():
    kind = 'HA' if label == 'HA_C40' else label
    row = expected[kind, T, N]
    result = tangent(*zip(*windows))
    assert result['resolved'] == (row['status'] == 'resolved')
    if result['resolved']:
        pair = [100 * result[k] for k in ('c_alpha', 'c_alpha2')]
        target = [float(row[k]) for k in ('c_alpha_Al_at_percent', 'c_alpha2_Al_at_percent')]
        assert max(abs(x-y) for x,y in zip(pair,target)) < 1e-8
        if T == 1000:
            print(kind, T, N, 'alpha / alpha2 (at.% Al):', *(round(v,2) for v in pair))
print('Checked all 45 MC groups against the published calculation data.')
