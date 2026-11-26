"""Inclusion–exclusion on illustrative total cell energies. Standard library only."""
import csv
from decimal import Decimal
from pathlib import Path
from cluster_energy import all_subsets, binding_energy, connected_energy

path = Path(__file__).with_name('illustrative_cell_energies.csv')
with path.open(newline='') as f:
    energies = {tuple(r['Al_sites']) if r['Al_sites'] != 'empty' else ():
                Decimal(r['E_total_cell_eV_illustrative']) for r in csv.DictReader(f)}
recursive = {}
for group in all_subsets(tuple('ABCD')):
    lower = [s for s in all_subsets(group) if s != group]
    recursive[group] = energies[group] - sum(recursive[s] for s in lower)
    assert recursive[group] == connected_energy(group, energies)
for name in ['AB','ABC','ABCD']:
    print(f'W({name}) = {recursive[tuple(name)]:+.2f} eV/cluster')
print(f'B(ABCD) = {binding_energy(tuple("ABCD"), energies):+.2f} eV/cluster')
