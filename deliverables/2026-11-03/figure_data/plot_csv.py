"""Example: python plot_csv.py Fig06_enthalpy/models.csv --x x_Al --y mixing_enthalpy_kJ_mol --group functional --error SEM_kJ_mol --output plot.png"""
import argparse,csv
from collections import defaultdict
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('csv_file');p.add_argument('--x',required=True);p.add_argument('--y',required=True)
p.add_argument('--group',nargs='*',default=[]);p.add_argument('--error');p.add_argument('--scatter',action='store_true');p.add_argument('--output',default='plot.png')
a=p.parse_args();groups=defaultdict(list)
with open(a.csv_file,encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):
  if r.get(a.x) and r.get(a.y):groups[tuple(r[k] for k in a.group)].append(r)
fig,ax=plt.subplots(layout='constrained')
for key,rows in groups.items():
 rows.sort(key=lambda r:float(r[a.x]));x=[float(r[a.x]) for r in rows];y=[float(r[a.y]) for r in rows]
 if a.scatter:ax.scatter(x,y,s=5,label=' / '.join(key))
 else:ax.errorbar(x,y,yerr=[float(r[a.error] or 0) for r in rows] if a.error else None,fmt='o-',capsize=2,label=' / '.join(key))
ax.set(xlabel=a.x,ylabel=a.y)
if a.group:ax.legend(frameon=False)
fig.savefig(a.output,dpi=200)
