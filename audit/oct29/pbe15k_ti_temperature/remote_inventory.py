import pathlib,json,subprocess
p=pathlib.Path('/home6/u3129/tial-testsets')
print('RELEVANT_RUNS', [x.name for x in p.iterdir() if any(k in x.name.lower() for k in ['visc','titan','temperature'])])
print(subprocess.check_output(['squeue','-j','20962153','-o','%i %T %M %l %R'],universal_newlines=True))
r=p/'pbe15k-viscosity-1873-20261027-01'
print('PROGRESS', (r/'progress.json').read_text())
for d in sorted((r/'states').iterdir()):
 print(d.name, len(list(d.glob('replicas/replica_*/summary.json'))))
