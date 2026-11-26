from pathlib import Path
import json,ast,time
import workflow as wf
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text())
for p in R.rglob('*.py'):
 if 'analysis_deps' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'))
# Scheduling uses new-block counts, not the unequal historical counts30/15/15/15.
counts={s:0 for s in plan['order']};assert wf.choose(plan,counts)==('Ti_1873K',5)
counts['Ti_1873K']=5
# Simulate completed analysis without creating replica/analysis data.
original=wf.ROOT
class Virtual:
 def __truediv__(self,other):return self
 def exists(self):return True
wf.ROOT=Virtual()
assert wf.choose(plan,counts)==('Ti_1941K',5)
counts['Ti_1941K']=2;assert wf.choose(plan,counts)==('Ti_1941K',5)
for s in counts:counts[s]=5
assert wf.choose(plan,counts) is None
wf.ROOT=original
assert sum(plan['targets_new_replicas'].values())==20 and not plan['automatic_extension']
assert json.loads((R/'data_confirmation.json').read_text())['status']=='confirmed'
for s in plan['order']:
 cfg=json.loads((R/s/'protocol.json').read_text());assert cfg['legacy']['n_replicas'] in [15,30] and 'min_replicas' not in cfg['convergence']
 assert cfg['production_steps']==40000 and cfg['dt_ps']==.001 and cfg['states'][0]['counts']=={'Al':0,'Ti':512}
print(json.dumps(dict(status='PASS',tests=['round-robin five-replica scheduling','resume partial block','explicit finite plan cap','human dataset confirmation','legacy counts retained','no mandatory new replica quota','physical protocol retained'])))
