import json
import numpy as np
from grace_pilot_common import ROOT,dump
def main():
    plan=json.loads((ROOT/'plan.json').read_text());cases=plan['cases'];results={}
    for c in cases:
        p=ROOT/'smoke'/c['name'];o=json.loads((p/'optimizer-check.json').read_text());z=json.loads((p/'zero-step.json').read_text())
        assert o['status']==z['status']=='PASS' and o['steps']==5 and not o['frozen_changed'] and o['fresh_optimizer']
        results[c['name']]=o
    for xc in sorted({c['XC'] for c in cases}):
        hashes=[results[c['name']]['first5_batch_hashes'] for c in cases if c['XC']==xc]
        assert all(len(h)==5 and h==hashes[0] for h in hashes)
        if 'reference_first5_batch_hashes' in plan:assert hashes[0]==plan['reference_first5_batch_hashes'][xc]
    dump(ROOT/'optimizer-gate.json',dict(status='PASS',cases=results,matched_first5_batches_within_XC=True,smoke_weights_discarded=True,
        estimate_training_seconds={k:float(np.median(v['recent_step_seconds']))*10000 for k,v in results.items()},qualification='Warm optimizer timings extrapolated to10000 updates; validation, compilation, export, queue delay excluded.'))
if __name__=='__main__':main()
