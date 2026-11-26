import json
import numpy as np
from grace_pilot_common import ROOT,dump
def main():
    plan=json.loads((ROOT/'plan.json').read_text());cases=plan['cases'];results={}
    for c in cases:
        p=ROOT/'smoke'/c['name'];o=json.loads((p/'optimizer-check.json').read_text());z=json.loads((p/'zero-step.json').read_text())
        assert o['status']==z['status']=='PASS' and o['steps']==5 and not o['frozen_changed'] and o['fresh_optimizer']
        results[c['name']]=o
    grouped={}
    for xc,seed in sorted({(c['XC'],c['seed']) for c in cases}):
        hashes=[results[c['name']]['first5_batch_hashes'] for c in cases if c['XC']==xc and c['seed']==seed]
        assert all(len(h)==5 and h==hashes[0] for h in hashes)
        if 'reference_first5_batch_hashes' in plan:assert hashes[0]==plan['reference_first5_batch_hashes'][xc]
        grouped[f'{xc}-seed{seed}']=hashes[0]
    if len({c['seed'] for c in cases})>1:assert len({tuple(h) for h in grouped.values()})==len(grouped)
    dump(ROOT/'optimizer-gate.json',dict(status='PASS',cases=results,matched_first5_batches_within_XC_and_seed=True,batches_by_group=grouped,smoke_weights_discarded=True,
        estimate_training_seconds={c['name']:float(np.median(results[c['name']]['recent_step_seconds']))*c['steps'] for c in cases},qualification='Warm optimizer timings extrapolated to planned updates; validation, compilation, export, queue delay excluded.'))
if __name__=='__main__':main()
