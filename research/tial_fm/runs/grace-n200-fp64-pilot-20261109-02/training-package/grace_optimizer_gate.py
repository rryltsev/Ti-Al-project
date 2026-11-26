import json
import numpy as np
from grace_pilot_common import ROOT,dump
def main():
    cases=json.loads((ROOT/'plan.json').read_text())['cases'];results={}
    for c in cases:
        p=ROOT/'smoke'/c['name'];o=json.loads((p/'optimizer-check.json').read_text());z=json.loads((p/'zero-step.json').read_text())
        assert o['status']==z['status']=='PASS' and o['steps']==5 and not o['frozen_changed'] and o['fresh_optimizer']
        results[c['name']]=o
    for xc in ['PBE','r2SCAN']:assert results[f'{xc}-full-N200-seed10']['first5_batch_hashes']==results[f'{xc}-restricted-N200-seed10']['first5_batch_hashes']
    dump(ROOT/'optimizer-gate.json',dict(status='PASS',cases=results,matched_first5_batches_within_XC=True,smoke_weights_discarded=True,
        estimate_training_seconds={k:float(np.median(v['recent_step_seconds']))*10000 for k,v in results.items()},qualification='Warm optimizer timings extrapolated to10000 updates; validation, compilation, export, queue delay excluded.'))
if __name__=='__main__':main()
