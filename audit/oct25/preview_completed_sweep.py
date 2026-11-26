from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;P=HERE/'pbe_checkpoint_sweep';sys.path.insert(0,str(HERE.parents[1]/'audit/oct20/deps'))
import numpy as np
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(P/'remote-input.json');rows=read(P/'rows.json');refs=np.load(P/'labels-local-only.npz');out={}
for ik,key in enumerate(cfg['models']):
    if not all((P/f'result-{ik*8+j}.json').exists() for j in range(8)):continue
    vals=[]
    for j in range(8):
        meta=read(P/f'result-{ik*8+j}.json');pp=P/f'result-{ik*8+j}.npz'
        assert sha(pp)==meta['predictions_sha256'] and meta['model_sha256']==cfg['models'][key]['sha256']
        pred=np.load(pp)
        for i in meta['indices']:
            r=rows[i];e=1000*(float(pred[f'energy_{i}'])-float(refs[f'energy_{i}']))/r['n_atoms'];f=pred[f'forces_{i}']-refs[f'forces_{i}']
            vals.append({**r,'e':e,'fsse':float((f*f).sum()),'fn':f.size})
    assert len(vals)==1019
    out[key]={}
    for split in ['train','validation','test20']:
        rr=[r for r in vals if r['split']==split];ee=np.array([r['e'] for r in rr]);rich=[r['e'] for r in rr if r['x_Ti']>=.9]
        out[key][split]={'energy_rmse':float(np.sqrt(np.mean(ee**2))),'energy_bias':float(ee.mean()),
                         'force_rmse':float(np.sqrt(sum(r['fsse'] for r in rr)/sum(r['fn'] for r in rr))),
                         'Ti_rich_ge_0.9_energy_bias':float(np.mean(rich))}
(P/'partial_complete_checkpoint_metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
