from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_sweep'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(OUT/'remote-input.json');rows=read(OUT/'rows.json');info={};comparisons=[]
for key in ['5000','23400']:
    meta=read(OUT/f'pilot-{key}.json');pred=np.load(OUT/f'pilot-{key}.npz',allow_pickle=False)
    assert meta['status']=='passed' and meta['model_sha256']==cfg['models'][key]['sha256']
    assert meta['predictions_sha256']==sha(OUT/f'pilot-{key}.npz')
    assert meta['input_sha256']==sha(OUT/'remote-input.json') and meta['script_sha256']==sha(OUT/'checkpoint_sweep_remote.py')
    assert meta['indices']==cfg['pilot_indices']
    for i in meta['indices']:
        r=rows[i]
        if key=='5000' and r['split']=='test20':
            old=read(ROOT/'testset/monitor/predictions/PBE'/(r['id']+'.json'))
            diff={'id':r['id'],'energy_meV_atom':1000*abs(float(pred[f'energy_{i}'])-old['energy_eV'])/r['n_atoms'],
                'force_max_eV_A':float(np.max(abs(pred[f'forces_{i}']-np.array(old['forces_eV_A'])))),
                'virial_max_eV':float(np.max(abs(pred[f'virial_{i}']-np.array(old['virial_eV']))))}
            assert diff['energy_meV_atom']<.005 and diff['force_max_eV_A']<1e-4 and diff['virial_max_eV']<.01,diff
            comparisons.append(diff)
    t=np.array(meta['inference_seconds'])
    info[key]={'startup_seconds':meta['startup_seconds'],'mean_frame_seconds':float(t.mean()),
              'median_frame_seconds':float(np.median(t)),'node':meta['node'],'train_infos':meta['train_infos']}
result={'status':'passed','equivalence_to_existing_5000_export':comparisons,'timings':info,
    'full_sweep_ideal_wall_minutes_8_parallel':float(np.mean([v['mean_frame_seconds'] for v in info.values()])*cfg['total']*5/8/60),
    'assumption':'Excludes queue time and per-task startup, mixed CPU hardware and training atom counts may alter throughput.'}
(OUT/'pilot-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='equivalence_to_existing_5000_export'},indent=2))
