from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;P=HERE/'pbe_checkpoint_sweep'
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
RUN=SOURCE/'DPA4-checkpoints/TiAl_tune_pbe_Neo_DPA4_shard_20260920_140047'
BASE=SOURCE/'DPA4-finetune/TiAl_PBE_upd/npy_format'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(RUN/'input.json');result={}
for kind,listing in [('training_data','training_systems.json'),('validation_data','validation_systems.json')]:
    norm=lambda s:s.replace('\\','/').split('/npy_format/',1)[1]
    expected=sorted(norm(s) for s in read(BASE/listing))
    actual=sorted(norm(s) for s in cfg['training'][kind]['systems'])
    assert actual==expected,(kind,len(actual),len(expected))
    result[kind]={'systems':len(actual),'same_system_lists':True}
assert sha(RUN/'lcurve.out')==sha(SOURCE/'DPA4-finetune/lcurve_pbe.out')
assert sha(RUN/'model.ckpt.pt')==sha(RUN/'output_model/ckpt/model.ckpt-23400.pt')
result.update(status='passed',source_config_sha256=sha(RUN/'input.json'),
    same_training_log=True,root_model_is_final_23400=True)
(P/'dataset-identity-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
