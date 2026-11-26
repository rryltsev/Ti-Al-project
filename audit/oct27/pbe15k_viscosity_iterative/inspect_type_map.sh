set -e
squeue -j 20961934 -h -o '%i %T' || true
sacct -j 20961934 --format=JobID,State,ExitCode,Start,End -Pn
CUDA_VISIBLE_DEVICES='' DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python - <<'PY'
import json, pathlib, torch
from deepmd.infer import DeepPot
p=pathlib.Path('/home6/u3129/tial-testsets/pbe15k-viscosity-1873-20261027-01')
cfg=json.loads((p/'protocol.json').read_text())
meta=torch.load(cfg['checkpoint']['path'],map_location='cpu',weights_only=True)['model']['_extra_state']
dp=DeepPot(cfg['checkpoint']['path'])
names=dp.get_type_map()
print(json.dumps({'type_map':names,'type_indices':{s:names.index(s) for s in ['Al','Ti']},'step':meta['train_infos']['step'],'dim_fparam':dp.get_dim_fparam(),'dim_aparam':dp.get_dim_aparam(),'completed_replicas':len(list(p.glob('states/*/replicas/replica_*/summary.json')))},indent=2))
PY
