"""One-time cache for all frozen geometries; DFT labels are not used here."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

assert os.environ.get('SLURM_JOB_ID')
bundle = sys.argv[1] if len(sys.argv) > 1 else 'bundle.json'
subprocess.run([sys.executable,'predict.py',bundle,'model_registry.json','predictions'],check=True)
expected = len(json.loads(Path(bundle).read_text())['records']) * 3
paths = list(Path('predictions').glob('*/*.json'))
assert len(paths) == expected
Path('result.json').write_text(json.dumps({'status':'passed','predictions':len(paths),
    'unix':time.time(),'job_id':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
paths.append(Path('result.json'))
paths.extend(p for p in Path('.').rglob('*') if p.is_file() and
             p.name in ['verification.json','model_registry.json','model_frozen.pt2','checkpoint-metadata.json','freeze.log'])
Path('outputs.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for p in paths},indent=2)+'\n')
