"""Failure-visible postprocessing after all eight GPU jobs, without afterok stalls."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
results={}
for c in json.loads((ROOT/'plan.json').read_text())['cases']:
    result=subprocess.run([sys.executable,str(ROOT/'grace_pilot_harmonics.py'),'--case',c['name']],cwd=ROOT)
    results[c['name']]=result.returncode
for filename,argument in [('ti3al_phonon_worker.py','analyze'),('ti3al_asr_sensitivity.py',None)]:
    command=[sys.executable,str(ROOT/'ti3al'/filename)]+([argument] if argument else [])
    results[filename]=subprocess.run(command,cwd=ROOT/'ti3al').returncode
(ROOT/'postprocessing-status.json').write_text(json.dumps(results,indent=2))
result=subprocess.run([sys.executable,str(ROOT/'grace_pbe_seed_aggregate.py')],cwd=ROOT)
raise SystemExit(result.returncode or int(any(results.values())))
