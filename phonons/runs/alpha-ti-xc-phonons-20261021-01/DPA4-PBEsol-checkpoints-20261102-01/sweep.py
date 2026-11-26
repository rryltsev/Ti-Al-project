import subprocess,sys
from pathlib import Path
for step in [5000,10000,15000,20000]:
 print("CHECKPOINT",step,flush=True)
 subprocess.run([sys.executable,"worker.py"],cwd=Path(str(step)),check=True)
print("SWEEP COMPLETE",flush=True)
