"""Prepare immutable PBE15k matched-force run selected by the author."""
from pathlib import Path
p=Path(__file__).parent
s=(p/'dpa4_pair_cluster.py').read_text()
s=s.replace('XC=sys.argv[2]', "XC='PBE'")
s=s.replace("DPA4-'+XC+'-matched-20261101-01", "DPA4-'+XC+'-15k-matched-20261101-01")
s=s.replace("model=models['PBE-5000' if XC=='PBE' else 'r2SCAN-23200'];assert model['article_checkpoint']", "model=models['PBE-15000'];model=dict(model,selection='Explicit user choice 2026-11-01: PBE15k for article Figure2')")
(p/'dpa4_pbe15k_cluster.py').write_text(s)
