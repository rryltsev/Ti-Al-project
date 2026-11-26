from pathlib import Path
import json,re,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];audit=ROOT/'runs/label-fields-20261106-02'
records=[json.loads(x) for x in (audit/'records.jsonl').read_text().splitlines()]
meta={r['geometry_id']:r for r in map(json.loads,(ROOT/'manifests/data_manifest.jsonl').read_text().splitlines())}
report=[]
for r in sorted(records,key=lambda r:r['virial_max_difference_eV'],reverse=True)[:3]:
    lab=meta[r['id']]['labels'][r['XC']];i=lab['frame_index'];src=Path(r['source']);tree=ET.parse(src/'vasprun.xml').getroot()
    def arr(p):return np.array([[float(x) for x in v.text.split()] for v in tree.find(p).findall('v')])
    cell=arr("structure[@name='finalpos']/crystal/varray[@name='basis']");sxml=arr("calculation/varray[@name='stress']")
    b=np.load(Path(lab['coord_path']).with_name('box.npy'))[i].reshape(3,3);volume=np.linalg.det(b)
    snpy=np.load(lab['virial_path'])[i].reshape(3,3)*160.21766208*10/volume
    scalc=np.array([float(x) for x in re.findall(r'in kB\s+([-+.0-9Ee\s]+)',(src/'OUTCAR').read_text())[-1].split()[:6]])
    s6=sxml.ravel()[[0,4,8,1,5,2]];n6=snpy.ravel()[[0,4,8,1,5,2]]
    report.append(dict(id=r['id'],XC=r['XC'],stress_XML_kbar=s6.tolist(),stress_NPY_kbar=n6.tolist(),stress_OUTCAR_kbar=scalc.tolist(),NPY_vs_XML_max_GPa=float(np.max(np.abs(n6-s6))*.1),NPY_vs_OUTCAR_max_GPa=float(np.max(np.abs(n6-scalc))*.1),cell_max_diff_A=float(np.max(np.abs(cell-b)))))
(audit/'virial_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
