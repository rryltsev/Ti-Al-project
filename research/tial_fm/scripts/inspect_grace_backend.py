"""Capture installed public backend sources and metadata without importing TF."""
from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/grace-2l-preparation-20261107-01'
ENV='/home6/u3129/mlip-research/envs/grace_ft_a100'

def main():
    code='''import importlib.metadata as m,json
from pathlib import Path
root=Path(m.distribution('tensorpotential').locate_file('tensorpotential'))
names=['calculator/foundation_models.py','cli/wizard.py','cli/prepare.py','cli/gracemaker.py','cli/train.py','tensorpot.py','tpmodel.py','resources/input_template.yaml','cli/data.py']
print(json.dumps(dict(versions={x:m.version(x) for x in ['tensorpotential','tensorflow','tf-keras','numpy','ase']},root=str(root),sources={x:(root/x).read_text() for x in names})))
'''
    data=json.loads(c.ssh(ENV+'/bin/python -c '+shlex.quote(code)))
    OUT.mkdir(parents=True,exist_ok=True);source=OUT/'backend-source';source.mkdir(exist_ok=True)
    files={}
    for name,s in data.pop('sources').items():
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True)
        payload=s.encode();assert not p.exists() or p.read_bytes()==payload
        if not p.exists():p.write_bytes(payload)
        files[name]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    data.update(environment=ENV,files=files)
    (OUT/'backend-manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(versions=data['versions'],files=len(files))))

if __name__=='__main__':main()
