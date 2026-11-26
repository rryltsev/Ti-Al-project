"""Fetch the official fp32 counterpart at the same pinned checkpoint revision."""
from pathlib import Path
import hashlib,json,tarfile
from fetch_grace_2l import get
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/grace-precision-20261107-01'
OLD=ROOT/'runs/grace-2l-preparation-20261107-01/checkpoints/GRACE-2L-OMAT-medium-base-fp64'

def main():
    out=RUN/'checkpoint-fp32';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((OLD/'repository-revision.json').read_text());rev=meta['sha']
    filename='checkpoints/GRACE-2L-OMAT-medium-base-checkpoint.tar.gz'
    item=next(x for x in meta['siblings'] if x['rfilename']==filename);expected=item['lfs']['sha256']
    url=f'https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/{rev}/{filename}'
    p=out/Path(filename).name
    if not p.exists():
        data=get(url);assert len(data)==item['lfs']['size'] and hashlib.sha256(data).hexdigest()==expected
        p.write_bytes(data)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    target=out/'extracted';target.mkdir(exist_ok=True)
    with tarfile.open(p) as tar:
        for member in tar.getmembers():
            dst=(target/member.name).resolve();assert dst.is_relative_to(target.resolve()) and (member.isfile() or member.isdir())
            if member.isfile():
                payload=tar.extractfile(member).read();dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():assert dst.read_bytes()==payload
                else:dst.write_bytes(payload)
    models=list(target.rglob('model.yaml'));assert len(models)==1
    card=dict(model='GRACE-2L-OMAT-medium-base',precision='float32',revision=rev,source_url=url,archive_sha256=expected,
        archive_bytes=p.stat().st_size,model_yaml=models[0].relative_to(out).as_posix(),
        files={x.relative_to(out).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in target.rglob('*') if x.is_file()},
        comparison='Same official release as fp64; exact tensor equality after fp64->fp32 cast must be verified, not assumed.')
    (out/'model-card.json').write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps({k:v for k,v in card.items() if k!='files'}))

if __name__=='__main__':main()
