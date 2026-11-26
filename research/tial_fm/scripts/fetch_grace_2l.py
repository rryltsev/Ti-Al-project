"""Pin the official fp64 OMat24 2L checkpoint; never silently update its revision."""
from pathlib import Path
import hashlib,json,tarfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]/'runs/grace-2l-preparation-20261107-01'
REPO='AMS-ICAMS-RUB/grace-foundation-models'
MODEL='GRACE-2L-OMAT-medium-base-fp64'
TAG='checkpoint-v3-uq-v6'
FILENAME=f'checkpoints/{MODEL}-checkpoint.tar.gz'

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TiAl-research/1.0'}),timeout=90) as r:return r.read()

def main():
    out=ROOT/'checkpoints'/MODEL;out.mkdir(parents=True,exist_ok=True)
    api=out/'repository-revision.json'
    if not api.exists():api.write_bytes(get('https://huggingface.co/api/models/'+REPO+'/revision/'+TAG+'?blobs=true'))
    metadata=json.loads(api.read_text());revision=metadata['sha']
    info=next(x for x in metadata['siblings'] if x['rfilename']==FILENAME)
    expected=info['lfs']['sha256'];size=info['lfs']['size']
    url=f'https://huggingface.co/{REPO}/resolve/{revision}/{FILENAME}'
    archive=out/Path(FILENAME).name
    if not archive.exists():
        payload=get(url);assert len(payload)==size and hashlib.sha256(payload).hexdigest()==expected
        archive.write_bytes(payload)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
    target=out/'extracted';target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            dest=(target/member.name).resolve();assert dest.is_relative_to(target.resolve())
            assert member.isfile() or member.isdir(),member.name
            if member.isfile():
                payload=tar.extractfile(member).read();dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert dest.read_bytes()==payload
                else:dest.write_bytes(payload)
    models=list(target.rglob('model.yaml'));assert len(models)==1
    files={p.relative_to(out).as_posix():dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(target.rglob('*')) if p.is_file()}
    card=dict(model=MODEL,layers=2,pretraining='OMat24',precision='float64',cutoff_A=6,repository=REPO,revision=revision,source_url=url,archive_sha256=expected,archive_bytes=size,files=files,model_yaml=models[0].relative_to(out).as_posix(),
        license='Academic Software License (installed registry)',
        selection_reason='Current medium 2L OMat24 base fp64 checkpoint supported by installed TensorPotential0.6.1. Official docs deprecate legacy small GRACE-2L-OMAT. OMat24 base avoids extra SMAX or sAlex/MPtrj pretraining and energy-reweighting stages; fp64 supports derivative/phonon checks. No speed claim transferred from FS.',
        qualification='Weights pinned; Ti-Al numerical baseline and harmonic gates not yet passed.')
    (out/'model-card.json').write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps({k:v for k,v in card.items() if k!='files'}))

if __name__=='__main__':main()
