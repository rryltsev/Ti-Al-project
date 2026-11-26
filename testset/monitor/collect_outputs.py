"""Retrieve a model/inference run, verify every file, and populate local caches."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'dft'))
import cluster_test50 as cluster_config


def collect(run):
    submission = json.loads((run / 'submission.json').read_text())
    remote = submission['remote_root']
    cluster = cluster_config.cluster
    hashes = json.loads(cluster.run('cat ' + shlex.quote(remote + '/outputs.json')))
    for name in hashes:
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
    cmd = 'tar --dereference -czf - -C ' + shlex.quote(remote) + ' -- ' + ' '.join(shlex.quote(n) for n in hashes)
    proc = subprocess.run(cluster.ssh.ssh_args() + ['-o','BatchMode=yes','mlip-umt',cmd],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode='r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in hashes and member.name not in contents
            payload = tar.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == hashes[member.name], member.name
            contents[member.name] = payload
    assert set(contents) == set(hashes)
    assert json.loads(contents['result.json'])['status'] == 'passed'
    for name, payload in contents.items():
        p = run / 'collected' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            assert p.read_bytes() == payload
        else:
            p.write_bytes(payload)
    if 'model_registry.json' in contents:
        registry = json.loads(contents['model_registry.json'])
        previous = json.loads((ROOT / 'model_registry.json').read_text())
        if previous['status'] == 'frozen':
            assert previous == registry, 'Never silently replace pinned models'
        (ROOT / 'model_registry.json').write_bytes(contents['model_registry.json'])
    registry = json.loads((ROOT / 'model_registry.json').read_text())
    for name, payload in contents.items():
        p = PurePosixPath(name)
        if p.parts[0] == 'predictions':
            pred = json.loads(payload)
            if p.parts[1] == 'pretrained':
                pretrained = json.loads((ROOT / 'pretrained_registry.json').read_text())
                assert pred['model_role'] == 'pretrained' and pred['model_sha256'] == pretrained['sha256']
            else:
                assert pred['model_sha256'] == registry['models'][pred['functional']]['sha256']
            target = ROOT / name
        elif len(p.parts) == 2 and p.name in ['model_frozen.pt2', 'checkpoint-metadata.json']:
            if p.name == 'model_frozen.pt2':
                assert hashes[name] == registry['models'][p.parts[0]]['sha256']
            target = ROOT / 'models' / name
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old = target.read_bytes()
            # Identical predictions from a later compute run have new provenance timestamps.
            if p.parts[0] == 'predictions':
                a,b = json.loads(old),json.loads(payload)
                for key in ['model_sha256','poscar_sha256','symbols','energy_eV','forces_eV_A','virial_eV']:
                    assert a[key] == b[key], (name,key)
                continue
            assert old == payload, name
        else:
            target.write_bytes(payload)
    record = {'unix':time.time(),'remote_root':remote,'job_id':submission['job_id'],
              'verified_files':hashes,'status':'collected_and_verified'}
    (run / 'collection.json').write_text(json.dumps(record,indent=2)+'\n')
    if any(name.startswith('predictions/') for name in contents):
        cache = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'predictions').glob('*/*.json')}
        (ROOT / 'prediction-manifest.json').write_text(json.dumps(cache,indent=2)+'\n')
    print(json.dumps({'job_id':submission['job_id'],'verified_files':len(hashes)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    collect(parser.parse_args().run)
