"""Download verified triplets and retain an immutable snapshot for each input set."""
from pathlib import Path
import hashlib
import json
import shlex
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'dft'))
import cluster_test50 as cluster_config


def collect():
    code = (ROOT / 'collect_remote.py').read_text(encoding='utf-8')
    bundle = json.loads(cluster_config.cluster.run('python3 -c ' + shlex.quote(code)))
    assert not bundle['problems'], bundle['problems']
    expected = hashlib.sha256((ROOT.parent / 'TiAl-test50-v1' / 'manifest.json').read_bytes()).hexdigest()
    assert bundle['manifest_sha256'] == expected
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    payload = json.dumps(bundle, indent=2) + '\n'
    (data / 'latest.json').write_text(payload)
    fingerprint = hashlib.sha256(json.dumps([
        (r['metadata']['structure_id'], {xc: label['label_sha256'] for xc, label in r['labels'].items()})
        for r in bundle['records']], sort_keys=True).encode()).hexdigest()[:16]
    snapshot = data / ('triplets-' + fingerprint + '.json')
    if not snapshot.exists():
        snapshot.write_text(payload)
    print(json.dumps({'common': len(bundle['common_ids']), 'validated': bundle['validated_counts'],
                      'snapshot': str(snapshot), 'problems': bundle['problems']}))


if __name__ == '__main__':
    collect()
