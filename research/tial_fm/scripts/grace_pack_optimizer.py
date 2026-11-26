"""Losslessly compress a completed case's optimizer state, preserving recoverability."""
from pathlib import Path
import hashlib,json,sys,tarfile
from grace_pilot_common import ROOT,dump,sha
w=(ROOT/'cases'/sys.argv[1]).resolve();assert w.is_relative_to((ROOT/'cases').resolve())
t=json.loads((w/'training-summary.json').read_text());assert t['status']=='PASS'
assert json.loads((w/'evaluation-summary.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'ti3al/forces'/sys.argv[1]/'summary.json').read_text())['status']=='PASS'
for n,h in t['hashes'].items():assert sha(w/'final-checkpoint'/n)==h
folder=w/'seed'/str(t['case']['seed'])/'checkpoints'
files=sorted(folder.glob('checkpoint*'));assert {f.name for f in files}=={'checkpoint.index','checkpoint.data-00000-of-00001'}
hashes={f.name:sha(f) for f in files};archive=folder/'optimizer-final.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for f in files:tar.add(f,arcname=f.name)
with tarfile.open(archive) as tar:
    members=tar.getmembers();assert {m.name for m in members}==set(hashes)
    for m in members:
        assert m.isfile();h=hashlib.sha256();stream=tar.extractfile(m)
        for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
        assert h.hexdigest()==hashes[m.name]
for f in files:
    assert not f.is_symlink() and f.resolve().is_relative_to(folder.resolve()) and sha(f)==hashes[f.name];f.unlink()
dump(folder/'ARCHIVE.json',dict(status='PASS',archive_sha256=sha(archive),files=hashes,bytes=archive.stat().st_size,restore='Extract optimizer-final.tar.gz here to restore native optimizer checkpoint files; contains full Adam state.',final_model_preserved=True))
print('OPTIMIZER_ARCHIVE_PASS',archive.stat().st_size)
