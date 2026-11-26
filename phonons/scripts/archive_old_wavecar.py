"""Relocate one superseded WAVECAR to main-home archive, retaining its old path."""
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    os.umask(0o077)
    root=Path('/home6/u3129/tial-phonons/alpha-ti-xc-phonons-20261021-01')
    source=root/'r2SCAN/production-02/force-001-restart-02/WAVECAR'
    proof=json.loads((root/'r2SCAN/production-02/force-001-restart-03/restart-provenance.json').read_text())['checkpoints']['WAVECAR']
    archive=Path('/home/u3129/tial-phonons-archive/alpha-ti-xc-phonons-20261021-01/r2SCAN/force-001-restart-02/WAVECAR')
    assert source.is_file() and not source.is_symlink()
    assert source.resolve().is_relative_to(root.resolve())
    assert not archive.exists()
    archive.parent.mkdir(parents=True,exist_ok=True)
    assert archive.parent.resolve().is_relative_to(Path('/home/u3129').resolve())
    original=source.stat()
    assert original.st_size==proof['bytes']==25201037952
    assert original.st_dev!=archive.parent.stat().st_dev, 'Archive must use another filesystem'
    fs=os.statvfs(archive.parent)
    assert fs.f_bavail*fs.f_frsize>original.st_size+5*1024**3
    partial=archive.with_name('WAVECAR.partial')
    h=hashlib.sha256()
    with source.open('rb') as inp, partial.open('xb') as out:
        while True:
            chunk=inp.read(8*1024**2)
            if not chunk:break
            out.write(chunk);h.update(chunk)
        out.flush();os.fsync(out.fileno())
    assert h.hexdigest()==proof['sha256']
    with partial.open('rb') as f:copied=hashlib.file_digest(f,'sha256').hexdigest()
    assert copied==proof['sha256'] and partial.stat().st_size==original.st_size
    after=source.stat()
    assert (after.st_size,after.st_mtime_ns)==(original.st_size,original.st_mtime_ns)
    partial.rename(archive)
    directory=os.open(archive.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    record=dict(source=str(source),archive=str(archive),bytes=original.st_size,
                sha256=copied,source_sha256=h.hexdigest(),archive_verified=True,
                state='verified_copy_before_relink',epoch=time.time())
    manifest=source.with_name('WAVECAR.archive.json')
    with manifest.open('x') as f:json.dump(record,f,indent=2)
    link=source.with_name('WAVECAR.archive-link')
    os.symlink(str(archive),link)
    os.replace(link,source)
    assert source.is_symlink() and source.stat().st_size==original.st_size
    record.update(state='archived_with_original_path_preserved',epoch=time.time())
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
