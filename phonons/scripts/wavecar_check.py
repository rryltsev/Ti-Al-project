"""Validate VASP WAVECAR headers, record extent and sampled complex coefficients."""
import math
from pathlib import Path
import struct


def inspect_wavecar(path, cell, nkpts=112, nbands=576, encut=550):
    path=Path(path)
    with path.open('rb') as f:
        recl,spin,tag=struct.unpack('<3d',f.read(24))
        assert all(x==int(x) for x in [recl,spin,tag])
        recl,spin,tag=map(int,[recl,spin,tag])
        assert recl>104 and spin==1 and tag in (45200,45210),(recl,spin,tag)
        f.seek(recl)
        head=struct.unpack('<12d',f.read(96))
        nk,nb,cutoff=head[:3]
        assert nk==nkpts and nb==nbands and abs(cutoff-encut)<1e-8,head[:3]
        assert max(abs(head[3+3*i+j]-cell[i][j]) for i in range(3) for j in range(3))<1e-6
        expected=recl*(2+spin*nkpts*(nbands+1))
        assert path.stat().st_size==expected,(path.stat().st_size,expected)
        coefficient_bytes=8 if tag==45200 else 16
        fmt='<32f' if tag==45200 else '<32d'
        sample_bytes=128 if tag==45200 else 256
        plane_counts=[]
        for s in range(spin):
            for k in range(nkpts):
                rec=2+(s*nkpts+k)*(nbands+1)
                f.seek(rec*recl)
                band_header=struct.unpack('<'+str(4+3*nbands)+'d',f.read(8*(4+3*nbands)))
                npw=band_header[0]
                assert npw==int(npw) and npw>0 and npw*coefficient_bytes<=recl
                assert all(math.isfinite(x) for x in band_header)
                plane_counts.append(int(npw))
                for band in [0,nbands-1]:
                    f.seek((rec+1+band)*recl)
                    coeff=struct.unpack(fmt,f.read(sample_bytes))
                    assert all(math.isfinite(x) for x in coeff)
    return dict(bytes=expected,record_bytes=recl,spins=spin,precision_tag=tag,
                nkpts=nkpts,nbands=nbands,encut_eV=cutoff,
                min_plane_waves=min(plane_counts),max_plane_waves=max(plane_counts),
                record_extent_complete=True,band_headers_finite=True,sampled_coefficients_finite=True,
                full_coefficient_scan=False)
