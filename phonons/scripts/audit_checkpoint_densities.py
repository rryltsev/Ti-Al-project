"""Dependency-free audit of nonmagnetic VASP density grid completeness and charge."""
import json
import math
from pathlib import Path
from cluster import RECORD

def header(f):
    title=f.readline(); scale=float(f.readline())
    cell=[[float(x)*scale for x in f.readline().split()] for _ in range(3)]
    species=f.readline().split(); counts=[int(x) for x in f.readline().split()]
    mode=f.readline().strip().lower()
    if mode!='direct': raise ValueError('Expected Direct coordinates')
    positions=[[float(x) for x in f.readline().split()[:3]] for _ in range(sum(counts))]
    return dict(cell=cell,species=species,counts=counts,positions=positions)

def audit(path,poscar):
    with poscar.open() as f: reference=header(f)
    with path.open() as f:
        structure=header(f)
        assert structure['species']==reference['species'] and structure['counts']==reference['counts']
        cell_error=max(abs(a-b) for row,ref in zip(structure['cell'],reference['cell']) for a,b in zip(row,ref))
        fractional_error=max(abs(a-b) for row,ref in zip(structure['positions'],reference['positions']) for a,b in zip(row,ref))
        assert cell_error<1e-6 and fractional_error<1e-6
        line=f.readline()
        while line and not line.strip(): line=f.readline()
        shape=[int(x) for x in line.split()]; assert len(shape)==3
        size=math.prod(shape); count=0; total=0.
        while count<size:
            line=f.readline()
            if not line: raise ValueError('Truncated grid')
            values=[float(x) for x in line.split()]
            assert all(math.isfinite(x) for x in values)
            count+=len(values); total+=math.fsum(values)
        assert count==size
        augmentation=[]
        for line in f:
            if not line.strip(): continue
            parts=line.split()
            assert parts[:2]==['augmentation','occupancies'], line
            ion,n=int(parts[2]),int(parts[3]); seen=0
            while seen<n:
                values=[float(x) for x in f.readline().split()]
                if not values: raise ValueError('Truncated augmentation')
                assert all(math.isfinite(x) for x in values)
                seen+=len(values)
            assert seen==n
            augmentation.append(ion)
        if path.name=='CHGCAR':
            assert abs(total/size-10*sum(reference['counts']))<0.01
            assert augmentation==list(range(1,sum(reference['counts'])+1))
        else:
            assert not augmentation
    return dict(grid=shape,values=count,grid_mean=total/size,augmentation_atoms=len(augmentation),
                max_cell_rounding_A=cell_error,max_fractional_rounding=fractional_error,complete=True)

if __name__=='__main__':
    results={}
    for xc,names in [('PBE',['CHGCAR']),('r2SCAN',['CHGCAR','TAUCAR'])]:
        folder=RECORD/'collected'/xc/'production-02/force-001'
        for name in names: results[xc+'/'+name]=audit(folder/name,folder/'POSCAR')
    (RECORD/'density-integrity-audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
