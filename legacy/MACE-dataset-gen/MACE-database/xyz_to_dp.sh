
source /home/u2626/SOFT/anaconda3/bin/activate ase_project

rm -rf deepmd_split
cat > dp.py << EOF
from dpdata import LabeledSystem, MultiSystems
from glob import glob
from ase.io.trajectory import Trajectory, Atoms
import dpdata
"""
process multi systems
"""
fs = glob("./RESULT/**/*.xyz", recursive=True)  # все .xyz файлы  # remeber to change here !!!
print(fs)
ms = MultiSystems()
for f in fs:
    try:
        ls = dpdata.MultiSystems.from_file(f,fmt="ase/structure")
        if len(ls) > 0:
           ms.append(ls)
    except:
        print(f)
print(ms)
##ms.to_deepmd_raw("deepmd")
##ms.to_deepmd_npy("deepmd")

#print(ms.systems)
#print(ms.atom_names)
xtrain=0.85
print(ms.train_test_split(xtrain))
x=ms.train_test_split(xtrain)

#print(x[True])


#x[True].to_deepmd_raw("deepmd_train_ini")
#x[True].to_deepmd_npy("deepmd_train_ini")

#x[False].to_deepmd_raw("deepmd_forTest_ini")
#x[False].to_deepmd_npy("deepmd_forTest_ini")

#ms.to_deepmd_raw("deepmd_all")
#ms.to_deepmd_npy("deepmd_all")

import os
import shutil
mydir="mace_to_deepmd_TiAl"
try :  shutil.rmtree(mydir); os.mkdir(mydir)
except:
    print("dir does not exist")
    os.mkdir(mydir)

x[True].to_deepmd_hdf5(mydir+"/training.hdf5")
x[False].to_deepmd_hdf5(mydir+"/validation.hdf5")

x[True].to_deepmd_raw(mydir+"/training")
x[True].to_deepmd_npy(mydir+"/training")

x[False].to_deepmd_raw(mydir+"/validation")
x[False].to_deepmd_npy(mydir+"/validation")
EOF

python dp.py