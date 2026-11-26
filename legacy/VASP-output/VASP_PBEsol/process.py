import dpdata
import numpy as np
from dpdata import LabeledSystem, MultiSystems
from glob import glob 
import subprocess
import os
import csv

os.system('rm -f fparam.raw')
fparam=[]; ms=MultiSystems()
try:
    fp=open("/misc/home6/u3130/Ti_Al/VASP_PBEsol/xml_list_final.dat", "r")
    # Do stuff with fp
    lines = fp.readlines(); 
finally:
    fp.close() 

for f in lines:
    try:
        f=f.rstrip(); ls=LabeledSystem(f); print(ls); print(f); 
    except:
        print(f); ls=[]
    if len(ls)>0 :
        xstr = 'grep \"ENMAX\" ' + str(f) + " | awk '{print " + "$" + "3}' | tr -d \"</i>\" | head -1"
        stream = os.popen(xstr); ##print("t = ", float(stream))
        Encut = float(stream.readlines()[0].rstrip('\n')); ## print("Encut=",Encut)
        if Encut < 700.0 :
           ms.append(ls)
           nfrm=ls['energies'].size
           print("size=",nfrm)
           xstr = 'grep \"SIGMA\" ' + str(f) + " | awk '{print " + "$" + "3}' | tr -d \"</i>\" | tail -1"
           stream = os.popen(xstr); # print("t = ", stream.readlines()[0].rstrip('\n'))
           sigma = float(stream.readlines()[0].rstrip('\n'))
           T = sigma/(0.086*0.001)
           print("sigma =",sigma); print("T =",T); print("Encut=",Encut)
           for x in range(nfrm) :
               fparam.append(T)

#print(fparam)

np.savetxt('/misc/home6/u3130/Ti_Al/VASP_PBEsol/TiAl_PBEsol_database/temp/fparam.raw',np.transpose(fparam), delimiter=' ')

print(ms)
ms.to_deepmd_raw('/misc/home6/u3130/Ti_Al/VASP_PBEsol/TiAl_PBEsol_database/temp')
