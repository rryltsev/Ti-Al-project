#!/bin/sh

### »Ќ—“–” ÷»я
### ‘иксировать DPDATA -- директори€, куда будет копироватьс€ база данных
### ‘иксировать отсечку по ENCUT 
### фиксировать директории в for item in
ENCUT="700.0"

ourdir=`pwd`
xml_FILES_find_root_dir="$ourdir" ## директори€, откуда начнетс€ поиск xml-файлов
DPDATA="$ourdir/TiAl_PBEsol_database"  ## название директории, где будет создана база данных

xml_filter=1;

ndevide=1 # на столько сетов делим...
filteru=150500 # учитываем только данные с числом фреймов меньше filteru
filterd=0  # учитываем только данные с числом фреймов больше filterd

for DIR in "$DPDATA"; do
    if test ! -d $DIR ; then
        mkdir -p $DIR
    fi
done

find ${xml_FILES_find_root_dir} -name *.xml > $ourdir/xml_list.dat

### filter of xml files ###
rm -f $ourdir/xml_list_final.dat
if [ $xml_filter = 1 ] ; then
   echo "Start filtering xml files"
   while read p; do
      # echo "p=$p"
      Enmax=`grep "ENMAX" $p | awk '{print $3}' | tr -d "</i>" | head -1`; cond=$(echo "scale=0; ($Enmax)/1" | bc ); echo $cond
      if (( cond<610 )); then echo $p >> $ourdir/xml_list_final.dat ; fi
   done < $ourdir/xml_list.dat
else
   mv $ourdir/xml_list.dat $ourdir/xml_list_final.dat
fi

### end of the filter of xml files ###

TMP=$DPDATA/temp
if test ! -d $TMP ; then mkdir -p $TMP; else rm -rf $TMP; mkdir -p $TMP; fi

file="process.py"
cat > $ourdir/$file << EOF
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
    fp=open("$ourdir/xml_list_final.dat", "r")
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
        if Encut < $ENCUT :
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

np.savetxt('$TMP/fparam.raw',np.transpose(fparam), delimiter=' ')

print(ms)
ms.to_deepmd_raw('$TMP')
EOF

  
  cd $ourdir; python $file
  echo "transfer to raw done"

  cd $TMP

for DIR in ` ls -d */ | cut -f1 -d'/'`; do
  RAWdir0=$DIR; echo "RAWdir0=${RAWdir0}"; cp $TMP/fparam.raw $TMP/$DIR
  RAWdir="${RAWdir0}_sh"
  $ourdir/shuffle_raw.py ${RAWdir0}  $RAWdir
  cp ${RAWdir0}/type*.raw $RAWdir;  
  cd $RAWdir; pwd
   
  nframe=`wc -l energy.raw | awk '{print $1}'`
  
if (($nframe<filteru && $nframe>filterd)); then
  #nset=`python -c "print(round($nframe/$ndevide))"`
  nset=`python -c "import math ;print(math.ceil($nframe/$ndevide))"`

  $ourdir/raw_to_set.sh $nset > log_dp
  nsetsf=`grep "will make" log_dp | awk '{print $3}'`
  ls -p -v --hide "type*.raw" | grep -v / | xargs rm
  echo "raw to set done"
  
  cd $TMP; pwd
  #ff=$(echo "${f#*_}")
  #fff=$(echo "${ff%%.*}")
  cp -r $RAWdir $DPDATA/${RAWdir}_${i}_nfr${nframe}_nfrs${nset}_nst${nsetsf}
fi
done
  cd $ourdir 
  rm -rf $TMP

