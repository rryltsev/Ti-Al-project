#!/bin/bash

#find . -name type.raw|sed s/type.raw//g   > out
#find . -name type.raw|sed s/type.raw//g > out

ourdir=pwd
find '/misc/home6/u3130/SCAN/VASP_dataset_2/AlCu_liq2_scan' -name type.raw|sed s/type.raw//g  | grep nst  > out1
find '/misc/home6/u3130/AlCu_liq+_simplify_database' -name type.raw|sed s/type.raw//g  | grep nfrs  > out2

file="AlCu_liq+_cryst.json"

cat > $file << EOF
{  "_comment": " model parameters",
    "model": {
	"type_map":	["Al", "Cu"],
         "mass_map":	[27, 64],
	"descriptor" :{
	    "type":		"se_a",
	    	"sel":		[120,120],
		"rcut_smth":	1.50,
		"rcut":		7.0,
		"neuron":	[25, 50, 100],
		"resnet_dt":	false,
		"axis_neuron":	16,
		"seed":		0,
		"_comment":	" that's all"
	              },	    
	"fitting_net" : {
	    "neuron":		[240, 240, 240],
	    "resnet_dt":	true,
            "coord_norm":       true,
	    "seed":		1,
	    "_comment":		" that's all"
	},
	"_comment":	" that's all"
    },

    "learning_rate" :{
	"type":		"exp",
	"start_lr":	0.001,
	"decay_steps":	2000,
	"_comment":	"that's all"
    },

    "loss" :{
	"start_pref_e":	0.1,
	"limit_pref_e":	1.0,
	"start_pref_f":	1000,
	"limit_pref_f":	1.0,
	"start_pref_v":	0.01,
	"limit_pref_v":	0.05,
	"_comment":	" that's all"
    },

    "_comment": " traing controls",
    "training" : {
     "training_data": {
	"systems": [
EOF

i=0
for u in `cat out1`; do ((i++)); done
N=$i ## полное число файлов в базе данных
echo $N > info.dat # print the number of directories in training set

j=0; 
n=$N # Here you can put any number smaller or equal N
#n=2062 # take only first ten directories

for u in `cat out1`; do 
((j++))
if (( j < $n)); then
echo \"$u\"\,>>$file; 
else
echo \"$u\">>$file; break
fi
done 

cat >> $file << EOF
],
	"batch_size":	1,
"_comment":		"that's all"
	},
	"validation_data":{
	    "systems": [
EOF

i=0
for u in `cat out2`; do ((i++)); done
N=$i ## полное число файлов в базе данных
echo $N > info.dat # print the number of directories in validation set

j=0; 
n=$N # Here you can put any number smaller or equal N
#n=2062 # take only first ten directories


for u in `cat out2`; do 
((j++))
if (( j < $n)); then
echo \"$u\"\,>>$file; 
else
echo \"$u\">>$file; break
fi
done

cat >> $file << EOF
],
	    "batch_size":	1,
	    "numb_btch":	1,
	    "_comment":		"that's all"
	},
	"numb_steps":	600000,
	"seed":		1,
	"disp_file":	"lcurve.dat",
	"disp_freq":	2000,
	"save_freq":	2000,
	"_comment":	"that's all"
    },   
	    "_comment":		"that's all"
}
EOF