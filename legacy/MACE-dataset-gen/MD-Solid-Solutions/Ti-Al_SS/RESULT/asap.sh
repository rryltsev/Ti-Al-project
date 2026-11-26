#!/bin/bash -l

#SBATCH --job-name=asap         
#SBATCH --nodes=1              
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1            
#SBATCH --time=1200              
#SBATCH --mem=4G                    
#SBATCH --partition=apollo           

# conda activate asap_project

#asap gen_desc  -f TiAl-SS-mace.xyz -p TiAl-SS_mace_ASAP-desc --periodic soap  --crossover 

#asap map -f 'TiAl-SS_mace_ASAP-desc.xyz' -p TiAl-SS_mace_pca-map.png -dm '[*]' -c energy pca 

#asap select -n 200 -f TiAl-SS_mace_ASAP-desc.xyz -dm '[*]' -p TiAl-SS-mace_select-200 --savexyz
#asap select -n 1000 -f TiAl-SS_mace_ASAP-desc.xyz -dm '[*]' -p TiAl-SS-mace_select-10000 --savexyz