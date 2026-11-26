#!/bin/bash -l

#SBATCH --job-name=asap         
#SBATCH --nodes=1              
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1            
#SBATCH --time=1200              
#SBATCH --mem=8G                    
#SBATCH --partition=apollo  

export NUMBA_CACHE_DIR="$PWD/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"   

# conda activate asap_project

#asap gen_desc  -f TiAl-liq-relax-ss_select-1000.xyz -p TiAl-liq-relax-ss_select-1000_ASAP-desc --periodic soap  --crossover 

asap map -f 'TiAl_liq-relax_ASAP-desc.xyz' -p TiAl-liq-relax-ss_pca.png -dm '[*]' -c energy pca 

#asap select -n 1000 -f TiAl_liq-relax_ASAP-desc.xyz -dm '[*]' -p TiAl-liq-relax-ss_select-1000 --savexyz
#asap select -n 1000 -f TiAl_all_mace_ASAP-desc.xyz -dm '[*]' -p TiAl-SS-mace_select-10000 --savexyz