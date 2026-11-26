#!/bin/bash

# Define source and base target directories
ourdir=`pwd`
source_dir="POSCARS/Ti"
base_target_dir="RESULTS/Ti"

# Create the base target directory if it doesn't exist
mkdir -p "$base_target_dir"

# Iterate over all POSCAR.x files in the source directory
for file in "$source_dir"/POSCAR.*; do
    if [[ -f "$file" ]]; then
        # Extract the number x from the filename
        x=$(basename "$file" | grep -oP '(?<=POSCAR\.)\d+')
        
        if [[ -n "$x" ]]; then
            # Create the target directory RESULTS/Ti/x
            target_dir="$base_target_dir/$x"
            mkdir -p "$target_dir"
            
            # Copy the file to the target directory
            cp "$file" "$target_dir/POSCAR"
            echo "Copied $file to $target_dir/POSCAR"
	    cp "$ourdir/INCAR"  "$target_dir/INCAR"
            cp "$ourdir/POTCAR_Ti_pv"  "$target_dir/POTCAR"
	    cp "$ourdir/getKPoints.sh" "$target_dir/getKPoints.sh"
            cp "$ourdir/PRECALC"  "$target_dir/PRECALC"
            cp "$ourdir/run_vasp.sh" "$target_dir/run_vasp.sh"  
	    cd $target_dir
            bash getKPoints.sh
            sbatch run_vasp.sh
            cd $ourdir            
	    
        else
            echo "Skipping $file: No number found in filename."
        fi
    fi
done

echo "File copying complete."

