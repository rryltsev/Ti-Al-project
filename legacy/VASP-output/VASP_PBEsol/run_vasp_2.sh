#!/bin/sh
#SBATCH -t 1200
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=4g
#SBATCH --job-name=tial

srun  /misc/home6/u3130/vasp.6.3.2/bin/vasp_std 