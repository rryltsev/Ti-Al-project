#!/bin/sh
#SBATCH -t 1200
#SBATCH -N 2
#SBATCH -n 48
#SBATCH -p apollo
#SBATCH --mem-per-cpu=8g
#SBATCH --job-name=tial

srun  /misc/home6/u3130/vasp.6.3.2/bin/vasp_std 