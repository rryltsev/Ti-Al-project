#!/bin/sh
#SBATCH -t 1200
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -p apollo
#SBATCH --mem-per-cpu=8g
#SBATCH --job-name=tial

srun  /misc/home6/u3130/vasp.6.3.2/bin/vasp_std 