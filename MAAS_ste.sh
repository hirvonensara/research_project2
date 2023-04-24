#!/bin/bash
#SBATCH --job-name=maas_ste
#SBATCH --time=100:00:00
#SBATCH --error="err.txt"
#SBATCH --output="out.txt"
#SBATCH --mem-per-cpu=1G
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=24

python myste_train.py 0 11
