#!/bin/bash
#SBATCH -p polariton
#SBATCH --job-name=MFE              # create a name for your job
#SBATCH --ntasks=5                 # total number of tasks
#SBATCH --cpus-per-task=1           # cpu-cores per task
#SBATCH --mem-per-cpu=1G            # memory per cpu-core
#SBATCH -t 1-00:00:00               # total run time limit (HH:MM:SS)
#SBATCH --output=rho_ij.out
#SBATCH --error=rho_ij.err

mpiexec -n 5 python dynamics.py Data/