#!/bin/bash
#SBATCH -p polariton
#SBATCH --job-name=iRP              # create a name for your job
#SBATCH --ntasks=50                 # total number of tasks
#SBATCH --cpus-per-task=1           # cpu-cores per task
#SBATCH --mem-per-cpu=1G            # memory per cpu-core
#SBATCH -t 1-00:00:00               # total run time limit (HH:MM:SS)
#SBATCH --output=iRP.out
#SBATCH --error=iRP.err

mpiexec -n 50 python initBathMPI.py Data/