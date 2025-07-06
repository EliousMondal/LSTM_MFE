#!/bin/bash
#SBATCH -p polariton
#SBATCH --job-name=ρ_avg            # create a name for your job
#SBATCH --ntasks=100                # total number of tasks
#SBATCH --cpus-per-task=1           # cpu-cores per task
#SBATCH --mem-per-cpu=1G            # memory per cpu-core
#SBATCH -t 1-00:00:00               # total run time limit (HH:MM:SS)
#SBATCH --output=avg_rho.out
#SBATCH --error=avg_rho.err


mpiexec -n 100 python combineTraj.py