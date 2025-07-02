import numpy as np
from numpy import random as rd
from mpi4py import MPI
import time 

import os
import sys

import param_Frenkel as param
import method

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TrajDir     = sys.argv[1]
NTraj       = param.NTraj
NTasks      = NTraj//size
NRem        = NTraj - (NTasks*size)
TaskArray   = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
print(TaskArray)
        
# compiling the code
# ixp_dum         = np.loadtxt(f"Data/1/iRP_1_λ150.txt")
# R_dum, P_dum    = ixp_dum[:, 0], ixp_dum[:, 1]
ψtd, δεtd, δεcd = method.evolve_CPA(2)
# ψt_dum, δεt_dum = method.evolve(2)
        
st_rank = time.time()
  
for iTraj in TaskArray:
    st_traj  = time.time()
    
    # ixp          = np.loadtxt(f"Data/{iTraj+1}/iRP_{iTraj+1}_λ150.txt")
    # iR, iP       = ixp[:, 0], ixp[:, 1]
    # iR, iP       = np.loadtxt(f"Data/{iTraj+1}/iRP_{iTraj+1}_λ150.txt")
    # print("file loaded")
    ψt, δεt, δεc = method.evolve_CPA(param.NSteps)
    
#     ed_traj  = time.time()
#     print(f"Time for trajectory {iTraj+1} = {np.round(ed_traj-st_traj, 8)}", flush=True)
    
    os.makedirs(f"Data/{iTraj+1}", exist_ok=True)
    np.savetxt(f"Data/{iTraj+1}/psi_t_{iTraj+1}_λ150.txt", ψt)
    np.savetxt(f"Data/{iTraj+1}/energy_t_{iTraj+1}_λ150.txt", δεt)
    np.savetxt(f"Data/{iTraj+1}/energy_t_{iTraj+1}_λ150_CPA.txt", δεc)
    
ed_rank = time.time()
print(f"Process {rank} took {np.round(ed_rank - st_rank, 8)} seconds for computing {len(TaskArray)} trajectories.")