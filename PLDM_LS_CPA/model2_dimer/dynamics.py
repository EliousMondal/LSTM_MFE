import numpy as np
from numpy import random as rd
from mpi4py import MPI
import time 

import os
import sys

import param_Frenkel as param
import pldm as method
import initBath as iBth

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
        
# compiling the code
# ixp_dum      = np.loadtxt(f"Data/1/iRP_1.txt")
# R_dum, P_dum = iBth.initR()#ixp_dum[:, 0], ixp_dum[:, 1]
# ψFdum, ψBdum = method.evolve(R_dum, P_dum, 2, 1, 0)

μMat_sp = np.array([1.0, -0.2, 1.0, -0.2])
μMat_id = np.array([[0, 1], [0, 2], [1, 0], [2, 0]])
        
st_rank = time.time()
  
for iTraj in TaskArray:
    st_traj  = time.time()
    iR0, iP0 = iBth.initR()  
    os.makedirs(f"Data/{iTraj+1}", exist_ok=True)
    
    for iState in range(μMat_sp.shape[0]):
        iF, iB   = μMat_id[iState]
        R, P     = iR0[:], iP0[:]
        ψFt, ψBt, δεt, δεc = method.evolve(R, P, param.NSteps, iF, iB)
    
        np.savetxt(f"Data/{iTraj+1}/psiF_t_{iTraj+1}_{iF}{iB}.txt", ψFt)
        np.savetxt(f"Data/{iTraj+1}/psiB_t_{iTraj+1}_{iF}{iB}.txt", ψBt)
        np.savetxt(f"Data/{iTraj+1}/energy_t_{iTraj+1}_{iF}{iB}.txt", δεt, fmt='%20.10f')
        np.savetxt(f"Data/{iTraj+1}/energy_CPA_t_{iTraj+1}_{iF}{iB}.txt", δεc, fmt='%20.10f')
    
    ed_traj  = time.time()
    print(f"Time for trajectory {iTraj+1} = {np.round(ed_traj-st_traj, 8)}", flush=True)
    
ed_rank = time.time()
print(f"Process {rank} took {np.round(ed_rank - st_rank, 8)} seconds for computing {len(TaskArray)} trajectories.")