import numpy as np
from mpi4py import MPI
import time

import param_Frenkel as param

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NTraj = param.NTraj
NTasks = NTraj//size
NRem = NTraj - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
  
        
ρt  = np.zeros((param.NSteps, 4), dtype=np.complex128)
ρBf = np.zeros_like(ρt)

# γW = (2 / param.NStates) * (np.sqrt(param.NStates + 1) - 1)

count = 0
st = time.time()
for iTraj in TaskArray:
    # print(iTraj+1, flush=True)
    ψTraj  = np.loadtxt(f"Data/{iTraj+1}/psi_t_{iTraj+1}_λ{param.λ}.txt", dtype=np.complex128)
    for iStep in range(param.NSteps):
        ρt[iStep, :] += np.outer(ψTraj[iStep, :], np.conjugate(ψTraj[iStep, :])).reshape(4)
    count += 1
print(f"# trajectories = {count} in rank {rank}", flush=True)

comm.Reduce(ρt, ρBf, op=MPI.SUM, root=0)
et = time.time()

if comm.rank == 0:
    ρBf /= param.NTraj
    np.savetxt(f"Data/rho00_t_λ{param.λ}.txt", ρBf[:, 0])
    np.savetxt(f"Data/rho01_t_λ{param.λ}.txt", ρBf[:, 1])
    np.savetxt(f"Data/rho10_t_λ{param.λ}.txt", ρBf[:, 2])
    np.savetxt(f"Data/rho11_t_λ{param.λ}.txt", ρBf[:, 3])
    print(f"Time taken = {et-st} seconds\n")