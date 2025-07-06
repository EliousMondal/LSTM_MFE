import numpy as np
from mpi4py import MPI
import time

import param_monomer as param

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
   

Rt      = np.zeros(param.NSteps, dtype=np.complex128)
RBf     = np.zeros_like(Rt)

μMat    = np.zeros((2, 2))
μMat[1, 0], μMat[0, 1] =  1.0,  1.0

ρ0       = np.zeros((2, 2))
ρ0[0, 0] = 1.0
μxρ0     = (μMat @ ρ0) - (ρ0 @ μMat)

μMat_val = np.array([1.0, 1.0])
μMat_ind = np.array([[0, 1], [1, 0]])

count = 0
st = time.time()
for iTraj in TaskArray:
    RtTraj = np.zeros(param.NSteps, dtype=np.complex128)
    for iState in range(μMat_val.shape[0]):
        iF, iB   = μMat_ind[iState]
        ψFTraj  = np.loadtxt(f"Data/{iTraj+1}/psiF_t_{iTraj+1}_{iF}{iB}.txt", dtype=np.complex128)
        ψBTraj  = np.loadtxt(f"Data/{iTraj+1}/psiB_t_{iTraj+1}_{iF}{iB}.txt", dtype=np.complex128)
        
        for iStep in range(param.NSteps):
            μt = np.outer(ψFTraj[iStep, :], ψBTraj[iStep, :]) * μMat_val[iState]
            RtTraj[iStep] += 1j * np.trace(μt @ μxρ0)
        
    np.savetxt(f"Data/{iTraj+1}/R1t_{iTraj+1}.txt", RtTraj)
    Rt    += RtTraj[:]
    count += 1
print(f"# trajectories = {count} in rank {rank}", flush=True)

comm.Reduce(Rt, RBf, op=MPI.SUM, root=0)
et = time.time()

if comm.rank == 0:
    RBf /= param.NTraj
    np.savetxt(f"Data/R1t.txt", RBf)
    print(f"Time taken = {et-st} seconds\n")