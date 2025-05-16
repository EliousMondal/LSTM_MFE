import numpy as np
from numpy import random as rd
from mpi4py import MPI
import time 

import os
import sys

import param_Frenkel as param

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

def initR():
    '''Sampling the initial position and velocities of bath parameters from 
       wigner distribution'''

    β  = 1 / (param.Kb * param.T)

    σR_wigner = 1/np.sqrt(2*param.ω_nν*np.tanh(β*param.ω_nν*0.5))   # 1/√(2ωₙtanh(βωₙ/2))
    σP_wigner = np.sqrt(param.ω_nν/(2*np.tanh(β*param.ω_nν*0.5)))
    μR_wigner = 0
    μP_wigner = 0

    R = np.zeros(param.NModes)
    P = np.zeros(param.NModes)
    for n in range(param.NModes // param.Modes):
        for ν in range(param.Modes):
            # print(f"n = {n}, ν = {ν}, n_ν = {ν + n * param.Modes}")
            R[ν + n * param.Modes] = rd.normal(μR_wigner, σR_wigner[ν])
            P[ν + n * param.Modes] = rd.normal(μP_wigner, σP_wigner[ν])

    return R, P


st = time.time()
os.chdir(TrajDir)
for itraj in TaskArray:
    print(itraj+1, flush=True)
    os.makedirs(f"{itraj+1}", exist_ok=True)
    os.chdir(f"{itraj+1}")
    R0, P0 = initR()
    np.savetxt(f"iRP_{itraj+1}_λ{param.λ}.txt", np.array([R0, P0]).T, fmt='%24.16f')
    os.chdir("../")
    
ed = time.time()
print(f"jobs for rank {rank} finished in {ed-st} seconds")