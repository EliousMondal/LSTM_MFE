import numpy as np
import numba as nb
from numpy import random as rd

import param_Frenkel as param

@nb.jit(nopython=True)
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
            R[ν + n * param.Modes] = rd.normal(μR_wigner, σR_wigner[ν])
            P[ν + n * param.Modes] = rd.normal(μP_wigner, σP_wigner[ν])

    return R, P