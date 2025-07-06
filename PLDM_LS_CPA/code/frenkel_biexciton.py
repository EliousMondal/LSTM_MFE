import numpy as np
import numba as nb

import param_Frenkel as param


@nb.jit(nopython=True)
def H_sb(R):
    ε_sb    = np.zeros(2)
    ε_sb[0] = np.sum(param.g_nν * R[:param.Modes])
    ε_sb[1] = np.sum(param.g_nν * R[param.Modes:])
    return ε_sb


@nb.jit(nopython=True)
def H_sys(R):
    Hij       = np.zeros((param.NStates, param.NStates), dtype=np.complex128)
    ε_sb      = H_sb(R)
    
    Hij[1, 1] = param.ε[0] + ε_sb[0]
    Hij[2, 2] = param.ε[1] + ε_sb[1]
    
    Hij[1, 2] = param.Δ
    Hij[2, 1] = param.Δ
    
    return Hij


@nb.jit(nopython=True)
def F_nν(R, ψF, ψB):
    
    F    = -param.ω_nν_sq * R
    
    ψF2  = np.absolute(ψF) ** 2
    ψB2  = np.absolute(ψB) ** 2
    
    F[:param.Modes] -= param.g_nν * (ψF2[1] + ψB2[1]) / 2
    F[param.Modes:] -= param.g_nν * (ψF2[2] + ψB2[2]) / 2
    
    return F