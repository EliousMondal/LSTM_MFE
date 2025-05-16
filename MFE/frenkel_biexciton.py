import numpy as np
import numba as nb

import param_Frenkel as param


@nb.jit(nopython=True)
def H_sb(R):
    ε_sb    = np.zeros(param.NStates)
    ε_sb[0] = -np.sum(param.g_nν * R[:param.Modes])
    ε_sb[1] = -np.sum(param.g_nν * R[param.Modes:])
    return ε_sb


@nb.jit(nopython=True)
def H_sys(R):
    Hij       = np.zeros((param.NStates, param.NStates), dtype=np.complex128)
    ε_sb      = H_sb(R)
    
    Hij[0, 0] = param.ε[0] + ε_sb[0]
    Hij[1, 1] = param.ε[1] + ε_sb[1]
    
    Hij[0, 1] = param.Δ
    Hij[1, 0] = param.Δ
    
    return Hij


@nb.jit(nopython=True)
def F_nν(R, ψ):
    
    F    = -param.ω_nν_NM_sq * R
    ψ2   = np.absolute(ψ)**2
    
    F[:param.Modes] += param.g_nν * ψ2[0]
    F[param.Modes:] += param.g_nν * ψ2[1]
    
    return F