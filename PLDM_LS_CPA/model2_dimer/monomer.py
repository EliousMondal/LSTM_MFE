import numpy as np
import numba as nb

import param_monomer as param


@nb.jit(nopython=True)
def H_sb(R):
    ε_sb      = np.sum(param.g_nν * R[:])
    return ε_sb


@nb.jit(nopython=True)
def H_sys(R):
    Hij       = np.zeros((param.NStates, param.NStates), dtype=np.complex128)
    ε_sb      = H_sb(R)
    
    Hij[1, 1] = param.ε + ε_sb
    
    return Hij


@nb.jit(nopython=True)
def F_nν(R, ψF, ψB):
    
    F    = -param.ω_nν_sq * R
    
    ψF2  = np.absolute(ψF) ** 2
    ψB2  = np.absolute(ψB) ** 2
    
    F[:param.Modes] -= param.g_nν * (ψF2[1] + ψB2[1]) / 2
    
    return F