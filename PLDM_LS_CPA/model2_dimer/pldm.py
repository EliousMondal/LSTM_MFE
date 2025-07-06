import numpy as np
import numba as nb

import param_Frenkel as param
import frenkel_biexciton as model


@nb.jit(nopython=True)
def initψ(iF, iB):
    
    ψF = np.zeros(param.NStates, dtype=np.complex128)
    ψB = np.zeros(param.NStates, dtype=np.complex128)

    ψF[iF] = (1 + 1j) / np.sqrt(2)
    ψB[iB] = (1 - 1j) / np.sqrt(2)
        
    return ψF, ψB


@nb.jit(nopython=True)
def evolve_ψ(R, ψF, ψB, δt):
    
    Hτ    = model.H_sys(R)
    E, U  = np.linalg.eigh(Hτ)
    
    Uf_Hτ = U @ np.diag(np.exp( 1j * E * δt)) @ np.conjugate(U.T)
    Ub_Hτ = U @ np.diag(np.exp(-1j * E * δt)) @ np.conjugate(U.T)
    
    ψF_δt = Uf_Hτ @ ψF
    ψB_δt = ψB @ Ub_Hτ
    
    return ψF_δt, ψB_δt


@nb.jit(nopython=True)
def evolve_R(R, P, ψF, ψB, δt):
    
    F1    = model.F_nν(R, ψF, ψB)
    R_δt  = R + (P * δt) + (0.5 * F1 * δt ** 2)
    
    F2    = model.F_nν(R_δt, ψF, ψB) 
    P_δt  = P + 0.5 * (F1 + F2) * δt
    
    return R_δt, P_δt


@nb.jit(nopython=True)
def evolve_ψR(R, P, ψF, ψB, δt):
    
    ψF_hf, ψB_hf = evolve_ψ(R, ψF, ψB, δt/2)          # half-step system evolution
    R_δt, P_δt   = evolve_R(R, P, ψF_hf, ψB_hf, δt)   # Bath evolution
    ψF_δt, ψB_δt = evolve_ψ(R_δt, ψF_hf, ψB_hf, δt/2) # half-step system evolution
    
    return ψF_δt, ψB_δt, R_δt, P_δt


@nb.jit(nopython=True)
def evolve(R, P, nSteps, iF, iB):
    
    ψFt     = np.zeros((nSteps, param.NStates), dtype=np.complex128)
    ψBt     = np.zeros((nSteps, param.NStates), dtype=np.complex128)
    ψFt[0, :], ψBt[0, :] = initψ(iF, iB)
    
    for iStep in range(1, nSteps):
        ψFt[iStep, :], ψBt[iStep, :], R, P = evolve_ψR(R, P, ψFt[iStep-1, :], ψBt[iStep-1, :], param.dtN)
    
    return ψFt, ψBt