import numpy as np
import numba as nb

import param_monomer as param
import monomer as model


@nb.njit
def initψ(iF, iB):
    
    ψF = np.zeros(param.NStates, dtype=np.complex128)
    ψB = np.zeros(param.NStates, dtype=np.complex128)

    ψF[iF] = (1 + 1j) / np.sqrt(2)
    ψB[iB] = (1 - 1j) / np.sqrt(2)
        
    return ψF, ψB


@nb.njit
def evolve_ψ(R, ψF, ψB, δt):
    
    Hτ    = model.H_sys(R)
    E, U  = np.linalg.eigh(Hτ)
    
    Uf_Hτ = U @ np.diag(np.exp( 1j * E * δt)) @ np.conjugate(U.T)
    Ub_Hτ = U @ np.diag(np.exp(-1j * E * δt)) @ np.conjugate(U.T)
    
    ψF_δt = Uf_Hτ @ ψF
    ψB_δt = ψB @ Ub_Hτ
    
    return ψF_δt, ψB_δt


@nb.njit
def evolve_R(R, P, ψF, ψB, δt):
    
    F1    = model.F_nν(R, ψF, ψB)
    R_δt  = R + (P * δt) + (0.5 * F1 * δt ** 2)
    
    F2    = model.F_nν(R_δt, ψF, ψB) 
    P_δt  = P + 0.5 * (F1 + F2) * δt
    
    return R_δt, P_δt


@nb.jit(nopython=True)
def evolve_R_CPA(Rcpa, Pcpa, δt):
    
    F1       = -param.ω_nν_sq * Rcpa
    R_δt     = Rcpa + (Pcpa * δt) + (0.5 * F1 * δt ** 2)
    
    F2       = -param.ω_nν_sq * R_δt 
    P_δt     = Pcpa + 0.5 * (F1 + F2) * δt
    
    return R_δt, P_δt


@nb.njit
def evolve_ψR(R, P, ψF, ψB, δt):
    
    ψF_hf, ψB_hf = evolve_ψ(R, ψF, ψB, δt/2)          # half-step system evolution
    R_δt, P_δt   = evolve_R(R, P, ψF_hf, ψB_hf, δt)   # Bath evolution
    ψF_δt, ψB_δt = evolve_ψ(R_δt, ψF_hf, ψB_hf, δt/2) # half-step system evolution
    
    return ψF_δt, ψB_δt, R_δt, P_δt


@nb.njit
def evolve(R, P, nSteps, iF, iB):
    
    ψFt     = np.zeros((nSteps, param.NStates), dtype=np.complex128)
    ψBt     = np.zeros((nSteps, param.NStates), dtype=np.complex128)
    ψFt[0, :], ψBt[0, :] = initψ(iF, iB)
    
    δεt       = np.zeros(nSteps, dtype=np.float64)
    δεt[0]    = model.H_sb(R)
    
    Rc, Pc    = R[:], P[:]
    δεc       = np.zeros_like(δεt)
    δεc[0]    = model.H_sb(Rc)
    
    for iStep in range(1, nSteps):
        ψFt[iStep, :], ψBt[iStep, :], R, P = evolve_ψR(R, P, ψFt[iStep-1, :], ψBt[iStep-1, :], param.dtN)
        δεt[iStep]      = model.H_sb(R)
        
        Rc, Pc          = evolve_R_CPA(Rc, Pc, param.dtN)
        δεc[iStep]      = model.H_sb(Rc)
        
    return ψFt, ψBt, δεt, δεc