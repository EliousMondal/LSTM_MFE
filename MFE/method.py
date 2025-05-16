import numpy as np
import numba as nb

import param_Frenkel as param
import frenkel_biexciton as model
    

@nb.jit(nopython=True)
def evolve_ψ(R, ψ, δt):
    """
    Description:
        Evolves the electronic |ψ⟩ according to i∂ₜ|ψ⟩ = H(R)|ψ⟩ for a small time step δt.
    Input     : 
        1) R  → current position coordinates of the bath
        2) ψ  → current wavefunction of the system
        3) δt → nuclear time step
    Output    :
        The evolved wavefunction ψ(δt)
    Procedure :
        1) Compute the system Hamiltonian with the current bath position, R → Hτ
        2) Diagonalise Hτ → Eigenvalues E and Unitary rotation U
        3) Evolve ψ → U @ exp(-i E δt) @ U†
    """
    
    Hτ     = model.H_sys(R)
    E, U   = np.linalg.eigh(Hτ)
    U_Hτ   = U @ np.diag(np.exp(-1j * E * δt)) @ np.conjugate(U.T)
    return U_Hτ @ ψ


@nb.jit(nopython=True)
def evolve_R(R, P, ψ, δt):
    F1       = model.F_nν(R, ψ)
    R_δt     = R + (P * δt) + (0.5 * F1 * δt ** 2)
    
    F2       = model.F_nν(R_δt, ψ) 
    P_δt     = P + 0.5 * (F1 + F2) * δt
    return R_δt, P_δt


@nb.jit(nopython=True)
def evolve_ψR(R, P, ψ, δt):
    
    ψ_δt_hf    = evolve_ψ(R, ψ, δt/2)          # half-step system evolution
    R_δt, P_δt = evolve_R(R, P, ψ_δt_hf, δt)   # Bath evolution
    ψ_δt       = evolve_ψ(R_δt, ψ_δt_hf, δt/2) # half-step system evolution
    
    return ψ_δt, R_δt, P_δt


@nb.jit(nopython=True)
def evolve(R, P, nSteps):
    ψt        = np.zeros((nSteps, 2), dtype=np.complex128)
    ψt[0, :]  = param.initψ
    
    δεt       = np.zeros((nSteps, 2), dtype=np.float64)
    δεt[0, :] = model.H_sb(R)
    
    for iStep in range(1, nSteps):
        ψt[iStep, :], R, P = evolve_ψR(R, P, ψt[iStep-1, :], param.dtN)
        δεt[iStep, :]      = model.H_sb(R)
    
    return ψt, δεt