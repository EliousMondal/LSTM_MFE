import numpy as np
import numba as nb

import param_Frenkel as param
import frenkel_biexciton as model
import initBathMPI as iB
    

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
    """
    Description:
        Evolves the bath according to ∂ₜ²R = -⟨∂H/∂R⟩ for a small time step δt.
    Input     : 
        1) R  → current position coordinates of the bath
        2) P  → current momentum coordinates of the bath
        3) ψ  → current wavefunction of the system
        4) δt → nuclear time step
    Output    :
        The evolved R(δt) and P(δt)
    Procedure :
        1) Compute the electronic force on the nuclei at R → F₁
        2) Update R according to P and F1 -> R(δt) = R + Pδt + F₁δt²/2
        3) Compute the electronic force on the nuclei at R(δt) → F₂
        4) Update P according to → P(δt) = P + (F₁ + F₂)δt²/2
    """
    
    F1       = model.F_nν(R, ψ)
    R_δt     = R + (P * δt) + (0.5 * F1 * δt ** 2)
    
    F2       = model.F_nν(R_δt, ψ) 
    P_δt     = P + 0.5 * (F1 + F2) * δt
    return R_δt, P_δt


@nb.jit(nopython=True)
def evolve_ψR(R, P, ψ, δt):
    """
    Description:
        Evolves the both system and bath with Trotter steps for a small time step δt.
    Input     : 
        1) R  → current position coordinates of the bath
        2) P  → current momentum coordinates of the bath
        3) ψ  → current wavefunction of the system
        4) δt → nuclear time step
    Output    :
        The evolved ψ(δt), R(δt), P(δt)
    Procedure :
        1) Evolve ψ for half nuclear time step using R and P → ψ(δt/2)
        2) Evolve R and P using ψ(δt/2) → R(δt), P(δt)
        3) Evolve ψ for half nuclear time step using R(δt), P(δt) → ψ(δt)
    """
    
    ψ_δt_hf    = evolve_ψ(R, ψ, δt/2)          # half-step system evolution
    R_δt, P_δt = evolve_R(R, P, ψ_δt_hf, δt)   # Bath evolution
    ψ_δt       = evolve_ψ(R_δt, ψ_δt_hf, δt/2) # half-step system evolution
    
    return ψ_δt, R_δt, P_δt


@nb.jit(nopython=True)
def evolve(nSteps):
    """
    Description:
        Evolves the both system and bath for nsteps.
    Input     : 
        nSteps → number of total nuclear time steps in the simulation
    Output    :
        Time evolved wavefunction ψ(t) and system energy fulctutations δε(t)
    Procedure :
        1) Initialize ψ(0) and an empty array to store ψ(t)
        2) Initialize R(0), P(0) 
        3) Create empty array to store δε(t) and compute δε(0) using R(0)
        4) Evolve ψ(t), R(t) and P(t) for nSteps
    """
    
    ψt        = np.zeros((nSteps, 2), dtype=np.complex128)
    ψt[0, :]  = param.initψ
    
    R, P      = iB.initR()
    δεt       = np.zeros((nSteps, 2), dtype=np.float64)
    δεt[0, :] = model.H_sb(R)

    for iStep in range(1, nSteps):
        ψt[iStep, :], R, P = evolve_ψR(R, P, ψt[iStep-1, :], param.dtN)
        δεt[iStep, :]      = model.H_sb(R)
    
    return ψt, δεt


@nb.jit(nopython=True)
def evolve_R_CPA(Rcpa, Pcpa, δt):
    
    F1       = -param.ω_nν_NM_sq * Rcpa
    R_δt     = Rcpa + (Pcpa * δt) + (0.5 * F1 * δt ** 2)
    
    F2       = -param.ω_nν_NM_sq * R_δt 
    P_δt     = Pcpa + 0.5 * (F1 + F2) * δt
    
    return R_δt, P_δt


@nb.jit(nopython=True)
def evolve_CPA(nSteps):
    
    ψt        = np.zeros((nSteps, 2), dtype=np.complex128)
    ψt[0, :]  = param.initψ
    
    R, P      = iB.initR()
    Rc, Pc    = R[:], P[:]
    
    δεt       = np.zeros((nSteps, 2), dtype=np.float64)
    δεt[0, :] = model.H_sb(R)
    
    δεc       = np.zeros_like(δεt)
    δεc[0, :] = model.H_sb(Rc)

    for iStep in range(1, nSteps):
        ψt[iStep, :], R, P = evolve_ψR(R, P, ψt[iStep-1, :], param.dtN)
        δεt[iStep, :]      = model.H_sb(R)
        
        Rc, Pc             = evolve_R_CPA(Rc, Pc, param.dtN)
        δεc[iStep, :]      = model.H_sb(Rc)
    
    return ψt, δεt, δεc