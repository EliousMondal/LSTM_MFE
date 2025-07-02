import numpy as np
import numba as nb

import param_Frenkel as param


@nb.jit(nopython=True)
def H_sb(R):
    """
    Description:
        Computes the contribution from bath to the system.
    Input     : 
        R  → current position coordinates of the bath
    Output    :
        The site energy change due to R
    Procedure :
        Compute δε(t) for each site → δεᵢ(R) = -∑ᵤgᵤ⁽ⁱ⁾Rᵤ⁽ⁱ⁾
    """
    
    ε_sb    = np.zeros(param.NStates)
    ε_sb[0] = -np.sum(param.g_nν * R[:param.Modes])
    ε_sb[1] = -np.sum(param.g_nν * R[param.Modes:])
    return ε_sb


@nb.jit(nopython=True)
def H_sys(R):
    """
    Description:
        Generates the system Hamitonian for a R.
    Input     : 
        R  → current position coordinates of the bath
    Output    :
        The system Hamiltonian at R
    Procedure :
        H(R) → (ε₁ + δε₁)|1⟩⟨1| + (ε₂ + δε₂)|2⟩⟨2| + Δ(|1⟩⟨2| + |2⟩⟨1|)
    """
    
    Hij       = np.zeros((param.NStates, param.NStates), dtype=np.complex128)
    ε_sb      = H_sb(R)
    
    Hij[0, 0] = param.ε[0] + ε_sb[0]
    Hij[1, 1] = param.ε[1] + ε_sb[1]
    
    Hij[0, 1] = param.Δ
    Hij[1, 0] = param.Δ
    
    return Hij


@nb.jit(nopython=True)
def F_nν(R, ψ):
    """
    Description:
        Generates the system dependent force on R.
    Input     : 
        R  → current position coordinates of the bath
        ψ  → current wavefunction of the system
    Output    :
        The Force on bath coordinates
    Procedure :
        Fᵤ⁽ⁱ⁾ = -ωᵤ⁽ⁱ⁾² Rᵤ⁽ⁱ⁾ + gᵤ⁽ⁱ⁾ Rᵤ⁽ⁱ⁾
    """
    
    F    = -param.ω_nν_NM_sq * R
    ψ2   = np.absolute(ψ)**2
    
    F[:param.Modes] += param.g_nν * ψ2[0]
    F[param.Modes:] += param.g_nν * ψ2[1]
    
    return F