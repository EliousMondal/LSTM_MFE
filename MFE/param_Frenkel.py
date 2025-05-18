import numpy as np

# Fundamental constant conversions
fs2au      = 41.341374575751                   # fs to au
cminv2au   = 4.55633*1e-6                      # cm⁻¹ to au
eV2au      = 0.036749405469679                 # eV to au
K2au       = 0.00000316678                     # K to au
Kb         = 8.617333262*1e-5 * eV2au / K2au   # Boltzmann constant in au

# Trajectory parameters
NTraj      = 10000                             # Total number oftrajectories
SimTime    = 500                               # Total simulation time (fs) 
δt_fs      = 0.5                               # Bath time step (fs)
dtN        = δt_fs * fs2au                                               
NSteps     = int(SimTime / δt_fs) + 1

ε          = np.array([50, -50]) * cminv2au
Δ          = 100 * cminv2au
NStates    = ε.shape[0]
initψ      = np.array([1, 0])

ωc         = 200                               # Bath ωc (in cm⁻¹)
λ          = 150                               # Bath λ (in cm⁻¹)
Modes      = 20
NModes     = ε.shape[0] * Modes
T          = 72 * K2au                         # Temperature in au

g_nν       = np.loadtxt(f"BathParams/cj_{ωc}_{λ}_N{Modes}.txt")
ω_nν       = np.loadtxt(f"BathParams/ωj_{ωc}_{λ}_N{Modes}.txt")
# g_ων       = g_nν / (ω_nν**2)

ω_nν_NM    = np.hstack((ω_nν, ω_nν))
ω_nν_NM_sq = ω_nν_NM ** 2