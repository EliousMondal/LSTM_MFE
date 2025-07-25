import numpy as np

# Fundamental constant conversions
fs2au      = 41.341374575751                # fs to au
cminv2au   = 4.55633*1e-6                   # cm⁻¹ to au
eV2au      = 0.036749405469679              # eV to au
K2au       = 0.00000316678                  # K to au

# Trajectory parameters
NTraj      = 1000                           # Total number oftrajectories
SimTime    = 200                            # Total simulation time (fs) 
δt_fs      = 0.1                            # bath time step (fs)
dtN        = δt_fs * fs2au                                             
NSteps     = int(SimTime / δt_fs) + 1

ε          = 1050 * cminv2au
Δ          = 100 * cminv2au
μ          = np.array([1, -0.2])

NStates    = 2
ψ0         = 0
sampled    = 1

Modes      = 30
NModes     = (NStates-1) * Modes
λ_cm       = 300
ω_ch_cm    = 300
Kb         = 8.617333262*1e-5 * eV2au / K2au   # Boltzmann constant in au 
T          = 215.8 * K2au
β          = 1 / (Kb * T)
# model_no   = 10


g_nν       = np.loadtxt(f"../BathParams/cj_Λ{λ_cm}_Ωc{ω_ch_cm}_N{Modes}.txt")
ω_nν       = np.loadtxt(f"../BathParams/wj_Λ{λ_cm}_Ωc{ω_ch_cm}_N{Modes}.txt")
ω_nν_sq    = ω_nν ** 2