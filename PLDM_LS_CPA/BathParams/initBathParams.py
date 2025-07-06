import numpy as np
import matplotlib.pyplot as plt

fs2au       = 41.341374575751
cminv2au    = 4.55633*1e-6
eV2au       = 0.036749405469679
K2au        = 0.00000316678

# γ⁻¹ =  50 fs = 106.18 cm⁻¹ = ωc 
# γ⁻¹ = 100 fs =  53.09 cm⁻¹ = ωc
# γ⁻¹ = 500 fs =  10.62 cm⁻¹ = ωc

def fs2wc(val):
    """val -> value in fs⁻¹
       returns value in cm⁻¹"""
    val2au = val * fs2au
    return 1 / (cminv2au * val2au)

def wc2fs(val):
    """"val -> value in cm⁻¹
        returns value in fs⁻¹"""
    wc2au = val * cminv2au
    return 1 / (wc2au * fs2au)

# model_no    = 10
ω_ch_cm     = 300
λ_cm        = 300
# γ_fs        = 50

# ω_ch        = fs2wc(γ_fs) * cminv2au              # characteristic frequency 
ω_ch        = ω_ch_cm * cminv2au                    # characteristic frequency
λ           = λ_cm * cminv2au                       # Reorganisation energy
N           = 50                                    # Number of modes needed
ω_ct        = 250 * ω_ch                            # cutoff frequency
ω           = np.linspace(0.00000001, ω_ct, 30000)
dω          = ω[1]-ω[0]

def J(ω, ωc, λ):  
    f1 = 2 * ω * (λ * ωc)
    f2 = (ωc**2) + (ω**2)
    return f1/f2

Fω = np.zeros(len(ω))
for i in range(len(ω)):
    Fω[i] = (4/np.pi) * np.sum(J(ω[:i], ω_ch, λ)/ω[:i]) * dω

λs = Fω[-1]
print(λs/4/cminv2au)

ωj = np.zeros(N)
for i in range(N):
    costfunc = np.abs(Fω-(((i+0.5)/N)*λs))          # correct formula is +
    ωj[i] = ω[np.where(costfunc == np.min(costfunc))[0][0]]
cj = ωj * ((λs/(2*N))**0.5)

print(ωj[-1]/cminv2au)

np.savetxt(f"wj_Λ{λ_cm}_Ωc{ω_ch_cm}_N{N}.txt",ωj)
np.savetxt(f"cj_Λ{λ_cm}_Ωc{ω_ch_cm}_N{N}.txt",cj)