import numpy as np
import matplotlib.pyplot as plt

fs2au       = 41.341374575751
cminv2au    = 4.55633*1e-6
eV2au       = 0.036749405469679
K2au        = 0.00000316678

# 300.0K → 208  cm⁻¹
# 287.8K → 200  cm⁻¹
# 215.8K → 150  cm⁻¹
# 200.0K → 139  cm⁻¹
# 143.9K → 100  cm⁻¹
# 100.0K → 69.7 cm⁻¹

ω_ch_cm     = 200
λ_cm        = 150
print(ω_ch_cm, λ_cm)

ω_ch        = ω_ch_cm * cminv2au                    # characteristic frequency 
λ           = λ_cm * cminv2au                       # Reorganisation energy
N           = 100                                   # Number of modes needed
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
    # costfunc = np.abs(Fω-(((i-0.5)/N)*λs))
    costfunc = np.abs(Fω-(((i+0.5)/N)*λs))                  # correct formula is +
    # print(np.where(costfunc == np.min(costfunc))[0])
    ωj[i] = ω[np.where(costfunc == np.min(costfunc))[0][0]]
cj = ωj * ((λs/(2*N))**0.5)

np.savetxt(f"ωj_{ω_ch_cm}_{λ_cm}_N{N}.txt",ωj)
np.savetxt(f"cj_{ω_ch_cm}_{λ_cm}_N{N}.txt",cj)