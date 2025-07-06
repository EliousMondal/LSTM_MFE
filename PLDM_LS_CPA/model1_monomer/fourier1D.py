import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

import param_monomer as param


fs2au    = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au    = 0.036749405469679
K2au     = 0.00000316678

δω       = 750
ε0       = 1050
ωmin     = ε0 - δω
ωmax     = ε0 + δω
ω_max    = ωmax * (2 * np.pi) * sc.c/(10**13)          
ω_min    = ωmin * (2 * np.pi) * sc.c/(10**13)           
lenω     = 1001
ω        = np.linspace(ω_min, ω_max, lenω)
dω       = ω[1]-ω[0]

R1t      = np.loadtxt(f"Data/R1t.txt", dtype=np.complex128)
R1dim    = R1t.shape[0]
τ_array  = np.linspace(0, param.SimTime, param.NSteps) * fs2au
δtN      = τ_array[1] - τ_array[0]

# Fourier transform to frequency axis
R1ω = np.zeros(len(ω), dtype=np.complex128)
for omega in range(lenω):
    # print(ω[omega], " cm-1")
    exp_fact = np.exp(1j * ω[omega] * τ_array / fs2au)
    cos_fact = np.cos(np.pi * τ_array / (2 * np.max(τ_array)))        # smooth function
    int_func = exp_fact * R1t * cos_fact
    R1ω[omega] = -2 * np.sum(int_func) * δtN
    
R1ω_area = np.sum(R1ω) * dω
np.savetxt(f"absorption.txt",R1ω/np.abs(R1ω_area))

# ref_data = np.loadtxt("/scratch/mmondal/specTest/pySpec/PySpec/LinearSpec/dimer/model4/absorption_dimer.txt", dtype=np.complex128)

# Plotting frequency dependent response
pω = np.linspace(ωmin, ωmax, lenω) / 1000 - 1.05
# plt.plot(pω, R1ω.real/np.abs(R1ω_area), lw=4)
plt.plot(pω, -R1ω.imag/np.abs(R1ω_area), lw=4, color="#3498db")
# plt.plot(pω, -R1ω.imag/np.max(-R1ω.imag), lw=4, color="#3498db", label="c-PLDM")
# plt.plot(pω, -ref_data.imag/np.max(-ref_data.imag), lw=2, color="#e74c3c", label="ref-PLDM")
# plt.legend(frameon=False, fontsize=15, handlelength=0.75)
plt.xlabel(r"ω(cm⁻¹)", fontsize = 15)
plt.ylabel(r"R$^{1}$(ω)", fontsize = 15)
plt.xlim(np.min(pω), np.max(pω))
# plt.ylim(0, np.max(-R1ω.imag/np.max(-R1ω.imag))+0.02)
plt.ylim(0, np.max(-R1ω.imag/np.abs(R1ω_area))+0.2)
plt.xticks([-0.5, -0.25, 0.0, 0.25, 0.5], fontsize=15)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=15)
plt.yticks([0, 2, 4, 6, 8, 10], fontsize=15)
plt.grid()
# plt.savefig(f"absorption_compare.png", dpi=300, bbox_inches="tight")
plt.savefig(f"R1ω_cPLDM.png", dpi=300, bbox_inches="tight")
plt.close()