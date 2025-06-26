import numpy as np
import matplotlib.pyplot as plt
import param_Frenkel as param

ψ   = np.loadtxt(f"Data/2/psi_t_2_λ{param.λ}.txt", dtype=np.complex128)
δεt = np.loadtxt(f"Data/2/energy_t_2_λ{param.λ}.txt")

ρ11 = np.abs(ψ[:, 0])**2
δε1 = δεt[:, 0]
δε2 = δεt[:, 1]

τ_array = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(τ_array, ρ11.real,  lw=4, ls="-", color="#e74c3c", label="ρ₁₁")
# plt.plot(τ_array, δε1/np.max(δε1), lw=2, ls="-", color="#3498db", label="ε1")
# plt.plot(τ_array, δε2/np.max(δε2), lw=2, ls="-", color="#f39c12", label="ε2")

plt.xlim(0, param.SimTime)
plt.ylim(-0.1 ,1.1)
plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.75)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"psi_2.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"energy_2_2.png", dpi=300, bbox_inches="tight")