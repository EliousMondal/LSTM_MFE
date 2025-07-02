import numpy as np
import matplotlib.pyplot as plt
import param_Frenkel as param

traj_num = 5
ψ    = np.loadtxt(f"Data/{traj_num}/psi_t_{traj_num}_λ{param.λ}.txt", dtype=np.complex128)

ρ11  = np.abs(ψ[:, 0])**2
ρ22  = np.abs(ψ[:, 1])**2

τ_array = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(τ_array, ρ11.real,  lw=4, ls="-", color="#e74c3c", label="ρ₁₁")
plt.plot(τ_array, ρ22.real,  lw=4, ls="-", color="#3498db", label="ρ₂₂")

plt.xlim(0, param.SimTime)
plt.ylim(-0.1 ,1.1)
plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.25)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"psi_{traj_num}.png", dpi=300, bbox_inches="tight")