import numpy as np
import matplotlib.pyplot as plt
import param_Frenkel as param

# exact_data = np.loadtxt(f"exact_liu.txt")
# mfe_data   = np.loadtxt(f"mfe_{param.λ}.txt")

# ptmfe_ρ00   = np.loadtxt(f"../Frenkel_Biexciton/Data/rho00_t_λ{param.λ}.txt", dtype=np.complex128)
# ptmfe_ρ11   = np.loadtxt(f"../Frenkel_Biexciton/Data/rho11_t_λ{param.λ}.txt", dtype=np.complex128)
# σz_ptmfe    = ptmfe_ρ00 - ptmfe_ρ11

ρ00 = np.loadtxt(f"Data/rho00_t_λ{param.λ}.txt", dtype=np.complex128)
ρ11 = np.loadtxt(f"Data/rho11_t_λ{param.λ}.txt", dtype=np.complex128)
σz  = ρ00 - ρ11

τ_array = np.linspace(0, param.SimTime, param.NSteps)

# plt.plot(mfe_data[:, 0], mfe_data[:, 1], lw=4, color="#3498db", label="ref")
plt.plot(τ_array, σz.real,  lw=4, ls="-", color="#e74c3c", label="⟨σz⟩")
plt.plot(τ_array, ρ00.real, lw=4, ls="-", color="#3498db", label="ρ₀₀")
plt.plot(τ_array, ρ11.real, lw=4, ls="-", color="#f39c12", label="ρ₁₁")
# plt.scatter(exact_data[:, 0], exact_data[:, 1], lw=2, color="k", marker='o', label="Exact")
# plt.ylim(-0.75 ,1.1)
plt.xlim(0, param.SimTime)
plt.yticks([-0.5, 0.0, 0.5, 1.0], fontsize=15)
# plt.xticks([0, 4, 8, 12], fontsize=15)
plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.75)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"pop_λ{param.λ}.png", dpi=300, bbox_inches="tight")
# plt.savefig("sigma_z.pdf", dpi=300, bbox_inches="tight")