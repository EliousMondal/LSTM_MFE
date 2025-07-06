import numpy as np
import matplotlib.pyplot as plt
import param_monomer as param

traj_num = 2
iF   = 1
iB   = 0
ψF   = np.loadtxt(f"Data/{traj_num}/psiF_t_{traj_num}_{iF}{iB}.txt", dtype=np.complex128)
ψB   = np.loadtxt(f"Data/{traj_num}/psiB_t_{traj_num}_{iF}{iB}.txt", dtype=np.complex128)

ψFr  = ψF[:, 1].real
ψFi  = ψF[:, 1].imag
ψBr  = ψB[:, 1].real
ψBi  = ψB[:, 1].imag
# ρ11  = np.abs(ψ[:, 0])**2
# ρ22  = np.abs(ψ[:, 1])**2

τ_array = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(τ_array, ψFr,  lw=2, ls="-", color="#e74c3c", label="Re(ψF)")
# plt.plot(τ_array, ψBr,  lw=2, ls="-", color="#f39c12", label="Re(ψB)")
plt.plot(τ_array, ψFi,  lw=2, ls="-", color="#3498db", label="Im(ψF)")
# plt.plot(τ_array, ψBi,  lw=1, ls="-", color="#1abc9c", label="Im(ψB)")


plt.xlim(0, param.SimTime)
plt.ylim(-1.1 ,1.1)
# plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
# plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=15)
plt.xticks([0, 50, 100, 150, 200], fontsize=15)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.25)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"psi_{traj_num}_{iF}{iB}.png", dpi=300, bbox_inches="tight")