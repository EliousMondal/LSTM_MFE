import numpy as np
import matplotlib.pyplot as plt
import param_Frenkel as param

traj_num = 5
δεc  = np.loadtxt(f"Data/{traj_num}/energy_t_{traj_num}_λ{param.λ}_CPA.txt")

δεc1  = δεc[:, 0]
δεc1 -= np.mean(δεc1)
δεc2  = δεc[:, 1]
δεc2 -= np.mean(δεc2)

τ_array = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(τ_array, δεc1/np.max(np.abs(δεc1)), lw=2, ls="-", color="#e74c3c", label="ε1")
plt.plot(τ_array, δεc2/np.max(np.abs(δεc2)), lw=2, ls="-", color="#3498db", label="ε2")

plt.xlim(0, param.SimTime)
plt.ylim(-1.1 ,1.1)
plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.25)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"energy_{traj_num}_CPA.png", dpi=300, bbox_inches="tight")