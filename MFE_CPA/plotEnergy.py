import numpy as np
import matplotlib.pyplot as plt
import param_Frenkel as param

traj_num = 5
δεt  = np.loadtxt(f"Data/{traj_num}/energy_t_{traj_num}_λ{param.λ}.txt")

δεt1  = δεt[:, 0]
δεt1 -= np.mean(δεt1)
δεt2  = δεt[:, 1]
δεt2 -= np.mean(δεt2)

τ_array = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(τ_array, δεt1/np.max(np.abs(δεt1)), lw=2, ls="-", color="#e74c3c", label="ε1")
plt.plot(τ_array, δεt2/np.max(np.abs(δεt2)), lw=2, ls="-", color="#3498db", label="ε2")

plt.xlim(0, param.SimTime)
plt.ylim(-1.1 ,1.1)
plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.25)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"energy_{traj_num}.png", dpi=300, bbox_inches="tight")