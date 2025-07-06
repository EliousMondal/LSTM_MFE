import numpy as np
import matplotlib.pyplot as plt

import param_monomer as param


traj_num = 2
iF   = 1
iB   = 0
δεt  = np.loadtxt(f"Data/{traj_num}/energy_t_{traj_num}_{iF}{iB}.txt")
δεc  = np.loadtxt(f"Data/{traj_num}/energy_CPA_t_{traj_num}_{iF}{iB}.txt")

δεt -= np.mean(δεt)
δεc -= np.mean(δεc)

τ_array = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(τ_array, δεt/np.max(np.abs(δεt)), lw=1, ls="-", color="#e74c3c", label="ε1")
plt.plot(τ_array, δεc/np.max(np.abs(δεc)), lw=1, ls="-", color="#3498db", label="ε1-CPA")

plt.xlim(0, param.SimTime)
plt.ylim(-1.1 ,1.1)
# plt.xticks([0, 100, 200, 300, 400, 500], fontsize=15)
plt.xticks([0, 50, 100, 150, 200], fontsize=15)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.25)#, ncols=3, columnspacing=1.0, handletextpad=0.25, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"energy_{traj_num}_{iF}{iB}.png", dpi=300, bbox_inches="tight")