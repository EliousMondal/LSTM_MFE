import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import param_Frenkel as param


# For plotting trajectory response
iTraj = 2
Et    = np.loadtxt(f"Data/{iTraj}/energy_CPA_t_{iTraj}_01.txt") 
mEt_1 = np.max(np.abs(Et[:, 0]))
mEt_2 = np.max(np.abs(Et[:, 1]))

t     = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(t, Et[:, 0]/mEt_1, lw=4, color="#e74c3c", label="ε₁")
plt.plot(t, Et[:, 1]/mEt_2, lw=4, color="#3498db", label="ε₂")
plt.legend(frameon=False, fontsize=25, handlelength=0.5, handletextpad=0.25)

plt.xlim(0, param.SimTime)
plt.ylim(-1, 1)

plt.xticks(np.linspace(0, param.SimTime, 5), fontsize=25)
plt.yticks(np.linspace(-1.0, 1.0, 5), fontsize=25)

plt.xlabel("t(fs)", fontsize=25)
plt.ylabel(r"ε(t)", fontsize=25)

plt.grid()
plt.savefig(f"example_dynamics/Et_{iTraj}.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"example_dynamics/R1t_average.png", dpi=300, bbox_inches="tight")