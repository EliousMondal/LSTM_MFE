import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import param_Frenkel as param


# For plotting trajectory response
iTraj = 1
R1t   = np.loadtxt(f"Data/{iTraj}/R1t_{iTraj}.txt") 
mR1t  = np.max(R1t[:, 0]**2 + R1t[:, 1]**2)

# # For plotting average response
# R1t   = np.loadtxt(f"Data/R1t.txt")
# mR1t  = np.max(R1t[:, 0]**2 + R1t[:, 1]**2)

t     = np.linspace(0, param.SimTime, param.NSteps)

plt.plot(t, R1t[:, 0]/mR1t, lw=4, color="#e74c3c", label="Real")
plt.plot(t, R1t[:, 1]/mR1t, lw=4, color="#3498db", label="Imag")
plt.legend(frameon=False, fontsize=25, handlelength=0.5, handletextpad=0.25)

plt.xlim(0, param.SimTime)
plt.ylim(-1, 1)

plt.xticks(np.linspace(0, param.SimTime, 5), fontsize=25)
plt.yticks(np.linspace(-1.0, 1.0, 5), fontsize=25)

plt.xlabel("t(fs)", fontsize=25)
plt.ylabel(r"R$^{1}$(t)", fontsize=25)

plt.grid()
plt.savefig(f"example_dynamics/R1t_{iTraj}.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"example_dynamics/R1t_average.png", dpi=300, bbox_inches="tight")