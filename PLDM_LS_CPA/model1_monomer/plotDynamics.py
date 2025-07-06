import numpy as np
import matplotlib.pyplot as plt

import param_monomer as param


J1t = np.loadtxt(f"Data/R1t.txt", dtype=np.complex128)
R1t_real = J1t.real
R1t_imag = J1t.imag
τ_array = np.linspace(0, param.SimTime, param.NSteps) / 1000

plt.plot(τ_array, R1t_real/np.max(np.abs(J1t)), lw=4, ls="-", color="#e74c3c")
# plt.plot(τ_array, R1t_imag/np.max(np.abs(J1t)), lw=4, ls="-", color="#3498db", label="Imag")

plt.xlim(0, 0.3)
plt.ylim(-1.0 ,1.0)
plt.xticks([0, 0.1, 0.2, 0.3], fontsize=15)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=15)
plt.xlabel("t(ps)", fontsize=15)
plt.ylabel(r"R$^{(1)}$(t)", fontsize=15)
# plt.legend(frameon=False, fontsize=15, handlelength=0.75)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"R1t.png", dpi=300, bbox_inches="tight")