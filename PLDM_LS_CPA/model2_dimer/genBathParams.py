import numpy as np

η     = 0.1
ωc    = 1
M     = 100
label = "SB3"

ωj = np.zeros(M)
cj = np.zeros(M)

for imode in range(M):
    # ωj[imode] = ωc * np.log(M / (M - (imode+1) + 0.5))
    # cj[imode] = np.sqrt(η * ωc / M) * ωj[imode]
    ωj[imode] = -ωc * np.log(1 - (imode+1)/(1 + M))
    cj[imode] = np.sqrt(η * ωc / (M + 1)) * ωj[imode]
    
np.savetxt(f"ωj_{label}_N{M}.txt", ωj)
np.savetxt(f"cj_{label}_N{M}.txt", cj)