import numpy as np
from numpy import random as rd
import time 
import os
import sys

import param_Frenkel as param
import method
        
# compiling the functions
ψt_dum, δεt_dum = method.evolve(2)
        
st_run = time.time()
  
for iTraj in range(param.NTraj):
    st_traj  = time.time()
    ψt, δεt  = method.evolve(param.NSteps)
    ed_traj  = time.time()
    print(f"Time for trajectory {iTraj+1} = {np.round(ed_traj-st_traj, 8)}", flush=True)
    
    os.makedirs(f"Data/{iTraj+1}", exist_ok=True)
    np.savetxt(f"Data/{iTraj+1}/psi_t_{iTraj+1}_λ{param.λ}.txt", ψt)
    np.savetxt(f"Data/{iTraj+1}/energy_t_{iTraj+1}_λ{param.λ}.txt", δεt)
    
ed_run = time.time()
print(f"Dynamics took {np.round(ed_run - st_run, 8)} seconds for computing {param.NTraj} trajectories.")