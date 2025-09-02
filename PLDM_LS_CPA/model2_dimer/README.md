# How to run the coupled-model model?

We can compute the dimer spectra in the same procedure as the monomer model. Follow the following steps:

- make a directory by the name `Data` with the command `mkdir Data`.
- run the command `sbatch runDynamics.sh` which runs `dynamics.py` in parallel for `NTraj` trajectories stored in the `Data` folder.
- run the command `sbatch runCombine.sh` which runs `combineTraj.py` to combine the response functions from all trajectories as `R1t.txt` in the `Data` folder and also generates the response function for each trajectory within the respective trajectory folder as `R1t_$traj.txt` where `traj` is the index of the trajectory.
- Within each trajectory there is CPA energies for each of the monomer in the dimer as a function of time by the name `energy_CPA_t_$traj_$element.txt` where `traj` is the trajectory index and `element` corresponds to `01`, `02`, `10` or `20` and all of these contain the exact same data, so any of them can be used for training.

Our goal is to predict `R1t_$traj.txt` for a given `energy_CPA_t_$traj_$element.txt`. Some example response functions are plotted inside the `example_dynamics` folder. To obtain a spectra from the average response function, run the command `python fourier1D.py` which generates `R1ω_cPLDM.png` inside the example_dynamics folder. A converged linear spectra for the dimer is currently demonstrated in the `example_dynamics` folder.