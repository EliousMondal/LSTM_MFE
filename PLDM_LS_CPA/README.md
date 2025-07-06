# PLDM Spectroscopy

We compute the coherences using the Partial Linearized density matrix dynamics using which we compute the linear response (linear absorption spectra) of these (an)harmonic systems.

The `model1_monomer` folder contains trajectories for a simple monomeric system. The `Data` folder inside contains the data for each individual trajectory run. For example if we open `model1_monomer/Data/2`, this contains the data for 2nd trajectory run. Inside this ther will be 8 files having information about the time dependent energies and the time dependent forward and backward wavefunctions.

As an example of the file name `psiF_t_2_10.txt`, this represents the forward wavefunction (`psiF` part of the filename) of the 2nd trajectory (`2` part of the filename) for an initial coherence element of `10` or $|1⟩⟨0|$ (represented by `10` part of the filename). The data are of `np.complex128` datatype.

Similarly an example of the file name `energy_CPA_t_2_10.txt`, this represents the energy (with Classical Path Approximation) of the 2nd trajectory (`2` part of the filename) for an initial coherence element of `10` or $|1⟩⟨0|$ (represented by `10` part of the filename). The data are of `np.complex128` datatype.

### The initial goal
We try to train the `psiF_t_2_10.txt` with `energy_CPA_t_2_10.txt` for all the trajectories (the `2` in the file name will be replaced by the trajectory index).
