# MFE
This is a database for doing Mean Field Ehrenfest dynamics on simple systems like Frenkel-Biexciton and Spin-Boson models.
We work on basic system-bath models

<img src="https://latex.codecogs.com/svg.image?\hat{\mathrm{H}}=\hat{\mathrm{H}}_{\mathrm{s}}&plus;\hat{\mathrm{H}}_{\mathrm{b}}&plus;\hat{\mathrm{H}}_{\mathrm{sb}}" />

The Ehrenfest dynamics evolves the system degrees of freedom (DOF) using the Schrodinger equation

<img src="https://latex.codecogs.com/svg.image?\hat{\mathrm{H}}=\hat{\mathrm{H}}_{\mathrm{s}}&plus;\hat{\mathrm{H}}_{\mathrm{b}}&plus;\hat{\mathrm{H}}_{\mathrm{sb}}i\hbar\frac{\partial}{\partial&space;t}|\psi_{\mathrm{s}}(t)\rangle=(\hat{\mathrm{H}}_{\mathrm{s}}&plus;\hat{\mathrm{H}}_{\mathrm{sb}}(t))|\psi_{\mathrm{s}}(t)\rangle" />

The bath degrees of freedom evolve according to the Newton's equations of motion

<img src="https://latex.codecogs.com/svg.image?\frac{\partial^2}{\partial&space;t^2}R_{\nu}^{(i)}=-\left\langle\frac{\partial\hat{H}}{\partial\hat{R}_{\nu}^{(i)}}\right\rangle" />


# Fenkel Bi-exciton model
The simple Frenkel excitons consists of two states electronically coupled while having their own dephasing environments.
<img src="https://latex.codecogs.com/svg.image?\hat{\mathrm{H}}=\varepsilon\sigma_{\mathrm{z}}&plus;\Delta\sigma_{\mathrm{x}}&plus;\sum_{i,\nu}^{2,M}\left(\frac{\hat{P}_{i,\nu}^{2}}{2}&plus;\frac{1}{2}\omega_{i,\nu}^2\hat{R}_{i,\nu}^{2}\right)&plus;\sum_{i,\nu}^{2,M}g_{i,\nu}\hat{R}_{i,\nu}|i\rangle\langle&space;i|" />

## Dependencies
The basic dependencies for running the code serially are
- `numpy`
- `numba`

If you also want to run the parallelized verison of the code you must have 
- `mpi4py`

## Structure of the code
The MFE code is divided into various sub-files each having its own specific purpose.


#### `frenkel_biexciton.py`
This file contails the definitions of the system dependent functions. 
- `H_sb` contains the feedback of the bath to the system energies
- `H_sys` defines the system Hamiltonian at a specfic set of bath positions
- `F_nν` defines the force on the bath coordinates due to system dynamics

#### `param_Frenkel.py`
Conatins the parameters necessary for the MFE simulation along with the conversion factor of fundamental constants to atomic units

#### `method.py`
Contains the functions to perform dynamics.
- `evolve_ψ` evolves the current wavefunction for a small time step δt
- `evolve_R` evolves the current bath coordinates for a small time step δt
- `evolve_ψR` does the time evolution of both R and ψ for a small time step δt
- `evolve` Does a full time-dependent dynamics of both R and ψ by using the above functions. 

#### `dynamicsSerial.py`
Performs the dynamics of `NTraj` number of trajectories and saves the data for each trajectory in the folder `Data/iTraj` where `iTraj` is the index of trajectory.


## Usage
To run the serial version of the code, proceed with the following steps
- make a directory `Data`
- run the code by using the command `python dynamicsSerial.py Data/`