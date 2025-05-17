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
<img src="https://latex.codecogs.com/svg.image?\hat{\mathrm{H}}=\varepsilon\sigma_{\mathrm{z}}&plus;\Delta\sigma_{\mathrm{z}}&plus;\sum_{i=1}^{2}\left(\frac{\hat{P}_{i,\nu}^{2}}{2}&plus;\frac{1}{2}\omega_{i,\nu}^2\hat{R}_{i,\nu}^{2}\right)&plus;\sum_{i,\nu}g_{i,\nu}\hat{R}_{i,\nu}|i\rangle\langle&space;i|" />