# MFE
This is a database for doing Mean Field Ehrenfest dynamics on simple systems like Frenkel-Biexciton and Spin-Boson models.
We work on basic system-bath models
```
    \hat{\mathrm{H}} = \hat{\mathrm{H}}_{\mathrm{s}} + \hat{\mathrm{H}}_{\mathrm{b}} + \hat{\mathrm{H}}_{\mathrm{sb}}
```
The Ehrenfest dynamics evolves the system degrees of freedom (DOF) using the Schrodinger equation
$$
    i\hbar \frac{\partial}{\partial t} |\psi_{\mathrm{s}}(t)\rangle = (\hat{\mathrm{H}}_{\mathrm{s}} + \hat{\mathrm{H}}_{\mathrm{sb}}(t)) |\psi_{\mathrm{s}}(t)\rangle
$$
The bath degrees of freedom evolve according to the Newton's equations of motion
$$
    \frac{\partial^2}{\partial t^2} R_{\nu}^{(i)} = -\left\langle \frac{\partial \hat{H}}{\partial \hat{R}_{\nu}^{(i)}} \right\rangle
$$


# Fenkel Bi-exciton model
The simple Frenkel excitons consists of two states electronically coupled while having their own dephasing environments.
$$
    \hat{\mathrm{H}} = \varepsilon \sigma_{\mathrm{z}} + \Delta \sigma_{\mathrm{z}} + \sum_{i=1}^{2} \left( \frac{\hat{P}_{i,\nu}^{2}}{2} + \frac{1}{2}\omega_{i,\nu}^2 \hat{R}_{i,\nu}^{2} \right) + \sum_{i,\nu} g_{i,\nu}\hat{R}_{i,\nu}|i\rangle\langle i|
$$