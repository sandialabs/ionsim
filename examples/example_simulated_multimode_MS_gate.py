#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import ionsim as ism

import numpy as np
from scipy.sparse import kron as skron

from icecream import ic

sparse = False

modulate_amplitude = False

include_deybe_waller_effect = False

num_spins = 2

num_modes = 1

spins = [
    ism.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    for _ in range(num_spins)
]
spin_basis = ism.StandardBasis([*spins])

if num_spins == 4:
    target_spins = [spins[1], spins[2]]
    target_wavefunction = np.kron(np.kron(np.array([1,0]), 1/np.sqrt(2)*np.array([1, 0, 0, -1j])), np.array([1,0]))
elif num_spins == 2:
    target_spins = [spins[0], spins[1]]
    target_wavefunction = 1/np.sqrt(2)*np.array([1, 0, 0, -1j])
else:
    raise ValueError('Number of spins must be 2 or 4.')

splittings = [2*np.pi * 88.5e3 * k for k in range(num_modes)]
fock_dimension = 6
modes = [
    ism.MotionalMode.from_frequency(frequency=2*np.pi * 3e6 + splittings[k], fock_dimension=fock_dimension)
    for k in range(num_modes)   
]

basis = ism.StandardBasis([*spins, *modes])

def MS_hamiltonian(basis, modes, etas, rabi_rate, omega_b, omega_r, sparse=False, mod=None):

    motional_basis = ism.StandardBasis([*modes])

    operators = []
    for mode, eta in zip(modes, etas):

        phi = 0
        phase = phi - np.pi/2
        prefactor = np.exp(1j*phase) * rabi_rate/2 * (1j*eta) 

        raise_target_spins = [spin_basis.enlarge_matrix(ism.Pauli.plus, [spin]) for spin in target_spins]

        fock_dimension = len(mode.energy_levels)
        if include_deybe_waller_effect:
            raise_motion = motional_basis.enlarge_matrix(ism.Fock.debye_waller_raising(fock_dimension, eta), [mode])
            lower_motion = motional_basis.enlarge_matrix(ism.Fock.debye_waller_lowering(fock_dimension, eta), [mode])
        else:
            raise_motion = motional_basis.enlarge_matrix(ism.Fock.raising(fock_dimension), [mode])
            lower_motion = motional_basis.enlarge_matrix(ism.Fock.lowering(fock_dimension), [mode])

        operator_0b = prefactor * skron(raise_target_spins[0], raise_motion)
        operator_0r = prefactor * skron(raise_target_spins[0], lower_motion)
        operator_1b = prefactor * skron(raise_target_spins[1], raise_motion)
        operator_1r = prefactor * skron(raise_target_spins[1], lower_motion)

        operators.extend([
            ism.CouplingOperator.from_matrix(basis, operator_0b, omega_b, modulation_function=mod),
            ism.CouplingOperator.from_matrix(basis, operator_0r, omega_r, modulation_function=mod),
            ism.CouplingOperator.from_matrix(basis, operator_1b, omega_b, modulation_function=mod),
            ism.CouplingOperator.from_matrix(basis, operator_1r, omega_r, modulation_function=mod),
        ])
    interaction_frame_energies = [-state.energy for state in basis.states]
    return ism.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)

etas = [0.1 for _ in range(num_modes)]

def compute_pst_area(rabi_rate, detunings, ts, relative_phases=None, amp_mod=None):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    if relative_phases is None:
        relative_phases = np.array([0 for t in ts])
    if amp_mod is None:
        rabi_rates = np.array([rabi_rate for t in ts])
    else:
        rabi_rates = np.array([rabi_rate*amp_mod(t) for t in ts])
    thetas = cumtrapz(detunings, ts, initial=0)
    dalphas = np.array([rabi_rates[i]*np.exp(-1j*(thetas[i] + relative_phases[i])) for i,t in enumerate(ts)])
    alphas = cumtrapz(dalphas, ts, initial=0)
    temp = np.array([alphas[i]*dalphas[i].conj() - dalphas[i]*alphas[i].conj() for i,t in enumerate(ts)])
    dbetas = np.real(-1j/2 * temp)
    betas = cumtrapz(dbetas, ts, initial=0)
    return etas[0]*etas[0]*betas[-1]

if modulate_amplitude:
    tau = 125e-6 # s
    detuning_0 = 2*np.pi * 52e3 # rad./s
    def amp_mod(t):
        width = 16.58e-6
        gaussian = np.exp(-(t - tau/2)**2 / (2 * width**2))
        return gaussian
    ts = np.linspace(0, tau, 2001)
    detunings = np.array([detuning_0 for t in ts])
    area = compute_pst_area(1.0, detunings, ts, amp_mod=amp_mod)
    rabi_rate = np.sqrt((np.pi/2) / np.abs(area))
else:
    tau = 125e-6 # s
    loops = 2
    detuning_0 = 2 * np.pi * loops / tau
    rabi_rate = np.pi / etas[0] / tau * np.sqrt(loops)
    amp_mod = None

ic(detuning_0/(2*np.pi*1e3))
ic(rabi_rate/(2*np.pi*1e3))

omega_b = (
    + target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy
    + modes[0].energy_levels[1].energy - modes[0].energy_levels[0].energy
    + detuning_0
)
omega_r = (
    + target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy
    - (modes[0].energy_levels[1].energy - modes[0].energy_levels[0].energy)
    - detuning_0
)

hamiltonian = MS_hamiltonian(basis, modes, etas, rabi_rate, omega_b, omega_r, sparse=sparse, mod=amp_mod)

def main():
    import time
    from matplotlib import pyplot as plt

    start = time.perf_counter()
    ic(hamiltonian.hamiltonian_function(0))
    end = time.perf_counter()
    ic(f'Building Hamiltonian took {end - start} s.')

    duration = tau
    coefs = np.zeros(len(basis.states))
    coefs[0] = 1
    ic(len(spin_basis.states), len(basis.states))
    ic(len(spin_basis.states)*fock_dimension, len(basis.states))
    zero_zero = ism.State.from_coefficients(basis, list(coefs))

    times = np.linspace(0, duration, 41) # setting to None will return only the final spin state

    compute_state_fidelity = True
    compute_gate = False

    if compute_state_fidelity:

        start = time.perf_counter()
        psis = zero_zero.propagate_using_schrodinger_equation(hamiltonian, duration, times)
        end = time.perf_counter()
        ic(f'Propagating state took {end - start} s.')

        basis_xx = ism.XPauliAndFockBasis([*spins, *modes])
        psis_xx = [ism.State.from_state(basis_xx, psi) for psi in psis]

        alphas = np.array([psi.compute_coherent_displacements(spins, modes[0]) for psi in psis_xx])
        spin_basis_xx = ism.XPauliBasis(spins)

        for i,vector in enumerate(spin_basis_xx.vectors):
            plt.plot(times, alphas[:, i].real, label=i)
        plt.ylabel(r'Re[$\alpha$]')
        plt.xlabel('Gate Duration (s)')
        plt.legend()
        plt.show()

        for i,vector in enumerate(spin_basis_xx.vectors):
            plt.plot(times, alphas[:, i].imag, label=i)
        plt.ylabel(r'Im[$\alpha$]')
        plt.xlabel('Gate Duration (s)')
        plt.legend()
        plt.show()

        for i,vector in enumerate(spin_basis_xx.vectors):
            plt.plot(alphas[:, i].real, alphas[:, i].imag, label=i)
        plt.ylabel(r'Im[$\alpha$]')
        plt.xlabel(r'Re[$\alpha$]')
        plt.legend()
        plt.show()

        for mode in modes:
            psis = [psi.trace_out_degree_of_freedom(mode) for psi in psis]
        new_basis = psis[0].basis

        probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
        ic(probs[-1,:])

        target_psi = ism.State.from_wavefunction(new_basis, target_wavefunction)
        fidelity = psis[-1].compute_state_fidelity(target_psi.density_matrix)
        ic(fidelity)

        for i,state in enumerate(new_basis.states):
            plt.plot(times, probs[:, i], label=state.name)
        plt.ylabel('Probabilities')
        plt.xlabel('Gate Duration (s)')
        plt.legend()
        plt.show()

    if compute_gate:

        initial_motional_wavefunctions = [np.eye(fock_dimension)[0] for _ in modes]
        start = time.perf_counter()
        ms_gate = ism.Gate.from_hamiltonian(basis, hamiltonian, tau, modes, initial_motional_wavefunctions)
        end = time.perf_counter()
        ic(f'Simulating MS gate took {end - start} s.')


if __name__ == '__main__':
    main()
