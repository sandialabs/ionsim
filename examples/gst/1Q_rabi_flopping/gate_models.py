from pathlib import Path
import numpy as np
from scipy.sparse import kron as skron
from scipy.special import eval_laguerre as laguerre
import h5py
from ionsim.custom_math import trapz_for_matrix
from icecream import ic

import ionsim as sm

num_spins = 1
spins = [
    sm.AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    for _ in range(num_spins)
]
spin_basis = sm.StandardBasis([*spins])
target_spins = [spins[0]]
rabi_rate = 100e3 * 2*np.pi # rad./s
pi_time = abs(np.pi)/rabi_rate
detuning = 0
eta = 0.1
nbar_max = 100
nmax = 500
omega = (
    + target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy
    + detuning
)
#rabi_duration = 16 * pi_time

sparse = False
amp_mod = None

# Helper functions for building Ionsim hamiltonians, dissipators, etc.  
def R_hamiltonian(basis, phi, rabi_rate, omega, sparse=False, mod=None):

    phase = phi
    prefactor = np.exp(1j*phase) * rabi_rate/2  
    raise_target_spins = [basis.enlarge_matrix(sm.Pauli.plus, [spin]) for spin in target_spins]
    operator = prefactor * raise_target_spins[0]

    operators = [
        sm.CouplingOperator.from_matrix(basis, operator, omega, modulation_function=mod),
    ]
    interaction_frame_energies = [-state.energy for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
    return sm.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)

def energy_shift_hamiltonian(basis, energy_shift, sparse=False, mod=None):

    prefactor = energy_shift/2
    shift_target_spins = [basis.enlarge_matrix(sm.Pauli.Z, [spin]) for spin in target_spins]
    operator = prefactor * shift_target_spins[0]

    operators = [
        sm.EnergyShiftOperator.from_matrix(basis, operator),
    ]
    interaction_frame_energies = [-state.energy for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
    return sm.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)

def spin_dissipator(basis, spin_dephasing_rate, spin_flip_rate, sparse=False):

    ops_on_target_spins = [basis.enlarge_matrix(sm.Pauli.Z, [spin]) for spin in target_spins]
    spin_dephaser = np.sqrt(spin_dephasing_rate) * ops_on_target_spins[0]

    ops_on_target_spins = [basis.enlarge_matrix(sm.Pauli.X, [spin]) for spin in target_spins]
    spin_flipper = np.sqrt(spin_flip_rate) * ops_on_target_spins[0]

    # ic(print(sm.CouplingOperator.from_matrix(basis, spin_flipper, 0).static_matrix/np.sqrt(spin_flip_rate)))

    operators = [
        sm.EnergyShiftOperator.from_matrix(basis, spin_dephaser),
        sm.CouplingOperator.from_matrix(basis, spin_flipper, 0),
    ]
    interaction_frame_energies = [0 for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
    return sm.Dissipator(basis, operators, interaction_frame_energies, sparse=sparse)

def fock_state_population(n, nbar):
    return 1 / (1 + nbar) * (nbar / (1 + nbar))**n 

def debye_waller_factor(nbar):
    return np.exp(-eta**2*nbar)

def debye_waller_factor_from_summation(nbar, nmax):
    return np.sum(np.array([fock_state_population(n, nbar) * laguerre(n, eta**2) for n in range(nmax+1)]))

assert(np.abs(np.sum(np.array([fock_state_population(n, nbar_max) for n in range(nmax+1)])) - 1) < 1e-2)
#ic(debye_waller_factor(nbar_max), debye_waller_factor_from_summation(nbar_max, nmax))
assert(np.abs(debye_waller_factor(nbar_max) - debye_waller_factor_from_summation(nbar_max, nmax)) < 1e-2)




""" Process matrix models for gates/GST """ 
# X_pi/8 gate for Rabi flopping experiments  
#def X_pi_8_co_prop(spin_flip_rate: float, spin_dephasing_rate: float, heating_rate: float, nbar: float): 
def X_pi_8_co_prop(spin_flip_rate: float): 
    """ Process matrix Error model for X_pi/8 (small rotation angle) gate for Rabi flopping

        Returns a d2 x d2 process matrix 

        Error parameters: 
            - spin_flip_rate

            For rabi Coprop, the spin dephasing rate is set to zero, and the heating rate was unused and nbar was 0  
                so I have omitted them as function arguments.  

        - Hamiltonian will be time-independent in the interaction picture 
        - Dissipator will be time-independent in the interaction picture
            - Therefore, the lindbladian will be time-independent (this will save computation cost in building its process matrix.) 
    """
    rotation_angle = np.pi/8.
    phi = 0.
    spin_dephasing_rate = 0.
    ham = R_hamiltonian(basis, phi, rabi_rate, omega, False, mod)
    dissipator = spin_dissipator(basis, spin_dephasing_rate, spin_flip_rate)
    rabi_lindbladian = sm.Lindbladian(ham, dissipator) 

    # rotation angle = pi/8 = Omega*duration
    duration = rotation_angle/rabi_rate 
    gate = sm.Gate.from_lindbladian(basis, rabi_lindbladian, rabi_duration, lindbladian_time_independent=True) # see doc string 
    return gate.process_matrix 



def noisy_idle(energy_shift: float, spin_dephasing_rate: float):
    """ Idle gate that includes a Z-coherent energy shift contribution and spin dephasing 

        - the idle occurs for a set amount of time for it to be a process matrix 
        - GST will vary the total idle time by repeating this noisy idle gate "p" times. 
        - The hamiltonian and dissipator are both time-independent in the chosen frame.  
    """
    idle_duration = 1E-6 # 1 µs -> second  

    delay_lindbladian = sm.Lindbladian(
        hamiltonian=energy_shift_hamiltonian(basis, energy_shift),
        dissipator=spin_dissipator(basis, spin_dephasing_rate, 0), # bit flip rate is zero because laser is off
    )

    idle_gate = sm.Gate.from_lindbladian(basis, delay_lindbladian, idle_duration, lindbladian_time_independent=True) # see doc string 
    return idle_gate.process_matrix 





def rabi(basis,duration, spin_dephasing_rate, spin_flip_rate, heating_rate, plot_probs=False):

    nbar = 0

    pulse_lindbladian = sm.Lindbladian(
        hamiltonian=R_hamiltonian(
            basis,
            phi=0, 
            rabi_rate=debye_waller_factor(nbar)*rabi_rate, 
            omega=omega, 
            sparse=sparse, 
            mod=amp_mod
        ),
        dissipator=spin_dissipator(basis, spin_dephasing_rate, spin_flip_rate, sparse=sparse), 
    )

    coefs = np.zeros(len(basis.states))
    coefs[0] = 1
    initial_state = sm.State.from_coefficients(basis, list(coefs))

    dt = pi_time / 20
    times = np.linspace(0, duration, int(duration/dt) + 1) # setting to None will return only the final spin state

    psis = initial_state.propagate_using_master_equation(pulse_lindbladian, duration, times)

    if plot_probs:
        probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
        for i,state in enumerate(basis.states):
            plt.plot(times, probs[:, i], label=state.name)
        plt.ylabel('Probabilities')
        plt.xlabel('Gate Duration (s)')
        plt.legend()
        plt.show()

    return times, psis

 #def rabi_initial_state_probability_with_spin_dephasing(t, spin_dephasing_rate, nbar=0):
 #    rabi_thermal = rabi_rate * debye_waller_factor(nbar)
 #    rabi_eff = np.sqrt(rabi_thermal**2 - spin_dephasing_rate**2)
 #    return 0.5 + 0.5 * np.exp(-spin_dephasing_rate*t) * (np.cos(rabi_eff*t) + spin_dephasing_rate/rabi_eff*np.sin(rabi_eff*t))
 #
 #def rabi_initial_state_probability_with_spin_flipping(t, spin_flip_rate, nbar=0):
 #    rabi_thermal = rabi_rate * debye_waller_factor(nbar)
 #    return 0.5 + 0.5 * np.exp(-2*spin_flip_rate*t) * np.cos(rabi_thermal*t)

def ramsey_excited_state_probability_with_spin_dephasing(t, spin_dephasing_rate, nbar=0, phi=0):
    debye_waller_correction = np.sin(np.pi/2*debye_waller_factor(nbar))
    return 0.5 + 0.5 * debye_waller_correction * np.exp(-2*spin_dephasing_rate*t) * np.cos(phi)

def ramsey(basis, delay, energy_shift, spin_dephasing_rate, heating_rate, plot_probs=False):

    first_pulse_lindbladian = sm.Lindbladian(
        hamiltonian=R_hamiltonian(basis, phi=0, rabi_rate=rabi_rate, omega=omega, sparse=sparse, mod=amp_mod),
        dissipator=None,
    )
    delay_lindbladian = sm.Lindbladian(
        hamiltonian=energy_shift_hamiltonian(basis, energy_shift),
        dissipator=spin_dissipator(basis, spin_dephasing_rate, 0), # bit flip rate is zero because laser is off
    )
    nbar_final = heating_rate * delay
    second_pulse_lindbladian = sm.Lindbladian(
        hamiltonian=R_hamiltonian(basis, phi=0, rabi_rate=debye_waller_factor(nbar_final)*rabi_rate, omega=omega, sparse=sparse, mod=amp_mod),
        dissipator=None,
    )

    coefs = np.zeros(len(basis.states))
    coefs[0] = 1
    initial_state = sm.State.from_coefficients(basis, list(coefs))

    dt = pi_time/20
    times0 = np.linspace(0, pi_time/2, int(pi_time/2/dt) + 1) # setting to None will return only the final spin state
    times1 = np.linspace(times0[-1], times0[-1] + delay, int(delay/dt) + 1)
    times2 = np.linspace(times1[-1], times1[-1] + pi_time/2, int(pi_time/2/dt) + 1)

    psis0 = initial_state.propagate_using_master_equation(first_pulse_lindbladian, pi_time/2, times0)
    psis1 = psis0[-1].propagate_using_master_equation(delay_lindbladian, delay, times1)
    psis2 = psis1[-1].propagate_using_master_equation(second_pulse_lindbladian, pi_time/2, times2)

    psis = psis0[:-1] + psis1[:-1] + psis2
    times = np.concatenate((times0[:-1], times1[:-1], times2))

    if plot_probs:
        probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
        for i,state in enumerate(basis.states):
            plt.plot(times, probs[:, i], label=state.name)
        plt.ylabel('Probabilities')
        plt.xlabel('Gate Duration (s)')
        plt.legend()
        plt.show()

    return times, psis

def main():
    import time
    from matplotlib import pyplot as plt

    ### Control parameters ###
    rabi_duration = 16 * pi_time
    ramsey_delays = np.linspace(0, 50e-3, 21)

    ### Error parameters ###

    heating_rate = 1e3 # q/s
    spin_flip_rate = 1 / rabi_duration # For counter-prop, Debye-Waller effect would amplify the impact.
    rabi_noise_mean, rabi_noise_width = 0, 0.05 * rabi_rate # Only implement for quasi-static noise model.
    energy_shift = 0 
    co_prop_spin_dephasing_rate = 0
    counter_prop_spin_dephasing_rate = 1 / (2*83.3e-3) # Debye-Waller effect will amplify the impact.

    ### Rabi flopping, co-prop on carrier -> intensity fluctuations (white noise) ###

    times, psis = rabi(spin_basis, rabi_duration, co_prop_spin_dephasing_rate, spin_flip_rate, heating_rate)
    probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
    for i,state in enumerate(spin_basis.states):
        plt.plot(times*1e6, probs[:, i], '.', label=state.name)
    if co_prop_spin_dephasing_rate != 0 and spin_flip_rate == 0:
        plt.plot(times*1e6, [rabi_initial_state_probability_with_spin_dephasing(t, co_prop_spin_dephasing_rate) for t in times], '-', label='analytic')
    if co_prop_spin_dephasing_rate == 0 and spin_flip_rate != 0:
        plt.plot(times*1e6, [rabi_initial_state_probability_with_spin_flipping(t, spin_flip_rate) for t in times], '-', label='analytic')
    plt.title('Rabi Flopping (White Noise)')
    plt.ylabel('Probabilities')
    plt.xlabel('Time ' r'($\mu$s)')
    plt.legend()
    plt.show()
    ic(probs[-1,:])

    ### Rabi flopping, co-prop on carrier -> intensity fluctuations (quasi-static noise) ###

    psis_for_each_shift = {}
    rabi_shifts = np.linspace(-3*rabi_noise_width, 3*rabi_noise_width, 21)
    times = np.linspace(0, rabi_duration, 201)
    for ish, shift in enumerate(rabi_shifts):
        psis = []
        coefs = np.zeros(len(spin_basis.states))
        coefs[0] = 1
        initial_state = sm.State.from_coefficients(spin_basis, list(coefs))
        hamiltonian = R_hamiltonian(spin_basis, phi=0, rabi_rate=rabi_rate + rabi_noise_mean + shift, omega=omega)
        psis = initial_state.propagate_using_schrodinger_equation(hamiltonian, rabi_duration, times)
        psis_for_each_shift[ish] = psis

    rabi_noise = sm.Noise.from_named_pdf('rabi_shift', 'gaussian', {'standard_deviation': rabi_noise_width}, rabi_shifts)
    psis = []
    for it, time in enumerate(times):
        ys = np.array(
            [rabi_noise.probability_density_function(shift) * psis_for_each_shift[ish][it].density_matrix
            for ish, shift in enumerate(rabi_shifts)]
        )
        rho = trapz_for_matrix(ys, rabi_shifts)
        psis.append(sm.State.from_density_matrix(spin_basis, rho))

    probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
    for i,state in enumerate(spin_basis.states):
        plt.plot(times*1e6, probs[:, i], '.', label=state.name)
    plt.title('Rabi Flopping (Quasi-Static Noise)')
    plt.ylabel('Probabilities')
    plt.xlabel('Time ' r'($\mu$s)')
    plt.legend()
    plt.show()
    ic(probs[-1,:])

    ### Ramsey decay, counter-prop on carrier ###

    psis = []
    for delay in ramsey_delays:
        times, psis_at_times = ramsey(spin_basis, delay, energy_shift, counter_prop_spin_dephasing_rate, 0)
        psis.append(psis_at_times[-1])
    probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
    for i,state in enumerate(spin_basis.states):
        plt.plot(ramsey_delays*1e3, probs[:,i], '.', label=f'{state.name}; heating rate = {0} q/s')
    if counter_prop_spin_dephasing_rate != 0:
        plt.plot(
            ramsey_delays*1e3, 
            [ramsey_excited_state_probability_with_spin_dephasing(t, counter_prop_spin_dephasing_rate) for t in ramsey_delays], 
            '-', 
            label='analytic'
         )
    plt.title('Ramsey Decay')
    plt.ylabel('Probabilities')
    plt.xlabel('Delay (ms)')
    plt.legend()
    plt.show()

    for hr, style in [[0, '.'], [heating_rate, '.']]:
        psis = []
        for delay in ramsey_delays:
            times, psis_at_times = ramsey(spin_basis, delay, energy_shift, counter_prop_spin_dephasing_rate, hr)
            psis.append(psis_at_times[-1])
        probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
        contrasts = np.abs(probs[:, 1] - probs[:, 0])
        plt.plot(ramsey_delays*1e3, contrasts, style, label=f'heating rate = {hr} q/s')
        if counter_prop_spin_dephasing_rate != 0:
            plt.plot(
                ramsey_delays*1e3, 
                [2*ramsey_excited_state_probability_with_spin_dephasing(t, counter_prop_spin_dephasing_rate, nbar=hr*t) - 1 for t in ramsey_delays], 
                '-', 
                label=f'analytic; heating rate = {hr} q/s'
             )
    # hot_spin_dephasing_rate = 1 / (2*49.7e-3)
    # plt.plot(
    #     ramsey_delays*1e3, 
    #     [2*ramsey_excited_state_probability_with_spin_dephasing(t, hot_spin_dephasing_rate) - 1 for t in ramsey_delays], 
    #     '-', 
    #     label='analytic; measured rate with heating'
    #  )
    plt.title('Ramsey Decay')
    plt.ylabel('Fringe Contrast')
    plt.xlabel('Delay (ms)')
    plt.legend()
    plt.show()
    ic(probs[-1,:])
 








if __name__ == '__main__':
    main()
