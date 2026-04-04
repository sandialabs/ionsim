from pathlib import Path
import numpy as np
import h5py
import sys
import time
from scipy.sparse import kron as skron
from matplotlib import pyplot as plt
from icecream import ic
from typing import Callable

import ionsim as sm

""" ################ Single qubit GST Example ################## """ 
# Define Hamiltonian model: 
 #def R_hamiltonian(basis, phi, rabi_rate, omega, sparse=False, mod=None):
 #    phase = phi
 #    prefactor = np.exp(1j*phase) * rabi_rate/2  
 #
 #    raise_target_spins = [basis.enlarge_matrix(sm.Pauli.plus, [spin]) for spin in target_spins]
 #    operator = prefactor * raise_target_spins[0]
 #    operators = [
 #        sm.CouplingOperator.from_matrix(basis, operator, omega, modulation_function=mod),
 #    ]
 #    interaction_frame_energies = [-state.energy for state in basis.states] 
 #    return sm.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)

def main():
    # 1. Import GST sequence data 
    fname = './1Q_gst_sequence.gstdata' 

    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)

    print_head = True 
    if print_head:
        # Optional print out of first _ lines to check functionality  
        # Print circuit information: 
        head = 100
        for i, circ in enumerate(parsed_circuits):
            print(f"\n--- Experiment {i} ---")
            print(f"    Unparsed circuit line:  {circ.unparsed_data}")
            print(f"    Prep gates:    {circ.prep_gates}")
            print(f"    Germ gates:    {circ.germ_gates}")
            print(f"    Germ power:    {circ.germ_power}")
            print(f"    Measure gates:    {circ.measurement_gates}")
            print(f"    Measurement outcomes:    {circ.measurement_data.counts}")
            print(f"    Total shots:    {circ.total_counts}")
            print(f"    Circuit depth:    {circ.depth}")
            # Only print the first {head} 
            if i > head:
                break

    # Set up basic 1-qubit (1Q) basis  
    num_spins = 1
    
    spins = [
        sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    
    basis = sm.StandardBasis([*spins])
    target_spins = [spins[0]]


    ################ Define gate models: #################### 
    # Requires a basis to be defined 
    def idle(theta):
        """ Returns d^2 x d^2 process matrix in standard basis for Z-rotation by theta """  
        # Build identity matrix with Z rotation by theta:
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        def superoperator_function(theta):
            I = np.eye(2)
            I[0,0] = np.exp( - 1j * theta ) 
            I[1,1] = np.exp( 1j * theta ) 
            # Promote to a d^2 x d^2 superoperator 
            return self.basis.compute_superoperator_from_unitary_operator(I)
        parameters = {'idle angle': theta}
        return sm.Gate.from_process_matrix_function(basis, superoperator_function, parameters)
    
    
    
    def X_pi2(X_rot, Z_rot):
        """ Returns Gate from the d^2 x d^2 process matrix function in standard basis 
    
            X_pi2 = exp( -i [ (pi/2 + X_rot) X  - i(Z_rot)Z ] )
    
            - X_rot is an additional X_rotation parameter (over/under rotation).
            - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 
    
        """  
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        def superoperator_function(X_rot, Z_rot):
            x_angle = np.pi/2. + X_rot
            Rxpi2 = sm.Unitary.R_bloch([x_angle/2., 0., Z_rot/2.]) 
        
            # Promote to a d^2 x d^2 superoperator 
            return self.basis.compute_superoperator_from_unitary_operator(Rxpi2) # superoperator 
    
        parameters = {'excess X rotation': X_rot, 'excess Z rotation' : Z_rot }
        return sm.Gate.from_process_matrix_function(basis, superoperator_function, parameters)
    

    def Y_pi2(Y_rot, Z_rot):
        """ Returns Gate from the d^2 x d^2 process matrix function in standard basis 
    
            X_pi2 = exp( -i [ (pi/2 + X_rot) X  - i(Z_rot)Z ] )
    
            - X_rot is an additional X_rotation parameter (over/under rotation).
            - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 
    
        """  
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        def superoperator_function(Y_rot, Z_rot):
            y_angle = np.pi/2. + Y_rot
            Rypi2 = sm.Unitary.R_bloch([0., y_angle/2., Z_rot/2.]) 
    
            # Promote to a d^2 x d^2 superoperator 
            return self.basis.compute_superoperator_from_unitary_operator(Rypi2)
    
        parameters = {'excess Y rotation': Y_rot, 'excess Z rotation' : Z_rot }
        return sm.Gate.from_process_matrix_function(basis, superoperator_function, parameters)


    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    ism_gate_dictionary['Gxpi2']  = X_pi2 
    ism_gate_dictionary['Gypi2'] = Y_pi2
    ism_gate_dictionary['[]'] = idle
    ism_gate_dictionary['{}'] = None
    #ism_gate_dictionary['Gypi'] = Y_pi2
    # TODO: add 2Q gates 

    def gate_factory_function(gate_name: str, qubits: tuple[int, ...]) -> Callable:
        """ Function to map a gate name & qubit arguments to a gate function """ 
        # TODO: Generalize to 2Q gates 
        #   - for 1Q gates, this is made trivial by the dictionary. For 2Q, it requires functionality 
        assert len(qubits) == 1
        return ism_gate_dictionary[gate_name]

    # For GST, define rho prep state and POVM effects: 
    rho_prep = sm.State.from_coefficients(basis, list([1., 0.]))

    POVM_effects = {} 
    POVM_effects['0'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_0) 
    POVM_effects['1'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_1) 

    GST_analyzer = sm.GateSetTomography(basis, rho_prep, POVM_effects, parsed_circuits, gate_factory_function)
    sys.exit(0)




 #    rabi_rate = 100e3 * 2*np.pi # rad./s
 #    detuning = 0
 #    
 #    omega = (
 #        + target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy
 #        + detuning
 #    )
 #    
 #    amp_mod = None
 #    

#     def simulated_R(phi, theta, domega):
#         """ Builds R(phi, theta) Hamiltonian for a frequency change omega + domega, returns gate """ 
#         tau = abs(theta)/rabi_rate
#         hamiltonian = R_hamiltonian(basis, phi, rabi_rate, omega + domega, sparse=sparse, mod=amp_mod)
#         start = time.perf_counter()
#         ic(hamiltonian.hamiltonian_function(0))
#         end = time.perf_counter()
#         ic(f'Building Hamiltonian took {end - start} s.')
#         return sm.Gate.from_hamiltonian(basis, hamiltonian, tau)
# 
#     def R(phi, theta, domega, half_box_width):
#         """ Builds a process matrix function, then a gate by adding optional noise to it """ 
#         def process_matrix_function(domega):
#             gate = simulated_R(phi, theta, domega) # builds Hamiltonian and returns gate 
#             return gate.process_matrix
#         if half_box_width == 0:
#             omega_noise = None
#         else:
#             domegas = np.linspace(-half_box_width, half_box_width, 21)
#             omega_noise = sm.Noise.from_named_pdf('domega', 'box', {'half_width': half_box_width}, domegas)
#         return sm.Gate.from_process_matrix_function(
#                 basis, process_matrix_function, {'domega': domega}, omega_noise,
#             )
# 
#     def ideal_R(phi, theta):
#         return sm.Gate.from_unitary(basis, sm.Unitary.R(phi, theta), target_spins)
# 
#     def process_fidelity(phi, theta, dx, dy):
#         return R(phi, theta, dx, dy).compute_process_fidelity(ideal_R(phi, theta).process_matrix)

    compute_interpolated_gate = False 

    data_directory = Path.home() / "tmp" / "ionsim_examples_data"
    if not data_directory.exists():
        data_directory.mkdir(parents=True, exist_ok=True)

    data_filename = data_directory / "simr.hdf5"

    # Step 1: Set up a grid where you actually build the gates. 
    if compute_interpolated_gate:

        phi = 0
        theta = np.pi/2

        # Computing gate on a grid where x is a frequency offset from resonance 
        #  and y is a noise width. 
        #  Ex] So y = 0 corresponds to no noise. 
        #  Ex] x = 0 corresponds to being on resonance with some noise (unless y=0). 
        domegas = np.linspace(-50 * 2*np.pi*1e3, 50 * 2*np.pi*1e3, 5) 
        half_box_widths = np.linspace(0, 50 * 2*np.pi*1e3, 3) 

        dxs = domegas
        dys = half_box_widths

        gate_name = 'sqrtX'
        dx_name = 'domega'
        dy_name = 'half_box_width'

        grid_axes = {dx_name : dxs, dy_name : dys} 

        # Define a gate function to build the gate interpolant. 
        def R_function(domega, half_box_width):
            """ Gate function of the interpolation parameters; returns a Gate object """ 
            return R(phi, theta, domega, half_box_width)

        # 1. Construct the gate interpolant class instance 
        print("Building gate interpolant using gate function")
        R_gate_interpolant = sm.GateInterpolant.from_gate_function(R_function, grid_axes, gate_name) 

        # 2 Build a gate interpolating function (this uses cubic splines): returns Gate evaluated at grid / off-grid parameter values  
        """ Ex] interpolated_R(x = 0.5 * 2π * 1E3, y = 2.) returns an R Gate object at domega = 0.5 * 2π * 1E3, half_box_width 2. """ 
        interpolated_R = R_gate_interpolant.interpolated_gate_function # returns a Gate object at a grid point 

        # Optional: Write interpolant to a file using gate interpolant class 
        R_gate_interpolant.write_to_file(data_filename)

        dxs2 = np.linspace(dxs[0], dxs[-1], (len(dxs)-1)*2 + 1)
        dy = dys[-1]
        ms_gates = []
        for dx in dxs:
            ms_gates.append(R(phi, theta, dx, dy))
        ms_gates2 = []
        for dx in dxs2:
            ms_gates2.append(R(phi, theta, dx, dy))
        fidelities = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates]
        fidelities2 = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates2]
        approx_fids = [
            interpolated_R(dx, dy).compute_process_fidelity(
                ideal_R(phi, theta).process_matrix
            ) for dx in dxs2
        ]

        plt.rcParams.update({'font.size': 16})
        dx_scale = 1/(2*np.pi*1e3)
        plt.plot(dxs * dx_scale, 1-np.array(fidelities), 'o', label='simulation: grid point')
        plt.plot(dxs2 * dx_scale, 1-np.array(approx_fids), '-', label='interpolation')
        plt.plot(dxs2 * dx_scale, 1-np.array(fidelities2), '.', label='simulation: off grid')
        plt.xlabel(f'Frequency Error (kHz)')
        plt.ylabel('Infidelity')
        plt.legend()
        plt.savefig(data_directory / f'infidelity_vs_{dx_name}.pdf', bbox_inches='tight')
        plt.show()

        dx = dxs[-1]
        dys2 = np.linspace(dys[0], dys[-1], (len(dys)-1)*2 + 1)
        ms_gates = []
        for dy in dys:
            ms_gates.append(R(phi, theta, dx, dy))
        ms_gates2 = []
        for dy in dys2:
            ms_gates2.append(R(phi, theta, dx, dy))
        fidelities = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates]
        fidelities2 = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates2]
        approx_fids = [
            interpolated_R(dx, dy).compute_process_fidelity(
                ideal_R(phi, theta).process_matrix
            ) for dy in dys2
        ]

        dy_scale = 1/(2*np.pi*1e3)
        plt.plot(dys * dy_scale, 1-np.array(fidelities), 'o', label='simulation: grid point')
        plt.plot(dys2 * dy_scale, 1-np.array(approx_fids), '-', label='interpolation')
        plt.plot(dys2 * dy_scale, 1-np.array(fidelities2), '.', label='simulation: off grid')
        plt.xlabel(f'Half-Width of Boxed White Noise (kHz)')
        plt.ylabel('Infidelity')
        plt.legend()
        plt.savefig(data_directory / f'infidelity_vs_{dy_name}.pdf', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
