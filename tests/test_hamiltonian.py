import unittest
import numpy as np
from ionsim.basis import StandardBasis
from ionsim.coupling import CouplingOperator
from ionsim.named_operators import Pauli
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.hamiltonian import Hamiltonian

class TestHamiltonian(unittest.TestCase):

    def setUp(self):
        # Set up the basis and coupling operators for the tests
        spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([spin_a, spin_b])  # 00, 01, 10, 11 

        rabi_rate = 100e3 * 2 * np.pi  # rad./s
        omega = spin_a.energy_levels[1].energy - spin_a.energy_levels[0].energy
        static_operator = rabi_rate / 2 * Pauli.plus

        operator_a = CouplingOperator.from_matrix(self.basis, static_operator, omega, [spin_a])
        operator_b = CouplingOperator.from_matrix(self.basis, static_operator, omega, [spin_b])
        
        interaction_frame_energies = [-1 * state.energy for state in self.basis.states]
        self.hamiltonian = Hamiltonian(self.basis, [operator_a, operator_b], interaction_frame_energies, sparse=False)

    def test_hamiltonian_function(self):
        """Test the Hamiltonian function at time t=0."""
        H0 = self.hamiltonian.hamiltonian_function(0)
        expected_shape = (self.hamiltonian.size, self.hamiltonian.size)
        self.assertEqual(H0.shape, expected_shape)
        self.assertTrue(np.all(np.isfinite(H0)))

    def test_evolve_wavefunction(self):
        """Test the evolution of a wavefunction."""
        duration = np.pi / (100e3 * 2 * np.pi)  # duration based on rabi_rate
        wavefunction = np.array([1, 0, 0, 0])
        times = np.linspace(0, duration, 21)

        ts, ys = self.hamiltonian.evolve_wavefunction(wavefunction, duration, times, ode_solver='odeintz')
        
        self.assertEqual(len(ts), len(ys))
        self.assertEqual(len(ys[0]), len(wavefunction))  # Check that the output wavefunction has the correct size

        # Check that the final state is close to expected values (based on the output from the main function)
        expected_final_states = [
            np.array([1.00000000e+00 + 0.j, 0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j]),
            np.array([9.93844166e-01 + 0.j, 0.00000000e+00 - 0.07821724j, 0.00000000e+00 - 0.07821724j, -6.15583177e-03 + 0.j]),
            np.array([9.75528250e-01 + 0.j, 0.00000000e+00 - 0.15450852j, 0.00000000e+00 - 0.15450852j, -2.44717500e-02 + 0.j]),
            np.array([9.45503252e-01 + 0.j, 0.00000000e+00 - 0.22699526j, 0.00000000e+00 - 0.22699526j, -5.44967520e-02 + 0.j]),
            np.array([9.04508497e-01 + 0.j, 0.00000000e+00 - 0.29389263j, 0.00000000e+00 - 0.29389263j, -9.54914833e-02 + 0.j]),
            np.array([8.53553391e-01 + 0.j, 0.00000000e+00 - 0.35355339j, 0.00000000e+00 - 0.35355339j, -1.46446630e-01 + 0.j]),
            np.array([7.93892610e-01 + 0.j, 0.00000000e+00 - 0.40450849j, 0.00000000e+00 - 0.40450849j, -2.0610739e-01 + 0.j]),
            np.array([7.26995250e-01 + 0.j, 0.00000000e+00 - 0.44550325j, 0.00000000e+00 - 0.44550325j, -2.73004763e-01 + 0.j]),
            np.array([6.54508497e-01 + 0.j, 0.00000000e+00 - 0.47552824j, 0.00000000e+00 - 0.47552824j, -3.45491530e-01 + 0.j]),
            np.array([5.78217232e-01 + 0.j, 0.00000000e+00 - 0.49384416j, 0.00000000e+00 - 0.49384416j, -4.21782773e-01 + 0.j]),
            np.array([5.00000001e-01 + 0.j, 0.00000000e+00 - 0.49999998j, 0.00000000e+00 - 0.49999998j, -4.99999990e-01 + 0.j]),
            np.array([4.21782778e-01 + 0.j, 0.00000000e+00 - 0.49384416j, 0.00000000e+00 - 0.49384416j, -5.78217222e-01 + 0.j]),
            np.array([3.45491530e-01 + 0.j, 0.00000000e+00 - 0.47552825j, 0.00000000e+00 - 0.47552825j, -6.54508447e-01 + 0.j]),
            np.array([2.73004793e-01 + 0.j, 0.00000000e+00 - 0.44550326j, 0.00000000e+00 - 0.44550326j, -7.26995207e-01 + 0.j]),
            np.array([2.0610742e-01 + 0.j, 0.00000000e+00 - 0.40450850j, 0.00000000e+00 - 0.40450850j, -7.93892612e-01 + 0.j]),
            np.array([1.46446666e-01 + 0.j, 0.00000000e+00 - 0.35355340j, 0.00000000e+00 - 0.35355340j, -8.53553334e-01 + 0.j]),
            np.array([9.54914551e-02 + 0.j, 0.00000000e+00 - 0.29389265j, 0.00000000e+00 - 0.29389265j, -9.04508445e-01 + 0.j]),
            np.array([5.44967520e-02 + 0.j, 0.00000000e+00 - 0.22699529j, 0.00000000e+00 - 0.22699529j, -9.45503252e-01 + 0.j]),
            np.array([2.44717500e-02 + 0.j, 0.00000000e+00 - 0.15450854j, 0.00000000e+00 - 0.15450854j, -9.75528250e-01 + 0.j]),
            np.array([6.15583177e-03 + 0.j, 0.00000000e+00 - 0.07821728j, 0.00000000e+00 - 0.07821728j, -9.93844166e-01 + 0.j]),
            np.array([1.57101287e-08 + 0.j, 0.00000000e+00 - 5.00220575e-08j, 0.00000000e+00 - 5.00220575e-08j, -9.99999984e-01 + 0.j])
        ]

        for i, (expected, actual) in enumerate(zip(expected_final_states, ys)):
            with self.subTest(i=i, expected=expected, actual=actual):
                np.testing.assert_array_almost_equal(actual, expected, decimal=5)

    def test_time_steps(self):
        """Test the time steps generated during wavefunction evolution."""
        duration = np.pi / (100e3 * 2 * np.pi)  # duration based on rabi_rate
        wavefunction = np.array([1, 0, 0, 0])
        times = np.linspace(0, duration, 21)

        ts, ys = self.hamiltonian.evolve_wavefunction(wavefunction, duration, times, ode_solver='odeintz')

        # Check that the time steps are correct
        expected_time_steps = np.linspace(0, duration, 21)
        np.testing.assert_array_almost_equal(ts, expected_time_steps)

if __name__ == '__main__':
    unittest.main()
