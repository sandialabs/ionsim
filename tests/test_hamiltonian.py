import unittest
import numpy as np
from ionsim.basis import StandardBasis
from ionsim.operator import CouplingOperator
from ionsim.named_operators import Pauli
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.hamiltonian import Hamiltonian
from ionsim.testing import assert_array_close

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

        # Check that the final state is close to expected values (generated from a version of the code believed to be correct)
        expected_final_states = [
            np.array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]),
            np.array([0.993844170389312+0.j, 0.-0.07821724320603j, 0.-0.07821724320603j, -0.006155829610687+0.j]),
            np.array([0.975528252087859+0.j, 0.-0.154508522267863j,0.-0.154508522267863j,-0.024471747912141+0.j]),
            np.array([0.945503251341548+0.j, 0.-0.226995257659482j, 0.-0.226995257659482j, -0.054496748658451+0.j]),
            np.array([ 0.904508480846667+0.j, 0.-0.293892629188798j, 0.-0.293892629188798j, -0.095491519153333+0.j]),
            np.array([ 0.853553366733968+0.j, 0.-0.353553385120494j, 0.-0.353553385120494j, -0.146446633266032+0.j]),
            np.array([ 0.79389260933487 +0.j, 0.-0.404508487237185j, 0.-0.404508487237185j, -0.206107390665129+0.j]),
            np.array([ 0.726995240291916+0.j, 0.-0.445503250981497j, 0.-0.445503250981497j, -0.273004759708083+0.j]),
            np.array([ 0.654508492796428+0.j, 0.-0.475528244447519j, 0.-0.475528244447519j, -0.345491507203571+0.j]),
            np.array([ 0.578217232903036+0.j, 0.-0.493844155114556j, 0.-0.493844155114556j, -0.421782767096963+0.j]),
            np.array([ 0.500000006148127+0.j, 0.-0.499999984321081j, 0.-0.499999984321081j, -0.499999993851873+0.j]),
            np.array([ 0.421782784106066+0.j, 0.-0.493844155013673j, 0.-0.493844155013673j, -0.578217215893933+0.j]),
            np.array([ 0.345491529936832+0.j, 0.-0.475528246108701j, 0.-0.475528246108701j, -0.654508470063168+0.j]),
            np.array([ 0.273004786746149+0.j, 0.-0.445503256202241j, 0.-0.445503256202241j, -0.72699521325385 +0.j]),
            np.array([ 0.206107417808794+0.j, 0.-0.404508499654357j, 0.-0.404508499654357j, -0.793892582191205+0.j]),
            np.array([ 0.146446658536204+0.j, 0.-0.353553403519748j, 0.-0.353553403519748j, -0.853553341463795+0.j]),
            np.array([ 0.095491554957102+0.j, 0.-0.293892651275968j, 0.-0.293892651275968j, -0.904508445042897+0.j]),
            np.array([ 0.054496787855274+0.j, 0.-0.226995286734186j, 0.-0.226995286734186j, -0.945503212144725+0.j]),
            np.array([ 0.024471780303071+0.j, 0.-0.154508539917327j, 0.-0.154508539917327j, -0.975528219696929+0.j]),
np.array([ 0.006155855921655+0.j, 0.-0.078217279132654j, 0.-0.078217279132654j, -0.993844144078344+0.j]),

np.array([ 0.000000015710129+0.j, 0.-0.000000050022058j, 0.-0.000000050022058j, -0.999999984289871+0.j]),
        ]

        for i, (expected, actual) in enumerate(zip(expected_final_states, ys)):
            with self.subTest(i=i, expected=expected, actual=actual):
                assert_array_close(actual, expected)

    def test_time_steps(self):
        """Test the time steps generated during wavefunction evolution."""
        duration = np.pi / (100e3 * 2 * np.pi)  # duration based on rabi_rate
        wavefunction = np.array([1, 0, 0, 0])
        times = np.linspace(0, duration, 21)

        ts, ys = self.hamiltonian.evolve_wavefunction(wavefunction, duration, times, ode_solver='odeintz')

        # Check that the time steps are correct
        expected_time_steps = np.linspace(0, duration, 21)
        assert_array_close(ts, expected_time_steps)

if __name__ == '__main__':
    unittest.main()
