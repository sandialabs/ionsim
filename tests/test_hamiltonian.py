import unittest
import numpy as np
from ionsim.basis import StandardBasis
from ionsim.coupling import CouplingOperator
from ionsim.named_operators import Pauli
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.hamiltonian import Hamiltonian
from ionsim.testing import assert_array_close


class TestTimeIndependentHamiltonian(unittest.TestCase):
    def setUp(self):
        # Single spin system for time-independent Hamiltonian
        spin = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([spin])
        rabi_rate = 100e3 * 2 * np.pi  # rad/s
        static_operator = rabi_rate / 2 * Pauli.X
        omega_drive = spin.energy_levels[1].energy - spin.energy_levels[0].energy
        operator = CouplingOperator.from_matrix(self.basis, static_operator, omega_drive, [spin])
        interaction_frame_energies = [-omega_drive / 2, omega_drive / 2]
        # interaction_frame_energies = [-spin.energy_levels[0].energy, -spin.energy_levels[1].energy]
        self.hamiltonian = Hamiltonian(self.basis, [operator], interaction_frame_energies, sparse=False)

    def test_time_independent_hamiltonian_function(self):
        """Test the time-independent Hamiltonian function at t=0 and t=1."""
        H0 = self.hamiltonian.hamiltonian_function(0)
        H1 = self.hamiltonian.hamiltonian_function(1)
        # Should be identical for time-independent Hamiltonian
        assert_array_close(H0, H1)
        self.assertEqual(H0.shape, (self.hamiltonian.size, self.hamiltonian.size))
        self.assertTrue(np.all(np.isfinite(H0)))
        self.assertTrue(self.hamiltonian.all_rates_are_zero)
        self.assertTrue(self.hamiltonian.all_mods_are_none)

    def test_evolve_wavefunction_time_independent(self):
        """Test wavefunction evolution under a time-independent Hamiltonian."""
        duration = np.pi / (100e3 * 2 * np.pi)  # duration for a pi pulse
        wavefunction = np.array([1, 0])
        times = np.linspace(0, duration, 11)
        ts, ys = self.hamiltonian.evolve_wavefunction(wavefunction, duration, times)
        self.assertEqual(len(ts), len(ys))
        self.assertEqual(len(ys[0]), len(wavefunction))
        # For a pi pulse, final state should be close to [0, -1j]
        expected_final = np.array([0, -1j])
        assert_array_close(ys[-1], expected_final, atol=1e-6)

class TestTimeDependentHamiltonian(unittest.TestCase):

    def setUp(self):
        # Set up the basis and coupling operators for the tests
        spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([spin_a, spin_b])  # 00, 01, 10, 11 

        rabi_rate = 100e3 * 2 * np.pi  # rad./s
        omega_drive = spin_a.energy_levels[1].energy - spin_a.energy_levels[0].energy + 10e3 * 2 * np.pi  # Slightly detuned drive frequency
        static_operator = rabi_rate / 2 * Pauli.plus

        operator_a = CouplingOperator.from_matrix(self.basis, static_operator, omega_drive, [spin_a])
        operator_b = CouplingOperator.from_matrix(self.basis, static_operator, omega_drive, [spin_b])
        
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
            np.array([ 9.938442964907226e-01-3.217725435274174e-05j, -6.155702976021345e-04-7.821402104136019e-02j, -6.155702976021345e-04-7.821402104136019e-02j, -6.154943502553673e-03+9.668639144370442e-05j]),
            np.array([ 0.97553023522733 -0.0002545582128j  , -0.002446973139379-0.15448293705593j , -0.002446973139379-0.15448293705593j , -0.024457657500454+0.000768603117613j]),
            np.array([ 0.945512997998511-0.000843168754005j, -0.005448662581132-0.226909983123464j, -0.005448662581132-0.226909983123464j, -0.054426139538943+0.002566669146715j]),
            np.array([ 0.904538008318397-0.001946229050056j, -0.009545989746868-0.293694007001819j, -0.009545989746868-0.293694007001819j, -0.095271530432415+0.005993975347484j]),
            np.array([ 0.853621519310392-0.003671776224576j, -0.014637058309535-0.353174137849555j, -0.014637058309535-0.353174137849555j, -0.145919373755356+0.011484106152938j]),
            np.array([ 0.794024182822596-0.006076929397755j, -0.02059525626444 -0.403871173439384j, -0.02059525626444 -0.403871173439384j, -0.2050385391256  +0.019381850104437j]),
            np.array([ 0.727218212765111-0.009159390531033j, -0.027272410764693-0.444524321434987j, -0.027272410764693-0.444524321434987j, -0.271077114904056+0.029927194020726j]),
            np.array([ 0.654849231468311-0.012851483902969j, -0.034502467490057-0.47412257479359j , -0.034502467490057-0.47412257479359j , -0.342304052057521+0.043243061067691j]),
            np.array([ 0.578693635706532-0.017017096747651j, -0.042105621573693-0.491929855572053j, -0.042105621573693-0.491929855572053j, -0.416855614167193+0.059327343000315j]),
            np.array([ 0.50061270194152 -0.021451692433165j, -0.049892789736758-0.497503313275987j, -0.049892789736758-0.497503313275987j, -0.492785265725654+0.078049522784883j]),
            np.array([ 0.422504652763805-0.025885479219041j, -0.057670313074281-0.490704343360487j, -0.057670313074281-0.490704343360487j, -0.568115631569592+0.099152053550748j]),
            np.array([ 0.346255982828281-0.029989706266438j, -0.065244771859853-0.47170202717893j , -0.065244771859853-0.47170202717893j , -0.640891075399498+0.122256513184489j]),
            np.array([ 0.273693297229375-0.033385930297492j, -0.07242779631667 -0.440968933808858j, -0.07242779631667 -0.440968933808858j, -0.709229502727679+0.146874370529742j]),
            np.array([ 0.206536962686791-0.035657991235363j, -0.079040751531809-0.399269365591966j, -0.079040751531809-0.399269365591966j, -0.771371943049092+0.172422058181807j]),
            np.array([ 0.14635779404894 -0.036366326490538j, -0.084919179750803-0.347640351418875j, -0.084919179750803-0.347640351418875j, -0.825728560790995+0.198239888367696j]),
            np.array([ 0.094537903658325-0.035064155039658j, -0.089916890089368-0.287365855169059j, -0.089916890089368-0.287365855169059j, -0.870919856818205+0.223614212204894j]),
            np.array([ 0.052236721563972-0.0313149878828j  , -0.093909594480545-0.219944847964465j, -0.093909594480545-0.219944847964465j, -0.905811961698128+0.247802108565765j]),
            np.array([ 0.020363071569738-0.024710860994082j, -0.096797998143541-0.147054022879172j, -0.096797998143541-0.147054022879172j, -0.929545066638888+0.270057800859265j]),
            np.array([-4.460174985321799e-04-0.014890631041963j, -9.851026923098351e-02-0.07050610483035j , -9.851026923098351e-02-0.07050610483035j , -9.415542759268228e-01+0.289659909303738j]),
            np.array([-0.00983923572322 -0.001557673604616j, -0.099003825177341+0.007795233773572j, -0.099003825177341+0.007795233773572j, -0.941582335565059+0.305938636535124j])
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
