#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest

import numpy as np
from scipy.linalg import expm

from scipy.integrate import solve_ivp

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.hamiltonian import Hamiltonian
from ionsim.operator import CouplingOperator
from ionsim.time_averaging import time_averaged_hamiltonian, group_terms_by_rate
from ionsim.ionsim_error import IonSimError

TPI = 2. * np.pi


def floquet_quasienergies(H0: np.ndarray, terms: list, base_rate: float, steps: int = 4000) -> np.ndarray:
    """ Quasi-energies (sorted, in the first zone around 0) of H(t) = H0 + sum (h e^{-i w t} + h.c.) with every w an
        integer multiple of base_rate, from the one-period propagator (midpoint time stepping). """
    period = TPI / base_rate
    dt = period / steps
    U = np.eye(len(H0), dtype=complex)
    for k in range(steps):
        t = (k + 0.5) * dt
        H = H0.astype(complex).copy()
        for h, w in terms:
            H += h * np.exp(-1j * w * t) + h.conj().T * np.exp(1j * w * t)
        U = expm(-1j * H * dt) @ U
    phases = -np.angle(np.linalg.eigvals(U)) / period
    return np.sort(phases)


class TestTimeAveragedHamiltonian(unittest.TestCase):

    def test_two_level_stark_shift(self):
        """ h = (Omega/2)|e><g| at rate Delta: +Omega^2/(4 Delta) on |g>, -Omega^2/(4 Delta) on |e>, and it matches the
            exact dressed energies of the rotating-frame Hamiltonian -Delta |e><e| + (Omega/2) sigma_x """
        Omega, Delta = TPI * 1e5, TPI * 3e7
        h = np.zeros((2, 2), dtype=complex); h[1, 0] = Omega / 2.
        result = time_averaged_hamiltonian([h], [Delta])
        np.testing.assert_allclose(result.level_shifts, [Omega**2 / (4 * Delta), -Omega**2 / (4 * Delta)], rtol=1e-12)
        exact = np.linalg.eigvalsh(np.array([[0., Omega / 2.], [Omega / 2., -Delta]]))
        self.assertAlmostEqual(exact[1] / result.level_shifts[0], 1., delta=(Omega / Delta)**2)   # ground-state branch
        self.assertAlmostEqual(result.max_perturbation_ratio, Omega / 2. / Delta, places=12)

    def test_negative_rate_is_hermitian_conjugate(self):
        """ (h, -w) and (h^dagger, w) are the same physical term """
        rng = np.random.default_rng(3)
        h = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        a = time_averaged_hamiltonian([h], [-TPI * 1e6]).hamiltonian
        b = time_averaged_hamiltonian([h.conj().T], [TPI * 1e6]).hamiltonian
        np.testing.assert_allclose(a, b, atol=1e-12 * np.abs(a).max())

    def test_equal_rates_are_combined(self):
        """ Two couplings of the same rate interfere: their sum enters the commutator, not each separately """
        h1 = np.zeros((3, 3), dtype=complex); h1[2, 0] = 1.
        h2 = np.zeros((3, 3), dtype=complex); h2[2, 1] = 1.
        w = TPI * 1e6
        combined = time_averaged_hamiltonian([h1, h2], [w, w * (1. + 1e-12)], rate_tolerance=1e-3).hamiltonian
        separate = time_averaged_hamiltonian([h1], [w]).hamiltonian + time_averaged_hamiltonian([h2], [w]).hamiltonian
        self.assertGreater(abs(combined[0, 1]), 0.5 / w)            # the static |0><1| coupling only appears combined
        self.assertEqual(separate[0, 1], 0.)
        matrices, rates = group_terms_by_rate(np.array([h1, h2]), np.array([w, w]), 1e-3)
        self.assertEqual(len(rates), 1)

    def test_resonant_term_is_rejected(self):
        with self.assertRaises(IonSimError):
            time_averaged_hamiltonian([np.eye(2)], [0.])

    def test_against_floquet(self):
        """ Four levels with off-resonant couplings at several commensurate rates (and equal-rate pairs): the static
            effective Hamiltonian reproduces the exact Floquet quasi-energies to second order """
        rng = np.random.default_rng(7)
        base = TPI * 1.
        terms = []
        for w in (3., 5., 5., 8., 11.):
            h = 0.02 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) * base
            terms.append((h, w * base))
        H0 = np.zeros((4, 4))
        result = time_averaged_hamiltonian([h for h, _ in terms], [w for _, w in terms], rate_tolerance=1e-9)
        exact = floquet_quasienergies(H0, terms, base)
        predicted = np.sort(np.linalg.eigvalsh(result.hamiltonian))
        # second-order theory: relative error O(|h| / w) (~0.03 here; measured 1.2e-2)
        self.assertLess(np.abs(exact - predicted).max(), 0.03 * np.abs(predicted).max())
        # and weaker couplings converge (error falls faster than the shift)
        small = [(0.1 * h, w) for h, w in terms]
        result_small = time_averaged_hamiltonian([h for h, _ in small], [w for _, w in small], rate_tolerance=1e-9)
        exact_small = floquet_quasienergies(H0, small, base)
        predicted_small = np.sort(np.linalg.eigvalsh(result_small.hamiltonian))
        self.assertLess(np.abs(exact_small - predicted_small).max() / np.abs(predicted_small).max(),
                        0.15 * np.abs(exact - predicted).max() / np.abs(predicted).max())


class TestHamiltonianTimeAverage(unittest.TestCase):
    """ Hamiltonian.time_average on IonSim Hamiltonians (three S1/2 sublevels of 171Yb+, interaction picture) """

    def setUp(self):
        self.atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'],
                                                 level_names=['S1/2,0,0', 'S1/2,1,0', 'S1/2,1,1'], magnetic_field=5.)
        self.basis = StandardBasis([self.atom])
        self.E = np.array([s.energy for s in self.basis.states])
        self.frame = list(-self.E)                                      # interaction picture: H0 = 0

    def coupling(self, i, j, strength, total_rate):
        """ strength |i><j| + h.c. with i < j, oscillating at total_rate in the interaction picture """
        matrix = np.zeros((3, 3), dtype=complex)
        matrix[i, j], matrix[j, i] = strength, np.conj(strength)
        bare = total_rate - (self.frame[i] - self.frame[j])
        return CouplingOperator.from_matrix(self.basis, matrix, bare)

    def test_stark_shift_and_slow_terms_kept(self):
        """ A fast coupling |0>-|2> (Omega_f / 2 at rate Delta) becomes +-Omega_f^2/(4 Delta) shifts; the slow, resonant
            coupling |0>-|1> is kept unchanged """
        slow, fast, Delta = TPI * 1e4 / 2., TPI * 2e5 / 2., TPI * 5e6
        H = Hamiltonian(self.basis, [self.coupling(0, 1, slow, 0.), self.coupling(0, 2, fast, -Delta)], self.frame)
        averaged, result = H.time_average(TPI * 1e6, return_result=True)
        shift = fast**2 / Delta
        for t in (0., 1.7e-6, 3.1e-5):
            A = np.asarray(averaged.hamiltonian_function(t))
            B = np.asarray(H.hamiltonian_function(t))
            np.testing.assert_allclose(A[0, 1], B[0, 1], rtol=1e-12)      # slow coupling unchanged
            self.assertEqual(A[0, 2], 0.)                                   # fast coupling removed
        A = np.asarray(averaged.hamiltonian_function(0.))
        # h = fast |0><2| at rate -Delta, i.e. h^dagger = fast |2><0| at +Delta: |0> shifted by +fast^2/Delta
        np.testing.assert_allclose(np.real(np.diag(A)), [shift, 0., -shift], rtol=1e-10)
        self.assertAlmostEqual(result.max_perturbation_ratio, fast / Delta, places=12)

    def test_dynamics(self):
        """ Rabi flopping on |0>-|1> under a slow drive, with a fast off-resonant coupling |0>-|2> that Stark shifts |0>:
            the averaged Hamiltonian reproduces the exact |1> population; dropping the fast term instead does not """
        slow, fast, Delta = TPI * 1e4 / 2., TPI * 2e5 / 2., TPI * 5e6
        H = Hamiltonian(self.basis, [self.coupling(0, 1, slow, 0.), self.coupling(0, 2, fast, -Delta)], self.frame)
        averaged = H.time_average(TPI * 1e6)
        dropped = Hamiltonian(self.basis, [self.coupling(0, 1, slow, 0.)], self.frame)
        times = np.linspace(0., 1e-4, 6)
        def populations(hamiltonian):
            f = hamiltonian.hamiltonian_function
            sol = solve_ivp(lambda t, y: -1j * (np.asarray(f(t)) @ y), (0., times[-1]), np.array([1., 0., 0.], dtype=complex),
                            t_eval=times, method='DOP853', rtol=1e-10, atol=1e-12, max_step=2e-8)
            return np.abs(sol.y[1])**2
        exact, avg, drop = populations(H), populations(averaged), populations(dropped)
        np.testing.assert_allclose(avg, exact, atol=1e-3)            # measured 4.8e-4 ~ (fast / Delta)^2 admixture of |2>
        self.assertGreater(np.abs(drop - exact).max(), 0.05)

    def test_no_fast_terms_and_modulated(self):
        H = Hamiltonian(self.basis, [self.coupling(0, 1, TPI * 1e3, 0.)], self.frame)
        averaged = H.time_average(TPI * 1e6)
        np.testing.assert_allclose(np.asarray(averaged.hamiltonian_function(1e-6)), np.asarray(H.hamiltonian_function(1e-6)))
        matrix = np.zeros((3, 3), dtype=complex); matrix[0, 2] = matrix[2, 0] = TPI * 1e3
        modulated = CouplingOperator.from_matrix(self.basis, matrix, -TPI * 5e6 - (self.frame[0] - self.frame[2]),
                                                 modulation_function=lambda t: 1.)
        with self.assertRaises(NotImplementedError):
            Hamiltonian(self.basis, [modulated], self.frame).time_average(TPI * 1e6)


if __name__ == '__main__':
    unittest.main()
