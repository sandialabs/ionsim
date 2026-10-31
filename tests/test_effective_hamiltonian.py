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

from ionsim.effective_hamiltonian import effective_hamiltonian, group_terms_by_rate
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


class TestEffectiveHamiltonian(unittest.TestCase):

    def test_two_level_stark_shift(self):
        """ h = (Omega/2)|e><g| at rate Delta: +Omega^2/(4 Delta) on |g>, -Omega^2/(4 Delta) on |e>, and it matches the
            exact dressed energies of the rotating-frame Hamiltonian -Delta |e><e| + (Omega/2) sigma_x """
        Omega, Delta = TPI * 1e5, TPI * 3e7
        h = np.zeros((2, 2), dtype=complex); h[1, 0] = Omega / 2.
        result = effective_hamiltonian([h], [Delta])
        np.testing.assert_allclose(result.level_shifts, [Omega**2 / (4 * Delta), -Omega**2 / (4 * Delta)], rtol=1e-12)
        exact = np.linalg.eigvalsh(np.array([[0., Omega / 2.], [Omega / 2., -Delta]]))
        self.assertAlmostEqual(exact[1] / result.level_shifts[0], 1., delta=(Omega / Delta)**2)   # ground-state branch
        self.assertAlmostEqual(result.max_perturbation_ratio, Omega / 2. / Delta, places=12)

    def test_negative_rate_is_hermitian_conjugate(self):
        """ (h, -w) and (h^dagger, w) are the same physical term """
        rng = np.random.default_rng(3)
        h = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        a = effective_hamiltonian([h], [-TPI * 1e6]).hamiltonian
        b = effective_hamiltonian([h.conj().T], [TPI * 1e6]).hamiltonian
        np.testing.assert_allclose(a, b, atol=1e-12 * np.abs(a).max())

    def test_equal_rates_are_combined(self):
        """ Two couplings of the same rate interfere: their sum enters the commutator, not each separately """
        h1 = np.zeros((3, 3), dtype=complex); h1[2, 0] = 1.
        h2 = np.zeros((3, 3), dtype=complex); h2[2, 1] = 1.
        w = TPI * 1e6
        combined = effective_hamiltonian([h1, h2], [w, w * (1. + 1e-12)], rate_tolerance=1e-3).hamiltonian
        separate = effective_hamiltonian([h1], [w]).hamiltonian + effective_hamiltonian([h2], [w]).hamiltonian
        self.assertGreater(abs(combined[0, 1]), 0.5 / w)            # the static |0><1| coupling only appears combined
        self.assertEqual(separate[0, 1], 0.)
        matrices, rates = group_terms_by_rate(np.array([h1, h2]), np.array([w, w]), 1e-3)
        self.assertEqual(len(rates), 1)

    def test_resonant_term_is_rejected(self):
        with self.assertRaises(IonSimError):
            effective_hamiltonian([np.eye(2)], [0.])

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
        result = effective_hamiltonian([h for h, _ in terms], [w for _, w in terms], rate_tolerance=1e-9)
        exact = floquet_quasienergies(H0, terms, base)
        predicted = np.sort(np.linalg.eigvalsh(result.hamiltonian))
        # second-order theory: relative error O(|h| / w) (~0.03 here; measured 1.2e-2)
        self.assertLess(np.abs(exact - predicted).max(), 0.03 * np.abs(predicted).max())
        # and weaker couplings converge (error falls faster than the shift)
        small = [(0.1 * h, w) for h, w in terms]
        result_small = effective_hamiltonian([h for h, _ in small], [w for _, w in small], rate_tolerance=1e-9)
        exact_small = floquet_quasienergies(H0, small, base)
        predicted_small = np.sort(np.linalg.eigvalsh(result_small.hamiltonian))
        self.assertLess(np.abs(exact_small - predicted_small).max() / np.abs(predicted_small).max(),
                        0.15 * np.abs(exact - predicted).max() / np.abs(predicted).max())


if __name__ == '__main__':
    unittest.main()
