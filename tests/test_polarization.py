#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import inspect
import unittest

import numpy as np

from ionsim.polarization import (Polarization, LinearPolarization, CircularPolarization, SphericalPolarization,
                                 _perpendicular_basis, _spherical_basis_vectors)
from ionsim.atomic_internal_energy_level import LSFineLevel, compute_coupling_amplitude_between_atomic_levels
from ionsim.ionsim_error import IonSimError

X, Y, Z = np.eye(3)


class TestPolarizationComponents(unittest.TestCase):
    """Angular conventions of E1 and E2 couplings.

    For a J = 0 -> J = k transition (no spin or nuclear spin), <k m| T^(k)_q |0 0> is the same for every m = q, so the
    coupling to |k, m> must be proportional to the polarization's rank-k component with q = m.
    """

    def setUp(self):
        # Arbitrary elliptical polarization for an arbitrary propagation direction.
        n_hat = np.array([0.3, -0.5, 0.8]); n_hat /= np.linalg.norm(n_hat)
        e1 = np.cross(n_hat, [1., 0., 0.]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_hat, e1)
        self.n_hat = n_hat
        self.polarization = Polarization(np.cos(0.4) * e1 + np.exp(0.9j) * np.sin(0.4) * e2)
        self.ground = LSFineLevel(n=1, j=0, term_symbol='S0', fine_energy=0., hyperfine_A=0., l=0, s=0, mj=0)

    def assert_couplings_follow_components(self, k, components):
        term = {1: 'P1', 2: 'D2'}[k]
        excited = [LSFineLevel(n=1, j=k, term_symbol=term, fine_energy=1., hyperfine_A=0., l=k, s=0, mj=m) for m in range(-k, k + 1)]
        levels = [self.ground] + excited
        couplings = np.array([compute_coupling_amplitude_between_atomic_levels(self.ground, e, components, levels, k, wavenumber=1.)
                              for e in excited])
        expected = np.array([components[m] for m in range(-k, k + 1)])
        ratio = couplings / expected
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)  # proportional, with the same relative phases

    def test_dipole(self):
        self.assert_couplings_follow_components(1, self.polarization.dipole_components(Z))

    def test_quadrupole(self):
        self.assert_couplings_follow_components(2, self.polarization.quadrupole_components(self.n_hat, Z))

    def test_quadrupole_selection_rule(self):
        """n along x and polarization along z (quantization axis): the E2 tensor has only q = +1, -1 components."""
        components = Polarization(Z).quadrupole_components(X, Z)
        # sum_q |C_q|^2 = |symmetric part of eps (x) n|^2 = 1/2 for perpendicular unit vectors
        self.assertAlmostEqual(sum(abs(components[q])**2 for q in (-1, 1)), 0.5, places=12)
        for q in (-2, 0, 2):
            self.assertAlmostEqual(abs(components[q]), 0., places=12)


class TestTransverseBasis(unittest.TestCase):
    """Transverse unit vectors used for polarizations and for the spherical basis of a quantization axis."""

    def test_standard_spherical_basis_for_z(self):
        """e_{+1} = -(x + iy)/sqrt(2), e_0 = z, e_{-1} = (x - iy)/sqrt(2) (Steck, eq. 7.188)."""
        e = _spherical_basis_vectors(Z)
        np.testing.assert_allclose(e[1], -(X + 1j * Y) / np.sqrt(2), atol=1e-15)
        np.testing.assert_allclose(e[0], Z, atol=1e-15)
        np.testing.assert_allclose(e[-1], (X - 1j * Y) / np.sqrt(2), atol=1e-15)

    def test_polarizations_along_z(self):
        np.testing.assert_allclose(LinearPolarization(angle=0.).resolve(Z).jones_vector, X, atol=1e-15)
        np.testing.assert_allclose(LinearPolarization(angle=np.pi / 2).resolve(Z).jones_vector, Y, atol=1e-15)
        # (x + iy)/sqrt(2) = -e_{+1}: pure sigma+ for a z quantization axis.
        components = CircularPolarization('+').resolve(Z).dipole_components(Z)
        self.assertAlmostEqual(abs(components[1]), 1., places=12)
        self.assertAlmostEqual(abs(components[0]) + abs(components[-1]), 0., places=12)

    def test_right_handed_and_orthonormal(self):
        rng = np.random.default_rng(3)  # arbitrary seed
        for n in [*rng.normal(size=(5, 3)), -Z, X]:
            with self.subTest(n=n):
                n_hat = n / np.linalg.norm(n)
                e1, e2 = _perpendicular_basis(n_hat)
                np.testing.assert_allclose(np.array([e1, e2, n_hat]) @ np.array([e1, e2, n_hat]).T, np.eye(3), atol=1e-12)
                np.testing.assert_allclose(np.cross(e1, e2), n_hat, atol=1e-12)

    def test_reference_axis(self):
        """With a reference axis, angle 0 polarizes along the axis's component perpendicular to propagation."""
        polarization = LinearPolarization(angle=0., ref_axis=np.array([1., 1., 0.])).resolve(Y)
        np.testing.assert_allclose(polarization.jones_vector, X, atol=1e-15)
        with self.assertRaises(IonSimError):
            _perpendicular_basis(Y, ref_axis=Y)


class TestPolarizationSpecs(unittest.TestCase):
    """Explicit polarization vectors and the specs (linear, circular, spherical) resolved against a propagation direction."""

    def test_derived_attributes(self):
        polarization = Polarization(np.array([0., 3., 4.j]))                            # unnormalized input
        np.testing.assert_allclose(polarization.jones_vector, np.array([0., 0.6, 0.8j]))
        self.assertIs(polarization.resolve(X), polarization)                             # an explicit vector resolves to itself
        polarization.jones_vector[0] = 1.                                                # edits a copy only
        np.testing.assert_allclose(polarization.jones_vector, np.array([0., 0.6, 0.8j]))
        with self.assertRaises(AttributeError):                                          # no setter
            polarization.jones_vector = X

    def test_circular_handedness_is_relative_to_propagation(self):
        """'+' is sigma+ (q = +1) about z for light along +z and sigma- (q = -1) for light along -z; ref_axis changes only the
        overall phase. SphericalPolarization inverts dipole_components."""
        for direction, q in ((Z, 1), (-Z, -1)):
            with self.subTest(direction=direction):
                components = CircularPolarization('+').resolve(direction).dipole_components(Z)
                self.assertAlmostEqual(abs(components[q]), 1., places=12)
        a, b = (CircularPolarization('+', ref_axis=axis).resolve(Z).jones_vector for axis in (X, X + Y))
        self.assertAlmostEqual(abs(np.vdot(a, b)), 1., places=12)              # same polarization up to a phase
        self.assertGreater(abs(np.vdot(a, b) - 1.), 0.1)                        # ... which differs
        elliptical = Polarization(np.array([0.3, 0.5 + 0.2j, 0.]))
        round_trip = SphericalPolarization(elliptical.dipole_components(Z), Z).resolve(Z)
        np.testing.assert_allclose(round_trip.jones_vector, elliptical.jones_vector, atol=1e-12)

    def test_quantization_axis_is_required(self):
        """The spherical components depend strongly on the quantization axis, so no method gives it a default value."""
        for function in (Polarization.dipole_components, Polarization.quadrupole_components, Polarization.rank2_spherical_basis,
                         SphericalPolarization.__init__):
            with self.subTest(function=function.__qualname__):
                self.assertIs(inspect.signature(function).parameters['quantization_axis'].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            Polarization(X).dipole_components()


if __name__ == '__main__':
    unittest.main()
