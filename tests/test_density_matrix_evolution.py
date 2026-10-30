#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Density-matrix evolution of the Lindblad master equation, and the ODE solver layer. """

import unittest

import numpy as np

from ionsim.basis import StandardBasis
from ionsim.custom_math import solve_ode, solve_time_evolution_equation
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.hamiltonian import Hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.lindbladian import Dissipator, DissipatorSpontaneousEmission, Lindbladian
from ionsim.operator import Coupling, CouplingOperator, EnergyShift, EnergyShiftOperator, GeneralOperator
from ionsim.process import Gate
from ionsim.state import State

TPI = 2. * np.pi
ACCURATE = {'ode_solver': 'solve_ivp', 'method': 'DOP853', 'rtol': 1e-10, 'atol': 1e-12}

FULL_LEVELS = ('S1/2,0,0', 'S1/2,1,0', 'P1/2,1,0', 'P1/2,1,1')
SMALL_LEVELS = ('S1/2,0,0', 'P1/2,1,1')

def build_model(sparse: bool = False, levels: tuple = FULL_LEVELS, fock_dimension: int = 3):
    """ 171Yb+ levels x a motional mode (2 MHz), in the interaction picture (frame -E).

        Hamiltonian: drives S -> P detuned by a few MHz (one amplitude-modulated) and a modulated energy shift.
        Dissipator: spontaneous emission (one operator per path: a single rate each), a GeneralOperator with a diagonal
        element and motional couplings |n+1> -> |n> (frame rates of MHz, so it is time-dependent in the frame), and a
        pure-dephasing EnergyShiftOperator. All frame rates are MHz-scale, so the ODE is not stiff. """
    atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=list(levels))
    ground = [l for l in atom.energy_levels if l.term_symbol == 'S1/2']
    excited = [l for l in atom.energy_levels if l.term_symbol == 'P1/2']
    if fock_dimension < 2:
        raise ValueError("The test model needs a motional mode for its time-dependent Lindblad operator.")
    mode = MotionalMode.from_frequency(TPI * 2e6, fock_dimension)
    basis = StandardBasis([atom, mode])
    n = len(basis.states)
    frame = [-state.energy for state in basis.states]

    def drive(g, e, rabi, detuning, modulation=None):
        matrix = np.zeros((len(levels), len(levels)), dtype=complex)
        i, j = atom.energy_levels.index(e), atom.energy_levels.index(g)
        matrix[i, j], matrix[j, i] = rabi / 2., np.conj(rabi) / 2.
        # stored element |g><e| (g first): rate -omega, so that the total rate in the frame is -(omega - omega_0)
        return CouplingOperator.from_matrix(basis, matrix, -(e.energy - g.energy + detuning), current_dofs=[atom], modulation_function=modulation)

    operators = [drive(ground[0], excited[-1], TPI * 4e6, TPI * 3e6, modulation=lambda t: 0.7 + 0.3 * np.cos(TPI * 1e6 * t)),
                 EnergyShiftOperator.from_vector(basis, np.linspace(0., TPI * 0.5e6, n), modulation_function=lambda t: np.sin(TPI * 2e6 * t))]
    if len(ground) > 1:
        operators.append(drive(ground[1], excited[-1], TPI * 3e6 * np.exp(0.4j), TPI * 2.5e6))
    hamiltonian = Hamiltonian(basis, operators, frame, sparse)

    emission = DissipatorSpontaneousEmission.from_atomic_structure_data(basis, ground, excited, frame)
    states = basis.states
    general = GeneralOperator(basis, [EnergyShift(row_state=states[0], column_state=states[0], strength=np.sqrt(TPI * 0.2e6)),
                                      Coupling(row_state=states[0], column_state=states[1],
                                               strength=np.sqrt(TPI * 0.3e6) * np.exp(0.2j), oscillation_rate=0.),
                                      Coupling(row_state=states[n - 2], column_state=states[n - 1],
                                               strength=np.sqrt(TPI * 0.1e6), oscillation_rate=0.)])
    dephasing = EnergyShiftOperator.from_vector(basis, np.sqrt(TPI * 0.05e6) * np.linspace(-1., 1., n))
    dissipator = Dissipator(basis, list(emission.operators) + [general, dephasing], frame, sparse)
    return basis, hamiltonian, dissipator


def mixed_density_matrix(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


class TestDensityMatrixEvolution(unittest.TestCase):

    TIMES = np.linspace(0., 0.25e-6, 5)

    def compare(self, lindbladian: Lindbladian, basis: StandardBasis, rho0: np.ndarray, atol: float = 1e-8):
        _, supervectors = lindbladian.evolve_supervector(basis.compute_supervector_from_density_matrix(rho0), self.TIMES[-1],
                                                         self.TIMES, **ACCURATE)
        times, rhos = lindbladian.evolve_density_matrix(rho0, self.TIMES[-1], self.TIMES, **ACCURATE)
        np.testing.assert_allclose(times, self.TIMES)
        for supervector, rho in zip(supervectors, rhos):
            np.testing.assert_allclose(rho, basis.compute_density_matrix_from_supervector(supervector), atol=atol)
        return rhos

    def test_matches_supervector_dense_and_sparse(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                basis, hamiltonian, dissipator = build_model(sparse)
                rhos = self.compare(Lindbladian(hamiltonian, dissipator), basis, mixed_density_matrix(len(basis.states)))
                self.assertAlmostEqual(np.trace(rhos[-1]).real, 1., places=8)
                np.testing.assert_allclose(rhos[-1], rhos[-1].conj().T, atol=1e-10)
                self.assertGreater(np.max(np.abs(rhos[-1] - rhos[0])), 1e-2)  # nontrivial dynamics

    def test_hamiltonian_only_and_dissipator_only(self):
        basis, hamiltonian, dissipator = build_model()
        rho0 = mixed_density_matrix(len(basis.states))
        for lindbladian in (Lindbladian(hamiltonian, None), Lindbladian(None, dissipator)):
            with self.subTest(hamiltonian=lindbladian.hamiltonian is not None):
                self.compare(lindbladian, basis, rho0)

    def test_non_hermitian_initial_operator(self):
        """ |i><j| inputs (used to build process matrices) evolve linearly; no Hermiticity is assumed. """
        basis, hamiltonian, dissipator = build_model()
        n = len(basis.states)
        rho0 = np.zeros((n, n), dtype=complex); rho0[0, 1] = 1.
        self.compare(Lindbladian(hamiltonian, dissipator), basis, rho0)

    def test_operator_classification(self):
        """ Single-rate Lindblad operators are precomputed as static; operators mixing rates in the frame are not. """
        _, hamiltonian, dissipator = build_model()
        parts = dissipator.lindblad_operator_parts
        self.assertTrue(all(p.common_rate is not None for p in parts[:-2]))  # emission paths
        self.assertIsNone(parts[-2].common_rate)                             # GeneralOperator with several rates
        self.assertEqual(parts[-1].common_rate, 0.)                          # diagonal dephasing
        # the cached parts reproduce the Lindblad matrix functions at any time
        for t in (0., 1.7e-7):
            for p, L_of_t in zip(parts, dissipator.lindblad_matrix_functions):
                np.testing.assert_allclose(p.matrix(t), L_of_t(t))

    def test_state_propagation_representations(self):
        basis, hamiltonian, dissipator = build_model(levels=SMALL_LEVELS, fock_dimension=2)
        lindbladian = Lindbladian(hamiltonian, dissipator)
        state = State.from_density_matrix(basis, mixed_density_matrix(len(basis.states)))
        final = {r: state.propagate_using_master_equation(lindbladian, 0.15e-6, representation=r, **ACCURATE)
                 for r in ('supervector', 'density_matrix')}
        np.testing.assert_allclose(final['density_matrix'].density_matrix, final['supervector'].density_matrix, atol=1e-8)
        series = state.propagate_using_master_equation(lindbladian, 0.15e-6, np.linspace(0., 0.15e-6, 4), representation='density_matrix', **ACCURATE)
        self.assertEqual(len(series), 4)
        np.testing.assert_allclose(series[-1].density_matrix, final['supervector'].density_matrix, atol=1e-8)
        with self.assertRaises(IonSimError):
            state.propagate_using_master_equation(lindbladian, 0.15e-6, representation='matrix')

    def test_gate_from_lindbladian(self):
        """ The representation option passes through Gate.from_lindbladian (which evolves every |i><j|). """
        basis, hamiltonian, dissipator = build_model(levels=SMALL_LEVELS, fock_dimension=2)
        lindbladian = Lindbladian(hamiltonian, dissipator)
        default = Gate.from_lindbladian(basis, lindbladian, 0.1e-6, **ACCURATE)
        matrix_form = Gate.from_lindbladian(basis, lindbladian, 0.1e-6, representation='density_matrix', **ACCURATE)
        np.testing.assert_allclose(matrix_form.process_matrix, default.process_matrix, atol=1e-8)


class TestSolverLayer(unittest.TestCase):
    """ dy/dt = -i Omega/2 sigma_x y: y(t) = (cos(Omega t/2), -i sin(Omega t/2)). """

    OMEGA = TPI * 1e6
    H = np.array([[0., 0.5], [0.5, 0.]]) * OMEGA

    def exact(self, t):
        return np.array([np.cos(self.OMEGA * t / 2.), -1j * np.sin(self.OMEGA * t / 2.)])

    def test_solvers_accept_options(self):
        times = np.linspace(0., 1e-6, 5)
        for solver, options in (('odeintz', {'rtol': 1e-11, 'atol': 1e-12}), ('solve_ivp', {'method': 'DOP853', 'rtol': 1e-11, 'atol': 1e-12}),
                                ('zvode', {'rtol': 1e-12, 'atol': 1e-14, 'method': 'bdf'})):
            with self.subTest(solver=solver):
                ts, ys = solve_time_evolution_equation(lambda t: self.H, np.array([1., 0.]), times[-1], times, ode_solver=solver, **options)
                np.testing.assert_allclose(ts, times)
                for t, y in zip(ts, ys):
                    np.testing.assert_allclose(y, self.exact(t), atol=1e-8)

    def test_default_times(self):
        for solver, count in (('odeintz', 4), ('zvode', 3)):
            with self.subTest(solver=solver):
                ts, ys = solve_time_evolution_equation(lambda t: self.H, np.array([1., 0.]), 1e-6, ode_solver=solver)
                self.assertEqual(len(ts), count)
                self.assertEqual(len(ys), count)
                self.assertAlmostEqual(ts[-1], 1e-6)
                np.testing.assert_allclose(ys[-1], self.exact(1e-6), atol=1e-5)  # default (loose) solver tolerances

if __name__ == '__main__':
    unittest.main()
