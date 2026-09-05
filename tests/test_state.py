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

from ionsim.state import State
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis, XPauliBasis
from ionsim.named_operators import Pauli
from ionsim.testing import assert_array_close

class TestState(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])
        self.basis_x = XPauliBasis([self.spin_a, self.spin_b])
        self.eig_x = np.array([[1, 0], [0, -1]])
        self.state = State.from_density_matrix(self.basis_x, np.kron(self.eig_x, Pauli.I))

    def test_density_matrix_in_new_basis(self):
        """Test the density matrix in the new basis."""
        rho_p = State.from_state(self.basis, self.state).density_matrix
        expected_rho_p = np.kron(Pauli.X, Pauli.I)

        # Check that the density matrix matches the expected value
        assert_array_close(rho_p, expected_rho_p)

if __name__ == '__main__':
    unittest.main()
