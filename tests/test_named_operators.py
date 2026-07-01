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
from ionsim.named_operators import Pauli, Unitary
from ionsim.testing import assert_array_close

class TestNamedOperators(unittest.TestCase):

    def test_unitary_y(self):
        """Test that Unitary.Y is equal to Pauli.Y."""
        assert_array_close(Unitary.Y, Pauli.Y)

    def test_expm_equivalence(self):
        """Test the equivalence of the matrix exponential and the expected result."""
        phi, theta = np.pi / 8, -2 * np.pi / 3
        sig_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
        expected_result = (
            np.cos(theta / 2) * np.kron(Pauli.I, Pauli.I) - 
            1j * np.sin(theta / 2) * np.kron(sig_phi, sig_phi)
        )
        
        result = expm(-1j * theta / 2 * np.kron(sig_phi, sig_phi))
        assert_array_close(result, expected_result)

if __name__ == '__main__':
    unittest.main()
