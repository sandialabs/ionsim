import numpy as np
import warnings
import functools
import scipy.sparse

# keep a reference to the original numpy.kron
_orig_kron = np.kron

@functools.wraps(_orig_kron)
def patched_kron(a, b, *args, **kwargs):
    if scipy.sparse.issparse(a) or scipy.sparse.issparse(b):
        warnings.warn(
            "Calling numpy.kron on scipy.sparse matrices is known to be problematic. "
            "Use scipy.sparse.kron instead.",
            category=UserWarning,
            stacklevel=2
        )
    return _orig_kron(a, b, *args, **kwargs)

np.kron = patched_kron

from .basis import StandardBasis, XPauliBasis, XPauliAndFockBasis
from .degree_of_freedom import AtomicStructure, MotionalMode
from .hamiltonian import Hamiltonian
from .operator import Operator, EnergyShiftOperator, CouplingOperator, GeneralOperator
from .state import State
from .named_operators import Pauli, Fock, Unitary
from .process import Gate, Circuit
from .noise import Noise
from .zeeman_solver import ZeemanHyperfineSolver
from .composite_operator import CompositeOperator
from .dissipator import Dissipator, DissipatorSpontaneousEmission, Lindbladian 
