"""Functions, constants, and classes common to multiple tests."""
import numpy as np


# Default relative tolerance that we expect from most tests
DEFAULT_RTOL = 1e-7


# Default absolute tolerance from tests using it
DEFAULT_ATOL = 1e-14


def assert_array_close(actual, expected, rtol=None, atol=None, **kwargs):
    """Assert the first two arguments, which may be numpy arrays, are
    close to within some tolerance. Arguments are mostly interpreted
    the same as numpy.testing.assert_allclose.
    """

    rtol = rtol if rtol is not None else DEFAULT_RTOL
    atol = atol if atol is not None else DEFAULT_ATOL
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)
