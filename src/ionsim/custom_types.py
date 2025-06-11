import sys
from typing import Any, TypeAlias
from numpy.typing import NDArray
from scipy.sparse import spmatrix

if sys.version_info >= (3, 10):
    Vector: TypeAlias = NDArray[Any]
    Matrix: TypeAlias = NDArray[Any]

    SparseVector: TypeAlias = spmatrix
    SparseMatrix: TypeAlias = spmatrix

    AnyVector: TypeAlias = Vector | SparseVector
    AnyMatrix: TypeAlias = Matrix | SparseMatrix
else:
    Vector = NDArray[Any]
    Matrix = NDArray[Any]

    SparseVector = spmatrix
    SparseMatrix = spmatrix

    AnyVector = Vector | SparseVector
    AnyMatrix = Matrix | SparseMatrix
