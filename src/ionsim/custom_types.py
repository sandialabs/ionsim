from typing import Any
from numpy.typing import NDArray
from scipy.sparse import spmatrix

Vector = NDArray[Any]
Matrix = NDArray[Any]

SparseVector = spmatrix
SparseMatrix = spmatrix

AnyVector = Vector | SparseVector
AnyMatrix = Matrix | SparseMatrix
