#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import sys
from typing import Any, TypeAlias, cast, TypeVar, Union
from numpy.typing import NDArray
from scipy.sparse import spmatrix, issparse
import warnings

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


T = TypeVar("T", bound=Union[Vector, SparseVector, Matrix, SparseMatrix])

def as_dense(data: T, warn: bool = True) -> T:
    """Return the dense representation of the given data (vector or matrix),
    doing nothing if it is already dense."""
    if issparse(data):
        if warn:
            warnings.warn("Converting sparse to dense array")
        return cast(spmatrix, data).toarray()
    return data

def as_dense_vector(vector: AnyVector, warn: bool = True) -> Vector:
    return as_dense(vector, warn)

def as_dense_matrix(matrix: AnyMatrix, warn: bool = True) -> Matrix:
    return as_dense(matrix, warn)

