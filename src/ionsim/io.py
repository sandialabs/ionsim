
from pathlib import Path
import File 
import h5py
import numpy as np

from ionsim.custom_types import AnyMatrix 

""" Module for input/output, read/write functionality with common data used in IonSim """ 

def write_matrix(datafile: h5py.File, matrix: AnyMatrix, pathname: str, attributes=None):
    """Save a matrix in as a dataset in an HDF5 file."""
    dataset = datafile.require_dataset(pathname, shape=matrix.shape, dtype=matrix.dtype, data=matrix)
    if attributes:
        for name, value in attributes.items():
            dataset.attrs[name] = value
    return dataset


def read_matrix(datafile: h5py.File, pathname: str):
    """Load a matrix into a numpy array and return the attributes
    associated with the HDF5 dataset."""
    dataset = datafile[pathname]
    arr = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(arr)
    attributes = {name: value for name, value in dataset.attrs.items()}
    return arr, attributes




# example: 
with h5py.File(data_filename, 'w') as datafile:
    write_matrix(datafile, dxs, 'dx', attributes)
    write_matrix(datafile, dys, 'dy', attributes)
    write_matrix(datafile, F_data, 'relative_error', attributes)

