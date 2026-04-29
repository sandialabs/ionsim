import numpy as np
from pathlib import Path
import h5py

from ionsim.custom_types import AnyMatrix 

""" Module for input/output, read/write functionality with common data used in IonSim """ 
# Opening the file with 'w' allows reading and writing and
# truncates existing data. See https://docs.h5py.org/en/stable/high/file.html

def write_results_to_file(data_filename: str, results: dict, attributes: dict=None):
    """ Write a set of results to a data file """
    with h5py.File(data_filename, 'w') as datafile: 
        for key in results.keys(): 
            write_matrix(datafile, results[key], key, attributes)

    return 0 # successful write 


def read_results_from_file(data_filename: str):
    """ Read a set of results from a data file """
    results = {}
    attributes_from_file = {}
    with h5py.File(data_filename, 'r') as datafile: 
        # loop over all attributes in the file
        attributes = list(datafile.keys())
        for attribute in attributes:
            results[attribute], attributes_from_file[attribute] = read_matrix(datafile, attribute) 

    return results, attributes_from_file 
                

def write_matrix(datafile: h5py.File, matrix: AnyMatrix, pathname: str, attributes: dict = None):
    """ Save a matrix in as a dataset in an HDF5 file. The matrix is saved into its own directory ``pathname'' """
    dataset = datafile.require_dataset(pathname, shape=matrix.shape, dtype=matrix.dtype, data=matrix)
    if attributes:
        for name, value in attributes.items():
            dataset.attrs[name] = value
    return dataset


def read_matrix(datafile: h5py.File, pathname: str):
    """Load a matrix into a numpy array and return the attributes associated with the HDF5 dataset."""
    dataset = datafile[pathname]
    arr = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(arr)
    attributes = {name: value for name, value in dataset.attrs.items()}
    return arr, attributes
