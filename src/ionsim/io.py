

""" Module for input/output, read/write functionality with common data used in IonSim """ 


def save_matrix(datafile, matrix, pathname, attributes=None):
    """Save a matrix in as a dataset in an HDF5 file."""
    dataset = datafile.require_dataset(pathname, shape=matrix.shape, dtype=matrix.dtype, data=matrix)
    if attributes:
        for name, value in attributes.items():
            dataset.attrs[name] = value
    return dataset


def load_matrix(datafile, pathname):
    """Load a matrix into a numpy array and return the attributes
    associated with the HDF5 dataset."""
    dataset = datafile[pathname]
    arr = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(arr)
    attributes = {name: value for name, value in dataset.attrs.items()}
    return arr, attributes
