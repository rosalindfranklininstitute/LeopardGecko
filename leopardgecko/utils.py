import logging
import h5py
import numpy as np
import dask.array as da

def save_data_to_hdf5(data, file_path, internal_path="/data", chunking=True):
    logging.info(f"Saving data of shape {data.shape} to {file_path}.")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("/data", data=data, chunks=chunking)
def read_h5_to_np(file_path):
    with h5py.File(file_path,'r') as data_file:
        data_hdf5=np.array(data_file['data'])
        
    return data_hdf5

def numpy_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['/data'][()]
    return np.array(data)

def read_h5_to_da(file_path):
    #Lazy reading, so don't use `with``
    f=h5py.File(file_path, 'r')
    data_hdf5=da.from_array(f['data'])
    return data_hdf5
