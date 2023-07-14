'''
Copyright 2022 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


import logging
import h5py
import numpy as np
import dask.array as da

def save_data_to_hdf5(data, file_path, internal_path="/data",**kwargs):
    """
    saves data to hdf5 file with name given by file_path, and internal folder given by internal_path

    kwargs are passed to h5py.File create_dataset() function
    """
    logging.info(f"Saving data of shape {data.shape} to {file_path} with kwargs {kwargs}.")
    with h5py.File(file_path, "w") as f:
        f.create_dataset(internal_path, data=data)
        
def read_h5_to_np(file_path):
    with h5py.File(file_path,'r') as data_file:
        data_hdf5=np.array(data_file['data'])
        
    return data_hdf5

def numpy_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['/data'][()]
    return np.array(data)

def read_h5_to_da(file_path):
    #Lazy reading, so don't use `with` but need to close later
    f=h5py.File(file_path, 'r')
    data_hdf5=da.from_array(f['data'])
    return data_hdf5
