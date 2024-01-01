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


def map_vol_function_by_blocking(func0, data3d, block_shape, margins_shape):
    """
    Splits data3d to blocks of shape given, but with padding between them.
    Then it applies function func0 to each of the blocks and collect data.
    Resulting data is assumed to be the same shape as the input shape provided.
    The resulting data is reassembled to a large volume, with padded regions discarded
    except for margins.

    func0 must be func0( numpy array with ndim=3 ) with no extra arguments.

    If the function intended to be applied to the data volume requires arguments, please
    use functools to generate a function that only requires a single 3-dim numpy array

    An alternative to using this function is to use dask's map_overlap()
    https://docs.dask.org/en/latest/array-overlap.html

    Returns:
        datares: result after applying func0 to the whole volume using the 
        blocking algorithm described
    """

    shapedata=data3d.shape
    logging.debug(f"map_vol_function_by_blocking() , data3d.shape:{data3d.shape} ,dtype:{data3d.dtype} block_shape:{block_shape}, margins_shape:{margins_shape}")

    #If for some reason the block shape is not big enough along one or more directions
    bl_step0 = np.array([ block_shape[i]-2*margins_shape[i] for i in range(3) ])
    bl_step = np.where( bl_step0<=0, np.array(block_shape), bl_step0)

    logging.debug(f"bl_step:{bl_step}")

    datares = None #To collect results, it will be setup initially with correct dtype when first results arrive
    b_continue=True

    for iz0 in range(0,shapedata[0],bl_step[0]):
        if not b_continue:
            break

        iz00=iz0
        iz1 = iz0 + block_shape[0]
        if iz1>shapedata[0]:
            iz1 = shapedata[0]
            iz00 = iz1 - block_shape[0]
            if iz00<0: iz00=0
        
        for iy0 in range(0,shapedata[1], bl_step[1]):
            if not b_continue:
                break
            iy00 = iy0
            iy1 = iy0 + block_shape[1]
            if iy1>shapedata[1]:
                iy1 = shapedata[1]
                iy00 = iy1 - block_shape[1]
                if iy00<0: iy00=0

            for ix0 in range(0,shapedata[2], bl_step[2]):
                ix00 = ix0
                ix1 = ix0 + block_shape[2]
                if ix1>shapedata[2]:
                    ix1 = shapedata[2]
                    ix00 = ix1 - block_shape[2]
                    if ix00<0: ix00=0

                logging.info(f"BLOCK: New block, intended origin iz0,iy0,ix0 = {iz0},{iy0},{ix0} , use origin iz00,iy00,ix00 = {iz00},{iy00},{ix00} , end iz1,iy1,ix1 = {iz1},{iy1},{ix1}")

                #Get the data block
                datablock0 = data3d[iz00:iz1, iy00:iy1, ix00:ix1]
                
                logging.info("BLOCK: Start calculation with this block")

                #Do calculation with this datablock
                data_res_block = func0(datablock0)

                logging.info("BLOCK: This block's calculation completed")

                if data_res_block is None:
                    raise ValueError( "BLOCK: data_res_block is None. Check for errors. Stopping calculation")
                
                #Store the datablock result, only the valid part
                #unless it is the leftmost (first block) of the dimension given
                jz0=0
                jy0=0
                jx0=0

                #Crop the padded on the left side
                if iz0 !=0 :
                    #jz0 += int( (block_shape[0] - bl_step[0]) / 2)
                    jz0 += margins_shape[0]
                if iy0 !=0:
                    #jy0 += int( (block_shape[1] - bl_step[1]) / 2)
                    jy0 += margins_shape[1]
                if ix0 !=0:
                    #jx0 += int( (block_shape[2] - bl_step[2]) / 2)
                    jx0 += margins_shape[2]
                
                logging.info(f"BLOCK:Crop block result from origin jz0,jy0,jx0 = : {jz0},{jy0},{jx0}")
                
                logging.info(f"BLOCK:Copying cropped block to datares")

                if datares is None:
                    #Initialise
                    logging.info("BLOCK: First block result initialises datares")
                    datares = np.zeros(shapedata, dtype=data_res_block.dtype)

                datares[ iz00+jz0 : iz00+data_res_block.shape[0] , iy00+jy0 : iy00+data_res_block.shape[1] , ix00+jx0 : ix00+data_res_block.shape[2]] = data_res_block[jz0: , jy0: , jx0: ]

    logging.info("BLOCK: Completed. Results should be in datares")
    
    return datares