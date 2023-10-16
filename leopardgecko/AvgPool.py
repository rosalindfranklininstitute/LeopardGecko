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

import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import time
import dask.array as da
import logging
import sys

from .ScoreData import *

try:
    import torch
except Exception as e:
    logging.info("AvgPool.py. Could not import torch libraries. This module requires pytorch to be installed.")
    logging.info(str(e))
    sys.exit()

# TODO: Modify to use blocking algorithm in utils

def AvgPool3D_LargeData(data3d, *,blocksize = 512, k_width=256 , s_stride=8 ):
    #This function will do the avarage pooling of a 3D data volume using PyTorch AvgPool3D
    #with a windows with cubic size given by k_width (kernel width)
    # and stride (step) of s_stride.
    # To reduce GPU load, data is split into chunks of blocksize
    #and then combines the data automaticaly
    #It returns a ScoreData object


    def AvgPool3DPytorch(data3d_np , kwidth=8 , stride0=1):
        '''
        Applies Pytorch AvgPool3D on the numpy data object data3d_np with the width and stride parameters given
        Automatically uses GPU if available.
        This function will not check if the data is too large. Use this with caution
        It is recommended that data3d_np has a maximum of 512x512x512 size, to keep GPU usage acceptably low
        The function returns the average-pooled data as a numpy 3D array objects
        '''
        #Generic. It will use the GPU if available

        if torch.cuda.is_available():
                dev="cuda:0"
        else:
                dev="cpu"
        device = torch.device(dev)
        
        #convert to torch objects, and to gpu using cuda()
        data3d_torch = torch.unsqueeze( torch.unsqueeze( torch.from_numpy(data3d_np),0),0 ).to(device)
        
        #setup torch calculation
        torchc3d = torch.nn.AvgPool3d(kwidth, stride0)
        
        #Run the calculation
        result = torchc3d(data3d_torch)
        
        return result.cpu().detach().numpy()[0][0]

    def Get3DAvgPoolOfBlockWithCornerAt( data3d, iz,iy,ix , blocksize , k_width, s_stride ):
        '''
        Averages the 3D data3d (dask array), from corner (iz, iy, ix)
        and with blocks of bloscksize in all directions.
        So, it does AvgPool3D at region [ iz : iz+w_avg , iy : iy+w_avg , iy : iy+w_avg ]
        and with a kernel size of k_width x k_width x k_width , and stride (jump) of s_stride.
        If the desired block exceeds the limits of the data3d , it will adjust the index limits in order to fit
        Because it may change the index limits, the actual limits are also returned.
        datavol_avg
        '''

        #Check all is ok
        assert ( iz >=0 and iy>=0 and ix>=0 ) , "Error, indexes cannot be < 0 ."
        
        assert ( iz < data3d.shape[0] and
                iy < data3d.shape[1] and
                ix < data3d.shape[2]) , "Error, invalid indexes."
        
        #Adjust limits
        iz_da_min = iz
        iz_da_max = iz_da_min + blocksize
        if iz_da_max > data3d.shape[0] :
            iz_da_max = data3d.shape[0]
            iz_da_min = iz_da_max- blocksize
        
        iy_da_min = iy
        iy_da_max = iy_da_min + blocksize
        if iy_da_max > data3d.shape[1] :
            iy_da_max = data3d.shape[1]
            iy_da_min = iy_da_max- blocksize
        
        ix_da_min = ix
        ix_da_max = ix_da_min + blocksize
        if ix_da_max > data3d.shape[2] :
            ix_da_max = data3d.shape[2]
            ix_da_min = ix_da_max- blocksize
        
        logging.info( "iz_da_min=" +str(iz_da_min) + ", iz_da_max=" + str(iz_da_max) +
            ", iy_da_min=" + str(iy_da_min) + ", iy_da_max=" + str(iy_da_max) +
            ", ix_da_min=" + str(ix_da_min) + ", ix_da_max=" + str(ix_da_max)
            )
        
        #Get volume and convert to numpy array
        datavol_da = data3d [ iz_da_min:iz_da_max , iy_da_min:iy_da_max , ix_da_min:ix_da_max ]

        #print("datavol_da.shape = ", datavol_da.shape)
        #convert to numpy
        datavol_np = datavol_da.compute()
        #print("datavol_np.shape = ", datavol_np.shape)

        #Calculate here the AvgPooling (big calculation)
        #datavol_avg = AvgPool3DPytorchGPU(datavol_np , k_width , s_stride )
        datavol_avg = AvgPool3DPytorch(datavol_np , k_width , s_stride )

        logging.info("AvgPool3D calculation complete")
        #logging.info("datavol_avg.shape = ",datavol_avg.shape)
        
        torch.cuda.empty_cache()
        
        return datavol_avg, (iz_da_min , iz_da_max , iy_da_min , iy_da_max , ix_da_min , ix_da_max)


    res=None

    assert (blocksize > k_width), "w_avg (window width average) should be higher than kwidth"
    
    # if (do_weighting):
    #     setWeightedData('MaxMinSquare')
    
    # data3d = self.weightedData_da

    if data3d is not None :
        result_avg_of_vols = np.zeros( ( int( (data3d.shape[0]-k_width)/s_stride )+1 , 
                            int( (data3d.shape[1]-k_width)/s_stride )+1  ,
                            int( (data3d.shape[2]-k_width)/s_stride )+1  ))

        logging.info ("result_avg_of_vols.shape = " + str(result_avg_of_vols.shape) )
        
        # BIG CALCULATION
        
        #Nested iterations of w_avg x w_avg x w_avg volumes
        #step0 = int( (w_avg - k_width) / s_stride )
        step0 = int(blocksize - k_width)
        
        niter = 0 #Count the number of ierations
        time0 = time.perf_counter()
        time1 = time0
        
        ntotaliter = int(data3d.shape[0]/step0) * int(data3d.shape[1]/step0)* int(data3d.shape[2]/step0)
        
        logging.info ("ntotaliter  = " + str(ntotaliter) )
        
        time.sleep(2) #A little pause to see print output
        
        for iz_da in range(0 , data3d.shape[0] , step0):
            for iy_da in range(0 , data3d.shape[1] , step0):
                for ix_da in range(0 , data3d.shape[2] , step0):
                
                    #Show progress
                    #clear_output(wait=True)
                    
                    if (niter>0):
                        logging.info("niteration = " + str(niter) + "/" + str(ntotaliter) )
                        logging.info ("Estimated time to finish (s) = "+
                            str( round( (ntotaliter-niter)*(time1-time0)/niter) ) )
                    
                    logging.info("iz_da=" + str(iz_da) + "/" + str(data3d.shape[0]) +
                        " , iy_da=" + str(iy_da) + "/" + str(data3d.shape[1]) +
                        " , ix_da=" + str(ix_da) + "/" + str(data3d.shape[2])
                        )

                    datavol_avg , index_limits = Get3DAvgPoolOfBlockWithCornerAt(data3d, iz_da,iy_da,ix_da , blocksize , k_width, s_stride )
                    
                    #clear_output(wait=True)
                    
                    time1 = time.perf_counter()
                    niter += 1
                    
                    #With data collected, store it in appropriate array 
                    #print("index_limits = ", index_limits)
                    iz = int(index_limits[0] / s_stride)
                    iy = int(index_limits[2] / s_stride)
                    ix = int(index_limits[4] / s_stride)

                    #print ("Start indexes to store at result_avg_of_vols: " , iz , iy , ix)

                    result_avg_of_vols[ iz : (iz + datavol_avg.shape[0]) ,
                                    iy : (iy + datavol_avg.shape[1]) ,
                                    ix : (ix + datavol_avg.shape[2]) ] = datavol_avg

        logging.info("Completed.")      
        
        
        #Create the respective indexes
        #Indexes are the midpoints of the respective averaging volume
        # (No index should have a value of 0)
        result_avg_of_vols_x_range = np.arange( int(k_width/2) , data3d.shape[2]-int(k_width/2)+1, s_stride )
        result_avg_of_vols_y_range = np.arange( int(k_width/2) , data3d.shape[1]-int(k_width/2)+1 , s_stride )
        result_avg_of_vols_z_range = np.arange( int(k_width/2) , data3d.shape[0]-int(k_width/2)+1 , s_stride )

        #Attention, order of x,y,z has to be in this way
        #otherwise the vales will not correspond to the averaging point volumes.
        #In a 3D array, first index is zz, 2nd is yy, and 3rd is xx
        result_avg_of_vols_z , result_avg_of_vols_y , result_avg_of_vols_x = np.meshgrid( result_avg_of_vols_z_range ,
                                                                                        result_avg_of_vols_y_range,
                                                                                        result_avg_of_vols_x_range,
                                                                                        indexing='ij')
        #Resulting meshgrids should have the same shape as result_avg_of_vols
        logging.info ("result_avg_of_vols_x.shape = " + str(result_avg_of_vols_x.shape ) )
        
        #Create a ScoreData object containing all the data
        res = ScoreData(result_avg_of_vols , result_avg_of_vols_z , result_avg_of_vols_y , result_avg_of_vols_x )

    else:
        logging.error("No data3d.")

    return res
