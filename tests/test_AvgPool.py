'''
Test Average Pool routines in leopard gecko

'''
# import sys
# import os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))

import leopardgecko.AvgPool as ap
import leopardgecko.ScoreData

import numpy as np
import dask.array as da

# import logging
# logging.getLogger().setLevel(logging.INFO)


def test_AvgPool3D_LargeData():
    #Create data
    datasize = 512

    data3d_da = da.random.random_sample((datasize,datasize,datasize))
    #print(f"data3d.shape = {data3d_da.shape}")

    s_stride=32
    k_width = 64
    blocksize = 256

    avgpooldata_scoredata = ap.AvgPool3D_LargeData(data3d_da, blocksize=blocksize, k_width= k_width ,s_stride=s_stride)
    # Returns a ScoreData object
    #print("Averagepool calculation finished")

    expectedsize = (datasize - k_width)/s_stride +1 #blocksize is only for gpu
    
    assert avgpooldata_scoredata.data3d.shape[0] == expectedsize