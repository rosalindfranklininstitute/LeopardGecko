'''
Test Average Pool routines in leopard gecko

'''
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import leopardgecko.AvgPool as ap

import numpy as np
import dask.array as da

import logging
logging.getLogger().setLevel(logging.INFO)

#Create data
data3d_da = da.random.random_sample((1024,1024,1024))
print(f"data3d.shape = {data3d_da.shape}")

avgpooldata = ap.AvgPool3D_LargeData(data3d_da, s_stride=64)
print("Averagepool calculation finished")

