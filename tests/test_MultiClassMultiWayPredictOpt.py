#Not working well
import numpy as np

# import sys
# import os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
import leopardgecko as lg

# TODO

# #Show logging.info
# import logging
# logging.basicConfig(level=logging.INFO)

# boxwidth = 10
# nclasses = 3
# nways = 12

# #Volume to start 
# a0 = np.floor(np.random.rand(boxwidth,boxwidth,boxwidth)*(nways+1))
# a1 = np.floor(np.random.rand(boxwidth,boxwidth,boxwidth) * (nways-a0) )
# a2 = nways-a0-a1

# #a_all = np.tensor([a0,a1,a2]) #Doesn't work
# #a_all = [a0,a1,a2]
# a_all = np.zeros((nclasses , boxwidth,boxwidth,boxwidth) )
# a_all[0] = a0
# a_all[1] = a1
# a_all[2] = a2
# #print(a_all)

# gt_rnd =  np.floor(np.random.rand(boxwidth,boxwidth,boxwidth)*nclasses) #Lets assume this is the ground truth

# a_all = a_all.astype(np.uint8)
# gt_rnd = gt_rnd.astype(np.uint8)

# #Gets MaxAccMetric pvector
# pMulti0 = lg.MultiClassMultiWayPredictOptimizer(nclasses, nways)

# p_accmax, metric = pMulti0.getPCritForMaxMetric(a_all, gt_rnd, savemetricdatafile="testMultiClassMultiWayPredict.txt")
# print(p_accmax)
# print(metric)