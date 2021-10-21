#Test MultiClassMultiWayPredictOptimizer class in leopardgecko.py

import numpy as np

import os.path
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import leopardgecko as lg

#Show logging.info
import logging
logging.basicConfig(level=logging.INFO)

#TODO: Create tests

#Consider
nclasses = 3
nways = 12

#Create class instance
myCm = lg.MultiClassMultiWayPredictOptimizer(nclasses,nways)

#Test getCombinations() listing all the hvectors possible
print("Test myCm.getCombinations() should list all combinations for hypervector")
print(myCm.getCombinations())
#Test ok


#test isValidHvector()
p0 = myCm.getCombinations()[20]
print(f"Check isValidHvector() with hvector {p0}")
print(f"Result = {myCm.isValidHvector(p0)}")


print("Test getSegmentationProjMatrix()")
hv0 = (12,0,0)
print(f"for hv0={hv0}")
print(myCm.getSegmentationProjMatrix(hv0))

hv0 = (0,12,0)
print(f"for hv0={hv0}")
print(myCm.getSegmentationProjMatrix(hv0))

hv0 = (0,0,12)
print(f"for hv0={hv0}")
print(myCm.getSegmentationProjMatrix(hv0))

hv0 = (4,4,4)
print(f"for hv0={hv0}")
print(myCm.getSegmentationProjMatrix(hv0))

#Appears to be ok


print("Set a value of pcrit and get the class segmentation for a given hvector")
pcrit0=(4,4,4)
myCm.set_pcrit( pcrit0 )
hvector0 = (8,2,2)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (2,8,2)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (2,2,8)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (12,0,0)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (0,12,0)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (0,0,12)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (4,4,4)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )

pcrit0=(6,3,3)
myCm.set_pcrit( pcrit0 )
hvector0 = (8,2,2)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (2,8,2)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (2,2,8)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (12,0,0)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (0,12,0)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (0,0,12)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (4,4,4)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (4,5,3)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (4,3,5)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )
hvector0 = (8,2,2)
print(f"With pcrit={pcrit0} and hvector = {hvector0}, classnumber = {myCm.getClassNumber(hvector0)} " )




print("Test dice score calculation of binary segmented MetricScoreOfVols_Dice()")

vol0s = np.zeros((8,8,8), dtype=np.uint8)
vol1s = np.ones((4,4,4), dtype=np.uint8)

A= np.zeros_like(vol0s)
A[1:5, 2:6 , 2:6] = vol1s[:,:,:]
#volume of ones = 4*4*4 = 64

B=np.zeros_like(vol0s)
B[3:7, 2:6, 2:6] = vol1s[:,:,:]
#volume of ones = 4*4*4 = 64

# overlap [3:5 , 2:6, 2:6], volume = 2*4*4 = 32
#Dice score expected, 2*32 / (64+64) = 64/128  = 1/2 = 0.5

print(f"Dicescore = {myCm.MetricScoreOfVols_Dice(A,B)} (expected 0.5)")


print("Test dice score calculation on a 3-class segmented MetricScoreOfVols_Dice()")
vol0s = np.zeros((8,8,8), dtype=np.uint8)
vol1s = np.ones((2,4,4), dtype=np.uint8)
vol2s = np.ones((2,5,4),dtype=np.uint8) * 2

A=np.zeros_like(vol0s)
A[2:4 , 1:5 , 2:6] = vol1s
A[4:6 , 1:6 , 2:6] = vol2s

B=np.zeros_like(vol0s)
B[2:4 , 3:7 , 2:6] = vol1s
B[4:6 , 2:7 , 2:6] = vol2s

#segm1
#volume in A = 2*4*4 = 32, volume in B = 2*4*4 = 32
#overlap segm1 = [2:4 , 3:5 , 2:6], volume = 2*2*4 = 16
#Dice for segm1 = 2*16 / (32+32) = 32/64 = 0.5

#segm2
#volume in A = 2*5*4 = 40, volume in B = 2*5*4 = 40
#overlap = [4:6 , 2:6 , 2:6], volume = 2*4*4 = 32
#dice  = 2*32 / (40+40) = 64/80 = 8/10 = 0.8

#Dice all = mean(0.5 , 0.8) = 0.65
print(f"Dicescore = {myCm.MetricScoreOfVols_Dice(A,B)} (expected 0.65)")


#print("Test identifiyClassFromVols()")
#TODO






# boxwidth = 10


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