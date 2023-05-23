import leopardgecko as lg
from leopardgecko.segmentor import *

traindata_crop_filename = "/ceph/users/ypu66991/Analysis/GasHydrate/ROI_1165_1421_512_768_512_768_ManualLabelFix/data_ROI_1165_1421_512_768_512_768.tiff"
trainlabels_crop_filename = "/ceph/users/ypu66991/Analysis/GasHydrate/ROI_1165_1421_512_768_512_768_ManualLabelFix/labels_manual.tif"

#Loads data
print("Loading data")
import tifffile
traindata = tifffile.imread(traindata_crop_filename)
trainlabels = tifffile.imread(trainlabels_crop_filename)

#Run segmentor with this data

#Create the class
lgsegmentor0 = cMultiAxisRotationsSegmentor()

print("Run segmentor train()")
#Run training
lgsegmentor0.train(traindata, trainlabels, get_metrics=True)

print("** Completed **")
#Completed successfully
