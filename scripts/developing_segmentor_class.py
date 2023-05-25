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

print("Segmentor training complete")

# #Save the segmentor
# lg_segm_fn = "lg_segmentor_model.pkl"
# print(f"Saving lg segmentor to file:{lg_segm_fn}")
# import pickle
# with open(lg_segm_fn, "wb") as f:
#     pickle.dump(lgsegmentor0,f) #Not working
#     # AttributeError: Can't pickle local object '_AbstractDiceLoss.__init__.<locals>.<lambda>'

# del(lgsegmentor0)
## Cannot save lg_segmentor model with pickle

# import joblib
# lg_segm_fn = "lg_segmentor_model.joblib"
# print(f"Saving lg segmentor to file:{lg_segm_fn}")
# joblib.dump(lgsegmentor0, lg_segm_fn)
## Saving wuth joblib also fails because it uses pickle

# will try without saving an running the full predictions with the object


#Prediction of large volume

print("Predict a whole (large) volume")

to_pred_fn="/ceph/users/ypu66991/data/GasHydrate/89062_1554x1554x200_uint8_data_clipped.h5"
print(f"Loading file: {to_pred_fn}")
data_to_pred = numpy_from_hdf5(to_pred_fn)

print("Prediction starting")

data_pred = lgsegmentor0.predict(data_to_pred)

#Save result
 
save_data_to_hdf5(data_pred, "89062_1554x1554x200_uint8_data_clipped_lg_segm_pred.h5")


print("** Completed **")
