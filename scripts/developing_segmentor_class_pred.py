"""
For predicting a whole volume from a lg_segmentor object
To be used in conjuction  developing_segmentor_clas.py



"""

import leopardgecko as lg
from leopardgecko.segmentor import *
import pickle

from leopardgecko.utils import *

#Reload lg model
lg_segm_fn = "lg_segmentor_model.pkl"
print(f"Reload lg segmentor from file: {lg_segm_fn}")
with open("lg_segmentor_model.pkl", "r") as f:
    lgsegmentor1 = pickle.load(f)

print("lg segmentor loaded")

print("Predict a whole (large) volume")

to_pred_fn="/ceph/users/ypu66991/data/GasHydrate/89062_1554x1554x200_uint8_data_clipped.h5"
print(f"Loading file: {to_pred_fn}")
data_to_pred = numpy_from_hdf5(to_pred_fn)

print("Prediction starting")

data_pred = lgsegmentor1.predict(data_to_pred)

#Save result
 
save_data_to_hdf5(data_pred, "89062_1554x1554x200_uint8_data_clipped_lg_segm_pred.h5")

print("** Completed **")
