"""
TODO: 

"""

import numpy as np
import matplotlib.pyplot as plt
import dask.array as da

#import subprocess
import tempfile

from pathlib import Path

import os
cwd = os.getcwd()
#print(cwd)

import tempfile
import logging
from types import SimpleNamespace

# NN1 is volume segmnatics
import volume_segmantics.utilities.config as cfg
from volume_segmantics.data import TrainingDataSlicer
from volume_segmantics.model import VolSeg2dTrainer

from volume_segmantics.data import get_settings_data
from volume_segmantics.model import VolSeg2DPredictionManager
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor
from volume_segmantics.utilities.base_data_utils import Axis

import random #For shuffling lists
import tqdm #progress bar in iterations

from sklearn.neural_network import MLPClassifier #NN2

from . import metrics
from .utils import *


class cMultiAxisRotationsSegmentor():

    def __init__(self, models_prefix="lg_segmentor_model_", temp_data_outdir=None):
        model_NN1_fn = models_prefix+"NN1.pytorch"
        #model_NN2_fn = models_prefix+"NN2.pk"
        
        self.model_NN1_path = Path(cwd,model_NN1_fn)
        #self.model_NN2_path = Path(cwd, model_NN2_fn)
        
        self.chunkwidth = 64
        self._init_settings()
        self.nlabels=None #Will be used for chunking data

        self.temp_data_outdir=temp_data_outdir

    def _init_settings(self):
        #Initialise internal settings for the neural networks
        settings0 = {'data_im_dirname': 'data',
            'seg_im_out_dirname': 'seg',
            'model_output_fn': 'trained_2d_model',
            'clip_data': False,
            'st_dev_factor': 2.575,
            'data_hdf5_path': '/data',
            'seg_hdf5_path': '/data',
            'training_axes': 'All',
            'image_size': 256,
            'downsample': False,
            'training_set_proportion': 0.8,
            'cuda_device': 0,
            'num_cyc_frozen': 8,
            'num_cyc_unfrozen': 5,
            'patience': 3,
            'loss_criterion': 'DiceLoss',
            'alpha': 0.75,
            'beta': 0.25,
            'eval_metric': 'MeanIoU',
            'pct_lr_inc': 0.3,
            'starting_lr': '1e-6',
            'end_lr': 50,
            'lr_find_epochs': 1,
            'lr_reduce_factor': 500,
            'plot_lr_graph': False,
            'model': {'type': 'U_Net',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet'}}

        self.NN1_train_settings = SimpleNamespace(**settings0)

        settings1 = {'quality': 'high',
            'output_probs': True,
            'clip_data': True,
            'st_dev_factor': 2.575,
            'data_hdf5_path': '/data',
            'cuda_device': 0,
            'downsample': False,
            'one_hot': False,
            'prediction_axis': 'Z'}

        self.NN1_pred_settings = SimpleNamespace(**settings1)

        #Default setting for NN" MLP classifier
        #Note that these settings are not saved like the others
        #They are just here if user wants to setup different settings for NN2 before training
        settingsNN2 ={
            'hidden_layer_sizes':[10,10],
            'random_state':1,
            'verbose':True,
            'activation':'tanh',
            'learning_rate_init':0.001,
            'solver':'sgd',
            'max_iter':1000
        } 
        self.NN2_settings = SimpleNamespace(**settingsNN2)

    
    def train(self, traindata, trainlabels, get_metrics=True):
        """
        Train NN1 (volume segmantics) and NN2 (MLP Classifier)
        """

        self.labels_dtype= trainlabels.dtype

        # logging.basicConfig(
        #     level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
        # )
        
        #Train NN1
        self.NN1_train(traindata, trainlabels)

        #Does the multi-axis multi-rotation predictions
        # and collects data files
        tempdir_pred=None
        if self.temp_data_outdir is None:
            tempdir_pred= tempfile.TemporaryDirectory()
            tempdir_pred_path = Path(tempdir_pred.name)
        else:
            tempdir_pred_path=Path(self.temp_data_outdir)
        
        print(f"tempdir_pred_path:{tempdir_pred_path}")

        #Predict multi-axis multi-rotations
        #Predictions are stored in h5 files in temprary folder
        pred_data_probs_filenames, pred_data_labels_filenames=self.NN1_predict(traindata, tempdir_pred_path)
        print("pred_data_probs_filenames")
        print(pred_data_probs_filenames)
        print("pred_data_labels_filenames")
        print(pred_data_labels_filenames)

        #Need to train next model by running predictions and optimize MLP

        #Use multi-predicted data and labels to train NN2

        #Build data object containing all predictions
        data0 = read_h5_to_np(pred_data_probs_filenames[0])

        all_shape = ( len(pred_data_probs_filenames), *data0.shape )
        data_all_np = np.zeros( all_shape)
        #Fill with data
        data_all_np[0,:,:,:,:]= data0
        for i in tqdm.trange(1,len(pred_data_probs_filenames), desc="Loading prediction files"):
            #print(i)
            data_i = read_h5_to_np(pred_data_probs_filenames[i])
            data_all_np[i,:,:,:,:]=data_i

        #Take this oportunity to calculate metrics of each prediction if required
        nn1_acc_dice_s= []
        if get_metrics:
            for i, label_fn0 in enumerate(pred_data_labels_filenames):
                data_i = read_h5_to_np(label_fn0)
                a0 =  metrics.MetricScoreOfVols_Accuracy(data_i,trainlabels)
                d0 = metrics.MetricScoreOfVols_Dice(data_i,trainlabels)
                nn1_acc_dice_s.append( [a0,d0])
                print(f"prediction:{i} , filename: {label_fn0}, accuracy:{a0}, dice:{d0}")


        #Train NN2 from multi-axis multi-angle predictions against labels (gnd truth)
        nn2_acc, nn2_dice = self.NN2_train(data_all_np, trainlabels, get_metrics=get_metrics)

        if not tempdir_pred is None:
            tempdir_pred.cleanup()

        return nn1_acc_dice_s, (nn2_acc, nn2_dice)
    
    def predict(self, data_in):
        """
        Creates predicted labels from a whole data volume
        using the double NN1+NN2 pipeline
        """
        #Predict from provided volumetric data using the trained model defined here

        #Check if the following objects are avaialble
        #self.volseg2pred #NN1 predictor (attention the NN1_predict() loads the model from file!!)
        
        if not self.model_NN1_path is None and not self.NN2 is None:
            print("NN1 prediction")

            tempdir_pred=None
            if self.temp_data_outdir is None:
                tempdir_pred= tempfile.TemporaryDirectory()
                tempdir_pred_path = Path(tempdir_pred.name)
            else:
                tempdir_pred_path=Path(self.temp_data_outdir)

            pred_data_probs_filenames, _ = self.NN1_predict(data_in, tempdir_pred_path) #Get prediction probs, not labels
            print("NN1 prediction, complete.")
            
            print("Building large object containing all predictions.")
            #Build data object containing all predictions
            #Try using numpy. If memory error use dask instead
            try:
                data0 = read_h5_to_np(pred_data_probs_filenames[0]) 
                all_shape = ( len(pred_data_probs_filenames), *data0.shape )
                print(f"all_shape:{all_shape}")
                data_all = np.zeros(all_shape, dtype=data0.dtype) #May lead to very large dataset which may lead to memory allocation error
                #Fill with data
                data_all[0,:,:,:]= data0
                for i in tqdm.trange(1,len(pred_data_probs_filenames), desc="Loading prediction files"):
                    data_i = read_h5_to_np(pred_data_probs_filenames[i])
                    data_all[i,:,:,:,:]=data_i
            except:
                print("Allocation using numpy failed. Will use dask.")

                data0 = read_h5_to_da(pred_data_probs_filenames[0]) 
                all_shape = ( len(pred_data_probs_filenames), *data0.shape )
                print(f"all_shape:{all_shape}")

                chunks_shape = (len(pred_data_probs_filenames), *data0.chunksize )
                print(f"dask data_all will have chunksize set to {chunks_shape}")
                data_all=da.zeros(all_shape, chunks=chunks_shape )
                #in case of 12 predictions and 3 labels, the chunks will be (12,128,128,128,3) size

                #Fill with data
                data_all[0,:,:,:]= data0
                for i in tqdm.trange(1,len(pred_data_probs_filenames), desc="Loading predictions"):
                    print(i)
                    data_i = read_h5_to_da(pred_data_probs_filenames[i])
                    data_all[i,:,:,:,:]=data_i

            print("NN2 prediction")
            d_prediction= self.NN2_predict( data_all)
            
            print("NN2 prediction complete.")

            if not tempdir_pred is None:
                print(f"Cleaning up tempdir_pred: {tempdir_pred_path}")
                tempdir_pred.cleanup()

            return d_prediction

        return None


    def NN2_train(self, train_data_all_probs, trainlabels, get_metrics=True):
        print("NN2 train")

        #Get several points to train NN2
        x_origs = np.arange(0, train_data_all_probs.shape[3],5)
        y_origs = np.arange(0,train_data_all_probs.shape[2],5)
        z_origs = np.arange(0,train_data_all_probs.shape[1],5)
        x_mg, y_mg, z_mg = np.meshgrid(x_origs,y_origs, z_origs)
        all_origs_list = np.transpose(np.vstack( (z_mg.flatten() , y_mg.flatten() , x_mg.flatten() ) ) ).tolist()

        random.shuffle(all_origs_list)
        ntrain = min(len(all_origs_list), 4096)

        X_train=[] # as list of volume data, flattened for each voxel
        
        for i in tqdm.trange(ntrain):
            el = all_origs_list[i]
            z,y,x = el
            data_vol = train_data_all_probs[:,z,y,x,:]
            data_vol_flat = data_vol.flatten()
            X_train.append(data_vol_flat)

        y_train=[] # labels
        for i in tqdm.trange(ntrain):
            el = all_origs_list[i]
            z,y,x = el
            label_vol_label = trainlabels[z,y,x]
            y_train.append(label_vol_label)

        #Setup classifier
        print("Setup NN2 MLPClassifier")
        #self.NN2 = MLPClassifier(hidden_layer_sizes=(10,10), random_state=1, activation='tanh', verbose=True, learning_rate_init=0.001,solver='sgd', max_iter=1000)
        self.NN2 = MLPClassifier(**self.NN2_settings.__dict__) #Unpack dict to become parameters

        #Do the training here
        print(f"NN2 MLPClassifier fit with {len(X_train)} samples, (y_train {len(y_train)} samples)")
        self.NN2.fit(X_train,y_train)

        print(f"NN2 train score:{self.NN2.score(X_train,y_train)}")

        nn2_acc=None
        nn2_dice=None
        if get_metrics:
            print("Preparing to predict the whole training volume")
        
            d_prediction= self.NN2_predict( train_data_all_probs)

            #Get metrics
            nn2_acc= metrics.MetricScoreOfVols_Accuracy(trainlabels,d_prediction)
            nn2_dice= metrics.MetricScoreOfVols_Dice(trainlabels,d_prediction, useBckgnd=False)

            print(f"NN2 acc:{nn2_acc}, dice:{nn2_dice}")
        
        return nn2_acc, nn2_dice


    def NN2_predict(self, data_all_probs):
        
        print("NN2_predict()")

        if isinstance(data_all_probs, np.ndarray):
            print("Data type is numpy.ndarray")

            #Need to flatten along the npred and nclasses
            data_2MLP_t= np.transpose(data_all_probs,(1,2,3,0,4))

            dsize = data_2MLP_t.shape[0]*data_2MLP_t.shape[1]*data_2MLP_t.shape[2]
            inputsize = data_2MLP_t.shape[3]*data_2MLP_t.shape[4]

            data_2MLP_t_reshape = np.reshape(data_2MLP_t, (dsize, inputsize))

            #Uses the MLP classifier
            mlppred = self.NN2.predict(data_2MLP_t_reshape)

            #Reshape back to 3D
            mlppred_3D = np.reshape(mlppred, data_2MLP_t.shape[0:3])

            return mlppred_3D
        
        elif isinstance(data_all_probs, da.core.Array):
            print("Data type is dask.core.Array")
            #Use dask reduction functionality to do the predictions

            def chunkf(x,axis, keepdims, computing_meta=False):
                #Function to apply to each chunk
                #Assumes that data has the right chunk dimensions
                data0 = np.asarray(x)
                data_2MLP_t= np.transpose(data0,(1,2,3,0,4))
                dsize = data_2MLP_t.shape[0]*data_2MLP_t.shape[1]*data_2MLP_t.shape[2]
                inputsize = data_2MLP_t.shape[3]*data_2MLP_t.shape[4]
                data_2MLP_t_reshape = np.reshape(data_2MLP_t, (dsize, inputsize))

                #Runs the MLPClassifier prediction on this chunk
                mlppred = self.NN2.predict(data_2MLP_t_reshape)
                
                #Reshape back to 5D
                mlppred_3D_chunk = np.reshape(mlppred, (1,*data_2MLP_t.shape[0:3],1))

                return mlppred_3D_chunk


            def aggf(x, axis, keepdims):
                #Function to aggregate chunks. In this case it just reduces the dimensions by
                # removing the axis with width of one (multiplane and label axis)
                #print(f"aggf: axis:{axis}, keepdims:{keepdims}, x.shape:", x.shape)
                if not keepdims:
                    x_res= np.squeeze(x, axis=(0,4)) #Remove axis 0 and 4
                    return x_res
                return x


            b = da.reduction(data_all_probs,
                            chunk=chunkf,
                            aggregate= aggf,
                            dtype=self.labels_dtype,
                            keepdims=False,
                            axis=(0,4)) #It appeears that his axis parameter is simply passed to chnkf and aggf and that's it.

            print("Starting dask computation")
            b_comp=b.compute()
            print(f"Completed. res shape:{b_comp.shape}")

            return b_comp

    def NN1_train(self, traindata, trainlabels):

        tempdir_data=None
        tempdir_seg=None
        if self.temp_data_outdir is None:
            tempdir_data = tempfile.TemporaryDirectory()
            tempdir_data_path=Path(tempdir_data.name)

            tempdir_seg = tempfile.TemporaryDirectory()
            tempdir_seg_path = Path(tempdir_seg.name)

            # tempdir_pred= tempfile.TemporaryDirectory()
            # tempdir_pred_path = Path(tempdir_pred.name)
        else:
            tempdir_data_path=Path(self.temp_data_outdir,"NN1_data")
            tempdir_data_path.mkdir(exist_ok=True)
            tempdir_seg_path=Path(self.temp_data_outdir, "NN1_seg")
            tempdir_seg_path.mkdir(exist_ok=True)

        # tempdir_data = tempfile.TemporaryDirectory()
        # tempdir_data_path=Path(tempdir_data.name)
        print(f"tempdir_data_path:{tempdir_data_path}")

        # tempdir_seg = tempfile.TemporaryDirectory()
        # tempdir_seg_path = Path(tempdir_seg.name)
        print(f"tempdir_seg_path:{tempdir_seg_path}")

        # Keep track of the number of labels
        max_label_no = 0
        label_codes = None

        # Set up the DataSlicer and slice the data volumes into image files
        slicer = TrainingDataSlicer(traindata, trainlabels, self.NN1_train_settings)
        data_prefix, label_prefix = "",""
        slicer.output_data_slices(tempdir_data_path, data_prefix)
        slicer.output_label_slices(tempdir_seg_path, label_prefix)
        if slicer.num_seg_classes > max_label_no:
            max_label_no = slicer.num_seg_classes
            label_codes = slicer.codes

        # Set up the 2dTrainer
        self.trainer = VolSeg2dTrainer(tempdir_data_path, tempdir_seg_path, max_label_no, self.NN1_train_settings)
        # Train the model, first frozen, then unfrozen
        num_cyc_frozen = self.NN1_train_settings.num_cyc_frozen
        num_cyc_unfrozen = self.NN1_train_settings.num_cyc_unfrozen
        #model_type = settings.model["type"].name

        if num_cyc_frozen > 0:
            self.trainer.train_model(
                self.model_NN1_path, num_cyc_frozen, self.NN1_train_settings.patience, create=True, frozen=True
            )
        if num_cyc_unfrozen > 0 and num_cyc_frozen > 0:
            self.trainer.train_model(
                self.model_NN1_path, num_cyc_unfrozen, self.NN1_train_settings.patience, create=False, frozen=False
            )
        elif num_cyc_unfrozen > 0 and num_cyc_frozen == 0:
            self.trainer.train_model(
                self.model_NN1_path, num_cyc_unfrozen, self.NN1_train_settings.patience, create=True, frozen=False
            )
        #trainer.output_loss_fig(model_out)
        #trainer.output_prediction_figure(model_out)
        #plt.show()
        # Clean up all the saved slices
        slicer.clean_up_slices()

        if not tempdir_data is None:
            print("tempdir_data and tempdir_seg cleanup.")
            tempdir_data.cleanup()
            tempdir_seg.cleanup()


    def NN1_predict(self,data_to_predict, pred_folder_out):
        """
        
        Does the multi-axis multi-rotation predictions
        and returns predictions filenames of probablilities and labels

        predictions are probabilities (not labels)

        Returns: a tuple with
        pred_data_probs_filenames
        pred_data_labels_filenames

        """

        #Load volume segmantics model from file to class instance
        self.volseg2pred = VolSeg2dPredictor(self.model_NN1_path, self.NN1_pred_settings, use_dask=True)

        def save_pred_data(data, axis, rot):
            # Saves predicted data to h5 file in tempdir and return file path in case it is needed
            file_path = f"{pred_folder_out}/pred_{axis}_{rot}.h5"
            
            save_data_to_hdf5(data, file_path)
            return file_path
        
        data_vol = np.array(data_to_predict) #Copies

        pred_data_probs_filenames=[] #Will store results in files, and keep the filenames as reference
        pred_data_labels_filenames=[]
        for krot in range(0, 4):
            rot_angle_degrees = krot * 90
            logging.info(f"Volume rotated by {rot_angle_degrees} degrees")

            data_vol = np.rot90(np.array(data_to_predict),krot) #Copies

            #Predict 3 axis
            #YX
            logging.info("Predicting YX slices:")
            #returns (labels,probabilities)
            res = self.volseg2pred._predict_single_axis_all_probs(
                data_vol, axis=Axis.Z
            )
            pred_probs = np.rot90(res[1], -krot) #invert rotation before saving
            #Saves prediction labels
            #Sets nlabels from last dimension. Assumes last dimension is number of labels
            #Used to chunk data when saving
            self.nlabels=pred_probs.shape[-1]
            fn = save_pred_data(pred_probs, "YX", rot_angle_degrees)
            pred_data_probs_filenames.append(fn)

            pred_labels = np.rot90(res[0], -krot)
            fn = save_pred_data(pred_labels, "YX_labels", rot_angle_degrees)
            pred_data_labels_filenames.append(fn)
            
            #ZX
            logging.info("Predicting ZX slices:")
            res = self.volseg2pred._predict_single_axis_all_probs(
                data_vol, axis=Axis.Y
            )
            pred_probs = np.rot90(res[1], -krot) #invert rotation before saving
            fn = save_pred_data(pred_probs, "ZX", rot_angle_degrees)
            pred_data_probs_filenames.append(fn)

            pred_labels = np.rot90(res[0], -krot)
            fn = save_pred_data(pred_labels, "ZX_labels", rot_angle_degrees)
            pred_data_labels_filenames.append(fn)

            #ZY
            logging.info("Predicting ZY slices:")
            res= self.volseg2pred._predict_single_axis_all_probs(
                data_vol, axis=Axis.X
            )
            pred_probs = np.rot90(res[1], -krot) #invert rotation before saving
            fn = save_pred_data(pred_probs, "ZY", rot_angle_degrees)
            pred_data_probs_filenames.append(fn)

            pred_labels = np.rot90(res[0], -krot)
            fn = save_pred_data(pred_labels, "ZY_labels", rot_angle_degrees)
            pred_data_labels_filenames.append(fn)

        del(data_vol)

        return pred_data_probs_filenames, pred_data_labels_filenames


    def save_model(self, filename):
        """
        Saves model to a zip file containing the following
        NN1 model (volume segmantics pytorch)
        NN1 settings (yaml file?)
        NN2 model (MPL pickle)
        NN2 settings
        """

        #Generate files in temporary storage
        #tempdir_model = tempfile.TemporaryDirectory()
        #tempdir_model_path=Path(tempdir_model)

        import io
        import joblib
        #import pickle
        
        #NN1 settings
        nn1_train_settings_bytesio = io.BytesIO()
        joblib.dump(self.NN1_train_settings, nn1_train_settings_bytesio)

        nn1_pred_settings_bytesio = io.BytesIO()
        joblib.dump(self.NN1_pred_settings, nn1_pred_settings_bytesio)

        # nn2_settings_bytesio = io.BytesIO()
        # joblib.dump(self.NN2_settings, nn2_settings_bytesio)
        # Don't need to save NN2 settings seperately
        # as they are already included in NN2 MLPclassifier (self.NN2)

        #NN2 model
        nn2_model_bytesio = io.BytesIO()
        joblib.dump(self.NN2, nn2_model_bytesio)

        #NN1 model is in file path self.model_NN1_path

        from zipfile import ZipFile

        with ZipFile(filename, 'w') as zipobj:
            zipobj.write(self.model_NN1_path.name, arcname="NN1_model.pytorch")
            zipobj.writestr("NN1_train_settings.joblib",nn1_train_settings_bytesio.getvalue())
            zipobj.writestr("NN1_pred_settings.joblib", nn1_pred_settings_bytesio.getvalue())
            zipobj.writestr("NN2_model.joblib", nn2_model_bytesio.getvalue())


        nn1_pred_settings_bytesio.close()
        nn1_pred_settings_bytesio.close()
        nn2_model_bytesio.close()

    def load_model(self, filename):
        #import io
        import joblib
        from zipfile import ZipFile

        with ZipFile(filename, 'r') as zipobj:
            ##NN1 model
            nn1_model_temp_dir = tempfile.TemporaryDirectory()
            zipobj.extract("NN1_model.pytorch",nn1_model_temp_dir.name)
            self.model_NN1_path=Path(nn1_model_temp_dir.name,"NN1_model.pytorch")

            with zipobj.open("NN1_train_settings.joblib",'r') as z0:
                self.NN1_train_settings= joblib.load(z0)

            with zipobj.open("NN1_pred_settings.joblib",'r') as z1:
                self.NN1_pred_settings= joblib.load(z1)
            
            with zipobj.open("NN2_model.joblib",'r') as z2:
                self.NN2= joblib.load(z2)

    @staticmethod
    def create_from_model( filename):
        newobj = cMultiAxisRotationsSegmentor()
        newobj.load_model(filename)

        return newobj
