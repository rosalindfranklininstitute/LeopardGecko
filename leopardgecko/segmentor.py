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

import pandas as pd


class cMultiAxisRotationsSegmentor():

    def __init__(self, lgmodel_fn=None, models_prefix="lg_segmentor_model_", temp_data_outdir=None, cuda_device=0):
        logging.debug(f"cMultiAxisRotationsSegmentor __init__() with temp_data_outdir:{temp_data_outdir} , cuda_device:{cuda_device}")
        import random
        prefn = random.randint(0,10000)
        model_NN1_fn = models_prefix+f"NN1_{prefn:04}.pytorch" #4 digits random to prevent clashes

        #model_NN2_fn = models_prefix+"NN2.pk"
        
        #self.model_NN1_path = Path(cwd,model_NN1_fn)
        #self.model_NN2_path = Path(cwd, model_NN2_fn)

        #stops creating pytorch model in current folder
        self._pytorch_model_tempdir= tempfile.TemporaryDirectory()
        self.model_NN1_path = Path(self._pytorch_model_tempdir.name,model_NN1_fn)
        
        self.chunkwidth = 64
        
        self.nlabels=None #Will be used for chunking data

        self.temp_data_outdir=temp_data_outdir

        #self.cuda_device=cuda_device
        self._init_settings()
        self.set_cuda_device(cuda_device)

        self.all_nn1_pred_pd=None

        self._tempdir_pred=None

        if not lgmodel_fn is None:
            self.load_model(lgmodel_fn)

    def _init_settings(self):
        #Initialise internal settings for the neural networks
        NN1trainsettings0 = {'data_im_dirname': 'data',
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
            'encoder_weights': 'imagenet'},
            }

        self.NN1_train_settings = SimpleNamespace(**NN1trainsettings0)

        NN1predsettings0 = {'quality': 'high',
            'output_probs': True,
            'clip_data': True,
            'st_dev_factor': 2.575,
            'data_hdf5_path': '/data',
            'cuda_device': 0,
            'downsample': False,
            'one_hot': False,
            'prediction_axis': 'Z'}

        self.NN1_pred_settings = SimpleNamespace(**NN1predsettings0)

        #Default setting for NN" MLP classifier
        #Note that these settings are not saved like the others
        #They are just here if user wants to setup different settings for NN2 before training
        settingsNN2 ={
            'hidden_layer_sizes':[10,10],
            'activation':'tanh',
            'random_state':1,
            'verbose':True,
            'learning_rate_init':0.001,
            'solver':'sgd',
            'max_iter':1000,
            'ntrain':262144 #Note that this is not a MLPClassifier parameter
        }
        self.NN2_settings = SimpleNamespace(**settingsNN2)
        self.labels_dtype=None #default

        #added settings
        # self.NN1_volsegm_pred_path=None
        # self.NN1_consistencyscore_outpath=None


    def set_cuda_device(self,n):
        self._cuda_device=n
        self.NN1_train_settings.cuda_device=n
        self.NN1_pred_settings.cuda_device=n

    
    def train(self, traindata, trainlabels, get_metrics=True):
        """
        Train NN1 (volume segmantics) and NN2 (MLP Classifier)

        Returns:
            Tuple nn1_acc_dice_s, (nn2_acc, nn2_dice)
            with nn1_acc_dice_s being a list of accuracy, dice of each predictions

            and (nn2_acc, nn2_dice) being the accuracy and dice result from NN1+NN2 combination
            
        """
        logging.debug(f"train()")
        trainlabels0 = None
        traindata0=None
        #Check traindata is 3D or list
        if isinstance(traindata, np.ndarray) and isinstance(trainlabels, np.ndarray) :
            logging.info("traindata and trainlabels are ndarray")
            if traindata.ndim!=3 or trainlabels.ndim!=3:
                raise ValueError(f"traindata or trainlabels not 3D")
            else:
                #Convert to list so that can be used later
                traindata0 = [traindata]
                trainlabels0=[trainlabels]
        else:
            if isinstance(traindata, list) and isinstance(trainlabels, list):
                logging.info("traindata and trainlabels are list")
                if len(traindata)!=len(trainlabels):
                    raise ValueError("len(traindata)!=len(trainlabels) error. Must be the same number of items.")
                else:
                    traindata0=traindata
                    trainlabels0=trainlabels

        self.labels_dtype= trainlabels[0].dtype

        #How many sets?
        nsets=len(traindata0)
        logging.info(f"nsets:{nsets}")

        # logging.basicConfig(
        #     level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
        # )
        
        # ** Train NN1
        self.NN1_train(traindata0, trainlabels0)
        #(This does not return anything.)

        # ** Predict NN1
        #Does the multi-axis multi-rotation predictions
        # and collects data files
        
        # Setup temporary folders to store predictions
        self._tempdir_pred=None
        if self.temp_data_outdir is None:
            self._tempdir_pred= tempfile.TemporaryDirectory()
            tempdir_pred_path = Path(self._tempdir_pred.name)
        else:
            tempdir_pred_path=Path(self.temp_data_outdir)
        
        logging.info(f"tempdir_pred_path:{tempdir_pred_path}")

        #Predict multi-axis multi-rotations
        #Predictions are stored in h5 files in temporary folder

        self.all_nn1_pred_pd = self.NN1_predict(traindata0, tempdir_pred_path) #note that 
        logging.info("NN1_predict returned")
        logging.info(self.all_nn1_pred_pd)

        #Take this oportunity to calculate metrics of each prediction labels if required
        nn1_acc_dice_s= []
        #pred_data_probs_filenames=all_pred_pd['pred_data_labels_filenames'].tolist() #note that all sets will be included in this list
        if get_metrics:
            logging.info("Collecting NN1 metrics")
            for i, prow in self.all_nn1_pred_pd.iterrows():
                pred_labels_fn = prow['pred_data_labels_filenames']
                iset = prow['pred_sets']
                ipred = prow['pred_ipred']
                data_i = read_h5_to_np(pred_labels_fn)

                #What is the corresponding iset?
                a0 =  metrics.MetricScoreOfVols_Accuracy(data_i,trainlabels0[iset])
                d0 = metrics.MetricScoreOfVols_Dice(data_i,trainlabels0[iset])
                nn1_acc_dice_s.append( [a0,d0])
                logging.info(f"prediction iset:{iset}, ipred:{ipred}, filename: {pred_labels_fn}, accuracy:{a0}, dice:{d0}")

            #add metrics to pandas dataframe with results
            acc_list0 = [ ad0[0] for ad0 in nn1_acc_dice_s]
            self.all_nn1_pred_pd["accuracy"]= acc_list0

            dice_list0 = [ ad0[1][0] for ad0 in nn1_acc_dice_s]
            self.all_nn1_pred_pd["dice"]= dice_list0



        # ** NN2 training

        #Need to train next model by running predictions and optimize MLP
        #Use multi-predicted data and labels to train NN2
        #Build data object containing all predictions (5D - iset, ipred, z,y,x, class)

        npredictions_per_set = int(np.max(self.all_nn1_pred_pd['pred_ipred'].to_numpy())+1)
        logging.info(f"npredictions_per_set:{npredictions_per_set}")

        # data0 = read_h5_to_np(pred_data_probs_filenames[0])
        # print(f"data0.shape:{data0.shape}")

        # all_shape = (
        #     npredictions_per_set,
        #     int(data0.shape[0]*nsets),
        #     *data0.shape[1:]
        #     )
        # print(f"all_shape:{all_shape}")

        # data_all_np = np.zeros( all_shape, dtype=data0.dtype)
        # #Fill with data
        # data_all_np[0,:,:,:,:]= data0
        # for i in tqdm.trange(1,len(pred_data_probs_filenames), desc="Loading prediction files"):
        #     #print(i)
        #     data_i = read_h5_to_np(pred_data_probs_filenames[i])
        #     data_all_np[i,:,:,:,:]=data_i
        
        data_all_np5d=None

        logging.debug("Aggregating multiple sets onto a single volume data_all_np5d")
        # aggregate multiple sets for data
        for i,prow in self.all_nn1_pred_pd.iterrows():

            prob_filename = prow['pred_data_probs_filenames']
            data0 = read_h5_to_np(prob_filename)

            if i==0:
                #initialise
                logging.info(f"data0.shape:{data0.shape}")
                all_shape0 = (
                    nsets,
                    npredictions_per_set,
                    *data0.shape
                    )
                # (iset, ipred, iz,iy,ix, ilabel) , 5dim

                data_all_np5d=np.zeros( all_shape0 , dtype=data0.dtype)

            
            ipred=prow['pred_ipred']
            iset=prow['pred_sets']

            data_all_np5d[iset,ipred, :,:,:, :] = data0


        #Train NN2 from multi-axis multi-angle predictions against labels (gnd truth)
        nn2_acc, nn2_dice = self.NN2_train(data_all_np5d, trainlabels0, get_metrics=get_metrics)

        #Preserve for debugging
        # if not tempdir_pred is None:
        #     tempdir_pred.cleanup()

        return nn1_acc_dice_s, (nn2_acc, nn2_dice)
    
    
    def predict(self, data_in, use_dask=False):
        """
        Creates predicted labels from a whole data volume
        using the double NN1+NN2 pipeline
        """
        #Predict from provided volumetric data using the trained model defined here

        #Check if the following objects are avaialble
        #self.volseg2pred #NN1 predictor (attention the NN1_predict() loads the model from file!!)
        logging.debug(f"predict() data_in.shape:{data_in.shape}, data_in.dtype:{data_in.dtype}, use_dask:{use_dask}")

        if not self.model_NN1_path is None and not self.NN2 is None:
            logging.info("Setting up NN1 prediction")

            self._tempdir_pred=None
            if self.temp_data_outdir is None:
                self._tempdir_pred= tempfile.TemporaryDirectory()
                tempdir_pred_path = Path(self._tempdir_pred.name)
            else:
                tempdir_pred_path=Path(self.temp_data_outdir)

            #pred_data_probs_filenames, _ = self.NN1_predict(data_in, tempdir_pred_path) #Get prediction probs, not labels
            self.all_nn1_pred_pd = self.NN1_predict(data_in, tempdir_pred_path) #Get prediction probs, not labels
            logging.info("NN1 prediction, complete.")
            logging.info("all_pred_pd")
            logging.info(self.all_nn1_pred_pd)
            
            data_all = self.aggregate_nn1_pred_data(use_dask)

            if data_all is None:
                logging.error("After aggregation, data_all is None")
            
            d_prediction=None #Default return value
            if not data_all is None:
                logging.info("Setting up NN2 prediction")
                d_prediction= self.NN2_predict(data_all)
                
                logging.info("NN2 prediction complete.")

            # if not tempdir_pred is None:
            #     logging.info(f"Cleaning up tempdir_pred: {tempdir_pred_path}")
            #     tempdir_pred.cleanup()

            return d_prediction


    def NN1_train(self, traindata_list, trainlabels_list):
        """
        traindata: a list of 3d volumes
        trainlabels : a list of 3d volumes with corresponding labels

        This code is share similarities to volumesegmantics train_2d_model.py
        
        """

        if not(isinstance(traindata_list, list) and isinstance(trainlabels_list, list) ):
            raise ValueError("Invalid traindata_list or trainlabels_list")
        
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
        logging.info(f"tempdir_data_path:{tempdir_data_path}")

        # tempdir_seg = tempfile.TemporaryDirectory()
        # tempdir_seg_path = Path(tempdir_seg.name)
        logging.info(f"tempdir_seg_path:{tempdir_seg_path}")

        # Keep track of the number of labels
        max_label_no = 0
        label_codes = None

        # Set up the DataSlicer and slice the data volumes into image files
        for count , (traindata0, trainlabels0) in enumerate(zip(traindata_list, trainlabels_list)):
            slicer = TrainingDataSlicer(traindata0, trainlabels0, self.NN1_train_settings)
            data_prefix, label_prefix = f"data{count}", f"seg{count}"
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

        # Clean up all the saved slices
        slicer.clean_up_slices()

        if not tempdir_data is None:
            logging.info("tempdir_data and tempdir_seg cleanup.")
            tempdir_data.cleanup()
            tempdir_seg.cleanup()


    def NN1_predict(self,data_to_predict, pred_folder_out):
        """
        
        Does the multi-axis multi-rotation predictions
        and returns predictions filenames of probablilities and labels

        predictions are probabilities (not labels)

        Params:
            data_to_predict: a ndarray or a list of ndarrays with the 3D data to rund predictions from
            pred_folder_out: a string with the location of where to drop results in h5 file format


        Returns:
            a pandas Dataframe with results of predictions in
            filenames of probabilities and labels,
            and respective set, rotation, plane, and ipred

            Columns are
                'pred_data_probs_filenames'
                'pred_data_labels_filenames'
                'pred_sets'
                'pred_planes'
                'pred_rots'
                'pred_ipred'

        """

        #Load volume segmantics model from file to class instance
        #self.volseg2pred = VolSeg2dPredictor(self.model_NN1_path, self.NN1_pred_settings, use_dask=True)
        #Using this VolSeg2dPredictor will not clip data
        #Also moved this functionality to later

        # For volumesegmantics standard predictions if set
        self.labels_vs_2stack = None
        self.probs_vs_2stack = None

        from . import ConsistencyScore
        # For consistency score determination from predictions if set
        consistencyscore0 = ConsistencyScore.cConsistencyScoreMultipleWayProbsAccumulate()


        logging.debug("NN1_predict()")
        #Internal functions
        def _save_pred_data(data, count,axis, rot):
            # Saves predicted data to h5 file in tempdir and return file path in case it is needed
            file_path = f"{pred_folder_out}/pred_{count}_{axis}_{rot}.h5"
            
            save_data_to_hdf5(data, file_path)
            return file_path
        
        # def _handle_pred_data_probs(self,pred_probs, pred_labels, count,axis,rot):
        #     #nonlocal attempts to grab these variables defined in previous scope
        #     # Another way to handle this is to define variable with self.
        #     # nonlocal labels_vs
        #     # nonlocal probs_vs
            
        #     # pred_probs is in format (z,y,x, class)
        #     # pred_labels is in format (z,y,x)

        #     #Accumulate for consistency score
        #     logging.debug(f"_handle_pred_data_probs(), count,axis,rot:{count},{axis},{rot}, self.NN1_consistencyscore_outpath:{self.NN1_consistencyscore_outpath}, self.NN1_volsegm_pred_path:{self.NN1_volsegm_pred_path}")
            
        #     if not self.NN1_consistencyscore_outpath is None:
        #         consistencyscore0.accumulate(pred_probs)

        #     # Accumulate for volume segmantics
        #     if not self.NN1_volsegm_pred_path is None:
        #         logging.debug("Calculating probs_class_squeezed")
        #         # # Squeeze probabilities along class
        #         # # by grabing the highest probable class label and probability
        #         # max_prob_idx = np.argmax(pred_probs, axis=1, keepdims=True)
        #         # # Extract along axis from outputs
        #         # probs_class_squeezed = np.take(pred_probs, axis=1, indices=max_prob_idx) #(data,indices, axis)
        #         # # Remove the label dimension
        #         # probs_class_squeezed = np.squeeze(probs_class_squeezed, axis=1)

        #         # max_prob_idx = torch.argmax(probs, dim=1, keepdim=True)
        #         # # Extract along axis from outputs
        #         # probs = torch.gather(probs, 1, max_prob_idx)  #(data,dim, index)
        #         # # Remove the label dimension
        #         # probs = torch.squeeze(probs, dim=1)

        #         probs_class_squeezed = np.max(pred_probs, axis=pred_probs.ndim-1)

        #         if self.labels_vs_2stack is None:
        #             logging.info(f"self.NN1_volsegm_pred_path provided:{self.NN1_volsegm_pred_path}. Will merge and save to predicted labels using volumesegmantics method.")
        #             logging.debug("First labels and probs file initializes")
        #             shape_tup = pred_labels.shape
        #             self.labels_vs_2stack = np.empty((2, *shape_tup), dtype=pred_labels.dtype)
        #             self.probs_vs_2stack = np.empty((2, *shape_tup), dtype=pred_probs.dtype)
        #             self.labels_vs_2stack[0]=pred_labels
        #             self.probs_vs_2stack[0]=probs_class_squeezed
        #         else:
        #             self.labels_vs_2stack[1]=pred_labels
        #             self.probs_vs_2stack[1]=probs_class_squeezed

        #             self._squeeze_merge_vols_by_max_prob(self.probs_vs_2stack,self.labels_vs_2stack)

        #     #Save and return the result (filepath) from _save_pred_data() function 
        #     return _save_pred_data(pred_probs, count,axis,rot)

        data_to_predict_l=None
        if not isinstance(data_to_predict, list):
            logging.debug("data_to_predict not a list. Converting to list")
            data_to_predict_l=[data_to_predict]
        else:
            logging.debug("data_to_predict is a list. No conversion needed")
            data_to_predict_l=data_to_predict

        pred_data_probs_filenames=[] #Will store results in files, and keep the filenames as reference
        pred_data_labels_filenames=[]
        pred_sets=[]
        pred_planes=[]
        pred_rots=[]
        pred_ipred=[]
        pred_shapes=[]

        logging.info(f"number of data sets to predict: {len(data_to_predict_l)}")
        
        for i, data_to_predict0 in enumerate(data_to_predict_l):
            logging.info(f"Data to predict index:{i}")
            data_vol1 = np.array(data_to_predict0) #Copies

            #setup Prediction Manager
            #It will also clip data depending on settings, and to get that data
            # it is property data_vol
            volseg2pred_m = VolSeg2DPredictionManager(
                model_file_path= self.model_NN1_path,
                data_vol=data_vol1,
                settings=self.NN1_pred_settings,
                #use_dask=True
                )

            data_vol0 = volseg2pred_m.data_vol  #Collects clipped data

            itag=0

            # reinitialise
            self.labels_vs_2stack = None 
            self.probs_vs_2stack = None
            consistencyscore0.clear()

            for krot in range(0, 4):
                rot_angle_degrees = krot * 90
                logging.info(f"Volume to be rotated by {rot_angle_degrees} degrees")

                #Predict 3 axis
                #YX
                planeYX=(1,2)
                logging.info("Predicting YX slices:")
                data_vol = np.rot90(np.array(data_vol0),krot, axes=planeYX) #rotate
                #returns (labels,probabilities)
                res = volseg2pred_m.predictor._predict_single_axis_all_probs(
                    data_vol,
                    axis=Axis.Z
                )
                pred_probs = np.rot90(res[1], -krot, axes=planeYX) #invert rotation before saving
                pred_labels = np.rot90(res[0], -krot, axes=planeYX)
                fn = _save_pred_data(pred_probs, i, "YX", rot_angle_degrees)
                #fn = _handle_pred_data_probs(self,pred_probs,pred_labels, i, "YX", rot_angle_degrees)

                #Saves prediction labels
                #Sets nlabels from last dimension. Assumes last dimension is number of labels
                #Used to chunk data when saving
                self.nlabels=pred_probs.shape[-1]
                pred_data_probs_filenames.append(fn)

                fn = _save_pred_data(pred_labels, i, "YX_labels", rot_angle_degrees)
                pred_data_labels_filenames.append(fn)

                pred_sets.append(i)
                pred_planes.append("YX")
                pred_rots.append(rot_angle_degrees)
                pred_ipred.append(itag)
                pred_shapes.append(pred_labels.shape)
                itag+=1

                
                #ZX
                logging.info("Predicting ZX slices:")
                planeZX=(0,2)
                data_vol = np.rot90(np.array(data_vol0),krot, axes=planeZX) #rotate
                res = volseg2pred_m.predictor._predict_single_axis_all_probs(
                    data_vol, axis=Axis.Y
                )
                pred_probs = np.rot90(res[1], -krot, axes=planeZX) #invert rotation before saving
                pred_labels = np.rot90(res[0], -krot, axes=planeZX)
                fn = _save_pred_data(pred_probs, i, "ZX", rot_angle_degrees)
                #fn = _handle_pred_data_probs(self,pred_probs,pred_labels, i, "ZX", rot_angle_degrees)
                pred_data_probs_filenames.append(fn)

                fn = _save_pred_data(pred_labels, i, "ZX_labels", rot_angle_degrees)
                pred_data_labels_filenames.append(fn)

                pred_sets.append(i)
                pred_planes.append("ZX")
                pred_rots.append(rot_angle_degrees)
                pred_ipred.append(itag)
                pred_shapes.append(pred_labels.shape)
                itag+=1

                #ZY
                logging.info("Predicting ZY slices:")
                planeZY=(0,1)
                data_vol = np.rot90(np.array(data_vol0),krot, axes=planeZY) #rotate
                res= volseg2pred_m.predictor._predict_single_axis_all_probs(
                    data_vol, axis=Axis.X
                )
                pred_probs = np.rot90(res[1], -krot, axes=planeZY) #invert rotation before saving
                pred_labels = np.rot90(res[0], -krot, axes=planeZY)
                fn = _save_pred_data(pred_probs, i, "ZY", rot_angle_degrees)
                #fn = _handle_pred_data_probs(self,pred_probs,pred_labels, i, "ZY", rot_angle_degrees)
                pred_data_probs_filenames.append(fn)

                pred_labels = np.rot90(res[0], -krot, axes=planeZY)
                fn = _save_pred_data(pred_labels, i, "ZY_labels", rot_angle_degrees)
                pred_data_labels_filenames.append(fn)

                pred_sets.append(i)
                pred_planes.append("ZY")
                pred_rots.append(rot_angle_degrees)
                pred_ipred.append(itag)
                pred_shapes.append(pred_labels.shape)
                itag+=1

            del(data_vol)

        logging.debug("Generating a DataFrame object with information about predictions")

        all_pred_pd = pd.DataFrame({
            'pred_data_probs_filenames': pred_data_probs_filenames,
            'pred_data_labels_filenames': pred_data_labels_filenames,
            'pred_sets':pred_sets,
            'pred_planes':pred_planes,
            'pred_rots':pred_rots,
            'pred_ipred':pred_ipred,
            'pred_shapes': pred_shapes,
        })
        
        # #This code below is untested
        # #Run standard volume segmantics merging of predicted volumes and saves as h5 file
        # if not self.NN1_volsegm_pred_path is None:
        #     logging.info(f"volsegm_pred_path provided:{self.NN1_volsegm_pred_path}, saving merged prediction labels")

        #     #Upon completion, save labels
        #     save_data_to_hdf5(self.labels_vs_2stack[0],self.NN1_volsegm_pred_path)

        # #Get the final consistency score
        # if not self.NN1_consistencyscore_outpath is None:
        #     save_data_to_hdf5(consistencyscore0.getCScore(),self.NN1_consistencyscore_outpath)

        #Clean up
        del(self.labels_vs_2stack)
        del(self.probs_vs_2stack)

        #return pred_data_probs_filenames, pred_data_labels_filenames
        return all_pred_pd


    def NN2_train(self, train_data_all_probs_5d, trainlabels_list, get_metrics=True):
        logging.debug("NN2 train()")

        #Assumes train_data_all_probs_list is 5d
        # and that trainlabels_list is a list of 3d volumes

        assert train_data_all_probs_5d.shape[0]==len(trainlabels_list)

        nsets= len(trainlabels_list)

        logging.debug("Getting several points to train NN2")
        # #This is probably not the best way to get a random points
        # #Get several points to train NN2
        # x_origs = np.arange(0, train_data_all_probs_5d.shape[3],5)
        # y_origs = np.arange(0,train_data_all_probs_5d.shape[2],5)
        # z_origs = np.arange(0,train_data_all_probs_5d.shape[1],5)
        # x_mg, y_mg, z_mg = np.meshgrid(x_origs,y_origs, z_origs)
        # all_origs_list = np.transpose(np.vstack( (z_mg.flatten() , y_mg.flatten() , x_mg.flatten() ) ) ).tolist()

        # random.shuffle(all_origs_list)
        # #ntrain = min(len(all_origs_list), 4096)
        # ntrain = min(len(all_origs_list), self.NN2_settings.ntrain)

        # X_train=[] # as list of volume data, flattened for each voxel
        
        # iset_randoms = np.random.default_rng().integers(0,nsets,ntrain)

        # for i in tqdm.trange(ntrain):
        #     el = all_origs_list[i]
        #     z,y,x = el
        #     data_vol = train_data_all_probs_5d[iset_randoms[i],:,z,y,x,:]
        #     data_vol_flat = data_vol.flatten()
        #     X_train.append(data_vol_flat)

        # y_train=[] # labels
        # for i in tqdm.trange(ntrain):
        #     el = all_origs_list[i]
        #     z,y,x = el
        #     label_vol_label = trainlabels_list[iset_randoms[i]][z,y,x]
        #     y_train.append(label_vol_label)

        ntrain=self.NN2_settings.ntrain

        iset_rnd = np.random.randint(0,nsets,ntrain)
        z_orig_rnd = np.random.randint(0,train_data_all_probs_5d.shape[2],ntrain)
        y_orig_rnd = np.random.randint(0,train_data_all_probs_5d.shape[3],ntrain)
        x_orig_rnd = np.random.randint(0,train_data_all_probs_5d.shape[4],ntrain)
        
        all_origs_list=np.column_stack( (iset_rnd,z_orig_rnd,y_orig_rnd,x_orig_rnd ))

        #Could probably check for duplicates, but I will sckip that part
        #Collect voxels data
        X_train=[]
        Y_train=[] # labels
        for i in tqdm.trange(ntrain):
            el = all_origs_list[i,:]
            iset,z,y,x = el
            data_vol = train_data_all_probs_5d[iset,:,z,y,x,:]
            data_vol_flat = data_vol.flatten()
            X_train.append(data_vol_flat)

            label_vol_label = trainlabels_list[iset][z,y,x]
            Y_train.append(label_vol_label)

        logging.debug(f"NN2 len(X_train):{len(X_train)} , len(Y_train):{len(Y_train)}")

        #Setup classifier
        logging.info("Setup NN2 MLPClassifier")
        #self.NN2 = MLPClassifier(hidden_layer_sizes=(10,10), random_state=1, activation='tanh', verbose=True, learning_rate_init=0.001,solver='sgd', max_iter=1000)
        #self.NN2 = MLPClassifier(**self.NN2_settings.__dict__) #Unpack dict to become parameters

        self.NN2 = MLPClassifier(
            hidden_layer_sizes=self.NN2_settings.hidden_layer_sizes,
            activation=self.NN2_settings.activation,
            random_state=self.NN2_settings.random_state,
            verbose=self.NN2_settings.verbose,
            learning_rate_init=self.NN2_settings.learning_rate_init,
            solver=self.NN2_settings.solver,
            max_iter=self.NN2_settings.max_iter
            )

        # self.NN2 = MLPClassifier(
        #     hidden_layer_sizes=self.NN2_settings.hidden_layer_sizes,
        #     activation=self.NN2_settings.activation,
        #     random_state=self.NN2_settings.random_state,
        #     verbose=self.NN2_settings.verbose,
        #     learning_rate_init=self.NN2_settings.learning_rate_init,
        #     solver=self.NN2_settings.solver,
        #     max_iter=self.NN2_settings.max_iter,
        #     loss= self.dice_loss_np #chatgpt advise, but was wrong, sklearn MLP does not support costum loss functions
        #     )
        
        #Do the training here
        logging.info(f"NN2 MLPClassifier fit with {len(X_train)} samples, (y_train {len(Y_train)} samples)")
        self.NN2.fit(X_train,Y_train)

        logging.info(f"NN2 train score:{self.NN2.score(X_train,Y_train)}")

        nn2_acc=[]
        nn2_dice=[]
        if get_metrics:
            logging.info("Preparing to predict the whole training volume")

            for i in range(nsets):
                d_prediction= self.NN2_predict( train_data_all_probs_5d[i,:,:,:,:,:])

                #Get metrics
                nn2_acc0= metrics.MetricScoreOfVols_Accuracy(trainlabels_list[i],d_prediction)
                nn2_dice0= metrics.MetricScoreOfVols_Dice(trainlabels_list[i],d_prediction, useBckgnd=False)

                logging.info(f"set {i}, NN2 acc:{nn2_acc0}, dice:{nn2_dice0}")
                nn2_acc.append(nn2_acc0)
                nn2_dice.append(nn2_dice0)
        
        return nn2_acc, nn2_dice
    

    def NN2_predict(self, data_all_probs):
        # version that uses ParallelPostfit
        logging.debug("NN2_predict()")
        from dask_ml.wrappers import ParallelPostFit
        from dask.diagnostics import ProgressBar

        data_all_probs_da=None
        if isinstance(data_all_probs, np.ndarray):
            logging.info("Data type is numpy.ndarray")
            data_all_probs_da = da.from_array(data_all_probs)
        
        elif isinstance(data_all_probs, da.core.Array):
            logging.info("Data type is dask.core.Array")
            #Use dask reduction functionality to do the predictions
            data_all_probs_da=data_all_probs
        
        if data_all_probs_da is None:
            raise ValueError("data_all_probs invalid")
        

        #Need to flatten along the npred and nclasses
        data_2MLP_t= da.transpose(data_all_probs_da,(1,2,3,0,4))

        dsize = data_2MLP_t.shape[0]*data_2MLP_t.shape[1]*data_2MLP_t.shape[2]
        inputsize = data_2MLP_t.shape[3]*data_2MLP_t.shape[4]

        data_2MLP_t_reshape = da.reshape(data_2MLP_t, (dsize, inputsize))

        mlp_PPF_parallel = ParallelPostFit(self.NN2)
        mlppred = mlp_PPF_parallel.predict(data_2MLP_t_reshape)

        #Reshape back to 3D
        mlppred_3D = da.reshape(mlppred, data_2MLP_t.shape[0:3])
        
        # logging.info("Starting NN2 predict dask computation")
        pbar = ProgressBar()
        with pbar:
            b_comp=mlppred_3D.compute() #compute and convert to numpy

        return b_comp


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

        logging.debug("save_model()")

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
            zipobj.write(str(self.model_NN1_path), arcname="NN1_model.pytorch")
            zipobj.writestr("NN1_train_settings.joblib",nn1_train_settings_bytesio.getvalue())
            zipobj.writestr("NN1_pred_settings.joblib", nn1_pred_settings_bytesio.getvalue())
            zipobj.writestr("NN2_model.joblib", nn2_model_bytesio.getvalue())


        nn1_pred_settings_bytesio.close()
        nn1_pred_settings_bytesio.close()
        nn2_model_bytesio.close()

    #Do not save the pandas file

    def load_model(self, filename):
        #import io
        logging.debug("load_model()")
        import joblib
        from zipfile import ZipFile

        with ZipFile(filename, 'r') as zipobj:
            ##NN1 model
            self._nn1_model_temp_dir = tempfile.TemporaryDirectory()
            zipobj.extract("NN1_model.pytorch",self._nn1_model_temp_dir.name)
            self.model_NN1_path=Path(self._nn1_model_temp_dir.name,"NN1_model.pytorch")

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
    
    def aggregate_nn1_pred_data(self, use_dask=False):
        logging.debug(f"aggregate_nn1_pred_data with use_dask:{use_dask}")
        if self.all_nn1_pred_pd is None:
            return None
        
        logging.info("Building large object containing all predictions.")
        #Build data object containing all predictions
        #Try using numpy. If memory error use dask instead

        data_all=None

        if not use_dask:
            logging.info("use_dask=False. Will try to aggregate data to a numpy.ndarray")
            try:
                data_all=None
                # aggregate multiple sets for data
                for i,prow in tqdm.tqdm(self.all_nn1_pred_pd.iterrows(), total=self.all_nn1_pred_pd.shape[0]):

                    prob_filename = prow['pred_data_probs_filenames']
                    data0 = read_h5_to_np(prob_filename)

                    if i==0:
                        #initialise
                        logging.info(f"data0.shape:{data0.shape}")
                        npredictions = int(np.max(self.all_nn1_pred_p['pred_ipred'].to_numpy())+1)
                        logging.info(f"npredictions:{npredictions}")
                        
                        all_shape = (
                            npredictions,
                            *data0.shape
                            )
                        # (ipred, iz,iy,ix, ilabel) , 5dim
                        
                        data_all = np.zeros(all_shape, dtype=data0.dtype)

                    data_all[i,:,:,:,:]=data0

            except Exception as exc0:
                logging.info("Allocation using numpy failed. Failsafe will use dask.")
                logging.info(f"Exception type:{type(exc0)}")
                use_dask=True
            
        if use_dask:
            logging.info("use_dask=True. Will aggregate data to a dask.array object")
            try:
                data_all=None
                # aggregate multiple sets for data
                for i,prow in tqdm.tqdm(self.all_nn1_pred_pd.iterrows(), total=self.all_nn1_pred_pd.shape[0]):

                    prob_filename = prow['pred_data_probs_filenames']
                    data0 = read_h5_to_da(prob_filename)

                    if i==0:
                        #initialise
                        logging.info(f"i:{i}, data0.shape:{data0.shape}, data0.chunksize:{data0.chunksize} ")
                        npredictions = int(np.max(self.all_nn1_pred_pd['pred_ipred'].to_numpy())+1)
                        logging.info(f"npredictions:{npredictions}")
                        
                        #chunks_shape = (npredictions, *data0.chunksize )
                        #in case of 12 predictions and 3 labels, the chunks will be (12,128,128,128,3) size
                        all_shape = ( npredictions,*data0.shape)

                        #max chunksize in xyz of 1024
                        zyx_chunks_orig= data0.chunksize[:-1]
                        zyx_chunks_max= [ min(s,1024) for s in zyx_chunks_orig ]
                        chunks_shape = ( npredictions,*zyx_chunks_max,data0. chunksize[-1] )

                        logging.info(f"data_all shape:{all_shape} chunks_shape:{chunks_shape}")

                        # (ipred, iz,iy,ix, ilabel) , 5dim
                        data_all=da.zeros(all_shape, chunks=chunks_shape , dtype=data0.dtype)

                    data_all[i,:,:,:,:]=data0

                bcomplete=True
            except Exception as exc0:
                logging.info("Allocation failed with dask. Returning None")
                logging.info(f"Exception type:{type(exc0)}")
                logging.info("Exception string:", str(exc0))
                data_all=None
        
        return data_all
    
    def NN1_predict_standard(self,data_vol, pred_file_h5_out):
        """
        Does the volume segmantics predictions using its 'standard' way, with a single volume output
        
        Params:
            data_vol: path to file or ndarray. Single, not a list
            pred_file_h5_out: a string with the location of where to drop results in h5 file format *.h5

        Returns:
            nothing. Output result should be saved in filename pred_file_h5_out
        
        """

        logging.debug("NN1_predict_standard()")

        if self.model_NN1_path is None:
            logging.error("self.model_NN1_path is None. Exiting")
            return None

        logging.info(f"pred_file_h5_out: {pred_file_h5_out}")

        volseg2pred_m = VolSeg2DPredictionManager(
                model_file_path= self.model_NN1_path,
                data_vol=data_vol,
                settings=self.NN1_pred_settings,
                #use_dask=True
                )
        
        pred_file_h5_out_path = Path(pred_file_h5_out)
        volseg2pred_m.predict_volume_to_path(pred_file_h5_out_path)

        logging.info(f"Prediction completed, saved to file: {pred_file_h5_out}")

    @staticmethod
    def _squeeze_merge_vols_by_max_prob( probs2, labels2):
        #Code from volumesegmantics that merges predictions
        logging.debug("_merge_vols_by_max_prob()")
        max_prob_idx = np.argmax(probs2, axis=0)
        max_prob_idx = max_prob_idx[np.newaxis, :, :, :]
        probs2[0] = np.squeeze(
            np.take_along_axis(probs2, max_prob_idx, axis=0)
        )
        labels2[0] = np.squeeze(
            np.take_along_axis(labels2, max_prob_idx, axis=0)
        )
        return
        

    def NN1_predict_extra_from_last_prediction(self, do_vspred=False, do_cs=False):
        """
        Does the volume-segmantics predictions and consistency score calcualation
        and returns a dictionary with the results.
        Uses the temporary output files from last prediction to work out the new prediction.

        If trying to do vspred and consistency score at the same time and this routine uses too much RAM
        try to execute one at the time

        TODO: option to use dask for handling data
        
        Params:
            do_vspred: calculate the predictions in 'standard' colume-segmantics way
            do_cs: calculate consistency score using the probabilities

        Returns:
            A dictionary
                'vspred' : predicted volume-segmantics
                'cs': the consistency score volume
        """

        logging.debug("NN1_predict_extra_from_last_prediction()")

        labels_vs_2stack=None
        probs_vs_2stack=None

        if do_cs: 
            from . import ConsistencyScore
            consistencyscore0 = ConsistencyScore.cConsistencyScoreMultipleWayProbsAccumulate()

        # Uses pandas list created with location of files
        for i,prow in tqdm.tqdm(self.all_nn1_pred_pd.iterrows(), total=self.all_nn1_pred_pd.shape[0]):

            if do_vspred or do_cs:
                pred_prob_filename = prow['pred_data_probs_filenames']
                pred_probs = read_h5_to_np(pred_prob_filename)

            if do_vspred:
                pred_labels_filename = prow['pred_data_labels_filenames']
                pred_labels = read_h5_to_np(pred_labels_filename)

                #Squeeze all probabilities along class dimension to maximum
                probs_class_squeezed = np.max(pred_probs, axis=pred_probs.ndim-1)

                if labels_vs_2stack is None:
                    logging.debug("First labels and probs file initializes")
                    shape_tup = pred_labels.shape
                    labels_vs_2stack = np.empty((2, *shape_tup), dtype=pred_labels.dtype)
                    probs_vs_2stack = np.empty((2, *shape_tup), dtype=pred_probs.dtype)
                    labels_vs_2stack[0]=pred_labels
                    probs_vs_2stack[0]=probs_class_squeezed
                else:
                    labels_vs_2stack[1]=pred_labels
                    probs_vs_2stack[1]=probs_class_squeezed
                    self._squeeze_merge_vols_by_max_prob(probs_vs_2stack,labels_vs_2stack)
            
            if do_cs:
                consistencyscore0.accumulate(pred_probs)
        
        vspred=None
        cs=None

        if do_vspred:
            vspred=labels_vs_2stack[0]

        if do_cs:
            cs = consistencyscore0.getCScore()
        
        return {'cs':cs, 'vspred':vspred}



    @staticmethod
    def copy_from(lgsegm0):
        """ create new lgsegmentor object with properties being a copy of
        existing lgsegmentor object
        """
        import copy

        lgsegm1 = cMultiAxisRotationsSegmentor()

        lgsegm1.model_NN1_path = lgsegm0.model_NN1_path
        lgsegm1.chunkwidth = lgsegm0.chunkwidth 
        lgsegm1.nlabels=lgsegm0.chunkwidth 
        lgsegm1.temp_data_outdir=lgsegm0.temp_data_outdir
        lgsegm1.cuda_device=lgsegm0.cuda_device

        lgsegm1.NN1_train_settings = copy.deepcopy(lgsegm0.NN1_train_settings)
        lgsegm1.NN1_pred_settings = copy.deepcopy(lgsegm0.NN1_pred_settings)

        lgsegm1.NN2_settings = copy.deepcopy(lgsegm0.NN2_settings)

        lgsegm1.labels_dtype = lgsegm0.labels_dtype

        lgsegm1.all_nn1_pred_pd=None
        if not lgsegm0.all_nn1_pred_pd is None:
            lgsegm1.all_nn1_pred_pd = lgsegm0.all_nn1_pred_pd.copy()

        lgsegm1._nn1_model_temp_dir = lgsegm0.nn1_model_temp_dir
        lgsegm1.model_NN1_path= lgsegm0.model_NN1_path

        
        lgsegm1.NN2= copy.deepcopy(lgsegm0)

        return lgsegm1


    def dice_loss_np(y_true, y_pred): #old, not used anymore
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        return 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

    def cleanup(self):
        try:
            self._tempdir_pred.cleanup()
        except:
            pass

        try:
            self._pytorch_model_tempdir.cleanup()
        except:
            pass
        
        try:
            self._nn1_model_temp_dir.cleanup()
        except:
            pass

    #Context manager, to clean resources such as tempfiles efficiently
    # and provide a way for using `with` statements
    def __exit__(self, *args):
        self.cleanup()
    
    def __enter__(self):
        #Required to run with `with`
        return self