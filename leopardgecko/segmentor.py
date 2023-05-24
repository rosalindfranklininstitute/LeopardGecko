import numpy as np
import matplotlib.pyplot as plt
import dask.array as da

#import subprocess
import tempfile

import h5py
def numpy_from_hdf5(path):
    with h5py.File(path, 'r') as f:
        data = f['/data'][()]
    return np.array(data)

from pathlib import Path

import os
cwd = os.getcwd()
print(cwd)

import tempfile
import logging
from types import SimpleNamespace

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

def save_data_to_hdf5(data, file_path, internal_path="/data", chunking=True):
    logging.info(f"Saving data of shape {data.shape} to {file_path}.")
    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            "/data", data=data, chunks=chunking, compression=cfg.HDF5_COMPRESSION
        )
def read_h5_to_np(file_path):
    with h5py.File(file_path,'r') as data_file:
        data_hdf5=np.array(data_file['data'])
        
    return data_hdf5

class cMultiAxisRotationsSegmentor():

    def __init__(self, models_prefix="lg_segmentor_model_"):
        model_NN1_fn = models_prefix+"NN1.pytorch"
        model_NN2_fn = models_prefix+"NN2.pk"
        
        self.model_NN1_path = Path(cwd,model_NN1_fn)
        self.model_NN2_path = Path(cwd, model_NN2_fn)
    
    def train(self, traindata, trainlabels, get_metrics=True):
        """
        Train NN1 (volume segmantics) and NN2 (MLP Classifier)
        """

        # logging.basicConfig(
        #     level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
        # )
        
        #Train NN1
        self.NN1_train(traindata, trainlabels)

        #Does the multi-axis multi-rotation predictions
        # and collects data files
        tempdir_pred= tempfile.TemporaryDirectory()
        tempdir_pred_path = Path(tempdir_pred.name)
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

        #Take this oportunity to calculate metrics of each prediction
        
        all_shape = ( len(pred_data_probs_filenames), *data0.shape )
        data_all = np.zeros( all_shape)
        #Fill with data
        data_all[0,:,:,:]= data0
        for i in range(1,len(pred_data_probs_filenames)):
            print(i)
            data_i = read_h5_to_np(pred_data_probs_filenames[i])
            data_all[i,:,:,:]=data_i


        nn1_acc_dice_s= []
        if get_metrics:
            for i, label_fn0 in enumerate(pred_data_labels_filenames):
                data_i = read_h5_to_np(label_fn0)
                a0 =  metrics.MetricScoreOfVols_Accuracy(data_i,trainlabels)
                d0 = metrics.MetricScoreOfVols_Dice(data_i,trainlabels)
                nn1_acc_dice_s.append( [a0,d0])
                print(f"prediction:{i} , filename: {label_fn0}, accuracy:{a0}, dice:{d0}")


        #Train NN2 from multi-axis multi-angle predictions against labels (gnd truth)
        nn2_acc, nn2_dice = self.NN2_train(data_all, trainlabels, get_metrics=get_metrics)

        tempdir_pred.cleanup()

        return nn1_acc_dice_s, (nn2_acc, nn2_dice)

    def NN2_train(self, train_data_all_probs, trainlabels, get_metrics=True):
                
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
        self.NN2 = MLPClassifier(hidden_layer_sizes=(10,10), random_state=1, activation='tanh', verbose=True, learning_rate_init=0.001,solver='sgd', max_iter=1000)

        #Do the training here
        self.NN2.fit(X_train,y_train)

        print(f"NN2 train score:{self.NN2.score(X_train,y_train)}")

        if get_metrics:
            print("Preparing to predict the whole training volume")
        
            d_prediction= self.NN2_predict( train_data_all_probs)

            #Get metrics
            nn2_acc= metrics.MetricScoreOfVols_Accuracy(trainlabels,d_prediction)
            nn2_dice= metrics.MetricScoreOfVols_Dice(trainlabels,d_prediction, useBckgnd=False)

            print(f"NN2 acc:{nn2_acc}, dice:{nn2_dice}")
        
            return nn2_acc, nn2_dice
        
        return None, None


    def NN2_predict(self, data_all_probs):
        
        #Need to flatten along the npred and nclasses
        data_2MLP_t= np.transpose(data_all_probs,(1,2,3,0,4))

        dsize = data_2MLP_t.shape[0]*data_2MLP_t.shape[1]*data_2MLP_t.shape[2]
        inputsize = data_2MLP_t.shape[3]*data_2MLP_t.shape[4]

        data_2MLP_t_reshape = np.reshape(data_2MLP_t, (dsize, inputsize))

        mlppred = self.NN2.predict(data_2MLP_t_reshape)

        #Reshape back to 3D
        mlppred_3D = np.reshape(mlppred, data_2MLP_t.shape[0:3])

        return mlppred_3D
    

    def NN1_train(self, traindata, trainlabels):
        tempdir_data = tempfile.TemporaryDirectory()
        tempdir_data_path=Path(tempdir_data.name)
        print(f"tempdir_data_path:{tempdir_data_path}")

        tempdir_seg = tempfile.TemporaryDirectory()
        tempdir_seg_path = Path(tempdir_seg.name)
        print(f"tempdir_seg_path:{tempdir_seg_path}")

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

        settings = SimpleNamespace(**settings0)

        # Keep track of the number of labels
        max_label_no = 0
        label_codes = None

        # Set up the DataSlicer and slice the data volumes into image files
        slicer = TrainingDataSlicer(traindata, trainlabels, settings)
        data_prefix, label_prefix = "",""
        slicer.output_data_slices(tempdir_data_path, data_prefix)
        slicer.output_label_slices(tempdir_seg_path, label_prefix)
        if slicer.num_seg_classes > max_label_no:
            max_label_no = slicer.num_seg_classes
            label_codes = slicer.codes

        # Set up the 2dTrainer
        self.trainer = VolSeg2dTrainer(tempdir_data_path, tempdir_seg_path, max_label_no, settings)
        # Train the model, first frozen, then unfrozen
        num_cyc_frozen = settings.num_cyc_frozen
        num_cyc_unfrozen = settings.num_cyc_unfrozen
        #model_type = settings.model["type"].name

        if num_cyc_frozen > 0:
            self.trainer.train_model(
                self.model_NN1_path, num_cyc_frozen, settings.patience, create=True, frozen=True
            )
        if num_cyc_unfrozen > 0 and num_cyc_frozen > 0:
            self.trainer.train_model(
                self.model_NN1_path, num_cyc_unfrozen, settings.patience, create=False, frozen=False
            )
        elif num_cyc_unfrozen > 0 and num_cyc_frozen == 0:
            self.trainer.train_model(
                self.model_NN1_path, num_cyc_unfrozen, settings.patience, create=True, frozen=False
            )
        #trainer.output_loss_fig(model_out)
        #trainer.output_prediction_figure(model_out)
        #plt.show()
        # Clean up all the saved slices
        slicer.clean_up_slices()

        tempdir_data.cleanup()
        tempdir_seg.cleanup()


    def NN1_predict(self,traindata, pred_folder_out):
        """
        
        Does the multi-axis multi-rotation predictions
        and returns predictions filenames of probablilities and labels

        predictions are probabilities (not labels)

        """

        settings1 = {'quality': 'high',
            'output_probs': True,
            'clip_data': True,
            'st_dev_factor': 2.575,
            'data_hdf5_path': '/data',
            'cuda_device': 0,
            'downsample': False,
            'one_hot': False,
            'prediction_axis': 'Z'}

        # TODO: Save also labels?

        settings = SimpleNamespace(**settings1)

        self.volseg2pred = VolSeg2dPredictor(self.model_NN1_path, settings, use_dask=True)

        def save_pred_data(data, axis, rot):
            # Saves predicted data to h5 file in tempdir and return file path in case it is needed
            file_path = f"{pred_folder_out}/pred_{axis}_{rot}.h5"
            save_data_to_hdf5(data, file_path)
            return file_path
        
        data_vol = np.array(traindata) #Copies

        pred_data_probs_filenames=[] #Will store results in files, and keep the filenames as reference
        pred_data_labels_filenames=[]
        for krot in range(0, 4):
            rot_angle_degrees = krot * 90
            logging.info(f"Volume rotated by {rot_angle_degrees} degrees")

            data_vol = np.rot90(np.array(traindata),krot) #Copies

            #Predict 3 axis
            #YX
            logging.info("Predicting YX slices:")
            #returns (labels,probabilities)
            res = self.volseg2pred._predict_single_axis_all_probs(
                data_vol, axis=Axis.Z
            )
            pred_probs = np.rot90(res[1], -krot) #invert rotation before saving
            #Saves prediction labels
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

