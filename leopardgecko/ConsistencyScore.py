'''
Copyright 2023 Rosalind Franklin Institute

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

# Formulas to calculate consistency score

import numpy as np
import logging

#Normalisation, to be used to ensure that CS scores range from 0 to 1
normc = lambda Nc: float(Nc)/ (float(Nc)-1.0)

class cConsistencyScoreMultipleWayProbsAccumulate():
    '''
    Utility class to calculate Consistency score from multiple way volume data
    Data must be in format (z,y,x, class)
    and is provided one by one, one "way" at the time, using accumulate() function

    Function getCScore() gets the CS score in a volume heatmap format (z,y,x)

    The formula used is given in getScore docstring

    This method of using class to calculate the consistency score differs from
    the function getCScoreFromMultipleWayProbsPred() in the way that uses less RAM
    footprint. getCScoreFromMultipleWayProbsPred() requires a 5dim array, whereas
    this class requres a 4dim array to be provided each time. Results obtained by
    either method are equivalent.

    '''

    def __init__(self):
        self.clear()

    def accumulate(self,data):
        '''
        Provide data from each prediction, one by one
        Data must be in format (z,y,x, class)
        '''

        logging.debug(f"accumulate(), data.shape:{data.shape}, data.dtype:{data.dtype}")

        if self.probs_accum is None:
            self.probs_accum = data.copy().astype(np.float32)
        else:
            self.probs_accum= self.probs_accum+data.astype(np.float32)
        self.count+=1

    def clear(self):
        logging.debug(f"clear()")
        self.probs_accum=None
        self.count=0

    def getCScore(self):
        '''
        Calculates consistency score from accumulated probabilities data using
        formula (P0avg- 1/Nc)^2 + (P1avg-1/Nc)^2 +...
        with P0avg being the probablility of class 0 averaged across the different predictions

        Returns a volume data with the consistency score per z,y,x coordinate
        
        Assumes the last axis is for the different labels
        '''
        # axis 0: different classes
        #Calculates the average probabilities
        Nc = self.probs_accum.shape[-1] #number of classes
        
        logging.debug(f"getCScore(), count:{self.count}, Nc:{Nc}")

        prob_mean = self.probs_accum/ float(self.count)

        prob_sq = np.power(prob_mean, 2)

        #axis=3?
        ax0= prob_mean.ndim-1 #gets last index
        #normc = 1/(1-1/float(Nc))**2
        cscore = normc(Nc)* ( np.sum(prob_sq, axis=ax0) + (1.0-2.0*np.sum(prob_mean,axis=ax0))/float(Nc) )

        return cscore.astype(np.float32)


def getCScoreFromMultipleWayProbsPred(data_way_probs):
    '''
    Calculates consistency score from data_way_probs
    

    Assumes that first dimension is the "way" index
    followed by z,y,x coordinates, and then the class dimension
    (ways, z,y,x, class)

    Could use class cConsistencyScoreProbsAccumulate to calculate
    but rather use the formula directly
    '''

    Nways= data_way_probs.shape[0]
    Nc = data_way_probs.shape[-1] #number of classes

    logging.debug(f"getCScoreFromAllProbsData(), Nways:{Nways} , Nc:{Nc}")
    print(f"getCScoreFromAllProbsData(), Nways:{Nways} , Nc:{Nc}")

    ax0= data_way_probs.ndim-1
    # normc = 1/(1-1/float(Nc))**2

    prob_mean = np.mean(data_way_probs, axis=0) #across all ways
    prob_sq = np.power(prob_mean, 2)

    ax0= len(prob_mean.shape)-1

    cscore = normc(Nc)* ( np.sum(prob_sq, axis=ax0) + (1.0-2.0*np.sum(prob_mean,axis=ax0))/float(Nc) )

    return cscore.astype(np.float32)


def getCScoreFromMultipleWayLabelsPred(data_way_labels, nclasses=None):
    '''
    Calculates consistency score from data_way_labels
    
    Assumes that first dimension is the "way" index
    followed by z,y,x coordinates. Data value refers to the label identified.
    (ways, z,y,x)

    The total number of classes should be provided. If none is given it will
    use the maximum but it will run slower

    '''

    Nways= data_way_labels.shape[0]
    if nclasses is None:
        nclasses = np.max(data_way_labels)+1
    
    logging.debug(f"getCScoreFromAllLabelsData(), Nways:{Nways} , nclasses:{nclasses}, datshape:{data_way_labels.shape[1:]}")

    Nc=nclasses

    #Handles each class label seperately
    data_xyz_label=np.zeros( (*data_way_labels.shape[1:], Nc))

    for ilabel in range(Nc):
        data_label0= np.where(data_way_labels==ilabel, 1, 0) #Marks the selected class with ones

        #Adds the occurences across the "ways"
        data_xyz_label[...,ilabel] = np.sum(data_label0, axis=0)
    
    # At this point data_xyz_label should have the number of voxels for the selected label

    ax0= data_xyz_label.ndim-1
    # normc = 1/(1-1/float(Nc))**2

    fract_sq = np.power( (data_xyz_label/ Nways), 2)

    cscore = normc(Nc)* ( np.sum(fract_sq, axis=ax0) + (1.0-2.0/Nways*np.sum(data_xyz_label,axis=ax0))/float(Nc) )

    return cscore.astype(np.float32)
