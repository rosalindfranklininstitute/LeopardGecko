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

class cConsistencyScoreProbsAccumulate(): 
    def __init__(self):
        self.clear()

    def accumulate(self,data):
        if self.probs_accum is None:
            self.probs_accum = data.copy()
        else:
            self.probs_accum= self.probs_accum+data
        self.count+=1

    def clear(self):
        self.probs_accum=None
        self.count=0

    def getCScore(self):
        # uses formula (P0avg- 1/Nc)^2 + (P1avg-1/Nc)^2 +...
        # with P0avg being the probablility of class 0 averaged across the different predictions
        # Returns a volume data
        # Assumes the last axis is for the different labels

        # axis 0: different classes
        #Calculates the average probabilities
        Nc = self.probs_accum.shape[-1] #number of classes
        
        logging.debug(f"getCScore(), count:{self.count}, Nc:{Nc}")

        prob_mean = self.probs_accum/ float(self.count)

        prob_sq = np.power(prob_mean, 2)

        #axis=3?
        ax0= len(self.probs_accum.shape)-1
        normc = 1/(1-1/float(Nc))**2
        cscore = normc* ( np.sum(prob_sq, axis=ax0) + (2.0*np.sum(prob_mean,axis=ax0)+1.0)/float(Nc) )

        return cscore




