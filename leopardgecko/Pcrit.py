import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import dask.array as da

from .metrics import *

import logging


class MultiClassMultiWayPredictOptimizer:
    '''
    This class contains functions that can optimize choice of per-voxel multi-class segmentation
    from nways (typically 12) predictions.
    
    This code was tested only on random data
    It needs to be tested with real data and respective ground truth

    This code was prototyped in
    /workspace/for_luis/Programming/Tests/TestMultiClassMultiWayPredictOptimizer.ipynb
    '''

    def getSegmentationProjMatrix(self, hvector ):
        '''
        Returns the alpha matrix with the hvector projected to the 'sides' (svector) of the multidimensional triangle
        
        Return matrix (called alpha) has elements alpha[i,j],
        corresponding to the projected hvector onto
        the svector [ 0 , ..., +nways (j element) , 0 , -nways (i element),...]
        This vector is basically one of the sides of the hypersurface triangle.

        The hvector is the hypervector. For example:
        In a 12way prediction with 3 semgmentations, each voxel will have several
        12 different predictions, with
        n0 being the number of predictions for class 0
        n1 being the number of predictions for class 1
        and n2 being the number of predictions for class 2

        Because there are 12ways, then n0+n1+n2 = 12
        The hypervector is the vector (v0,v1,v2)
        and is called hyper, because its components are such that they part of the
        hypersurface defined by the constrain n0+n1+n2=12
        Gets the projected hypervector hvector onto each between-segmentations vector
        The 2-dimensional matrix represents all the binary combinations,
        with the value being after projecting hvector

        This function is mostly used internally to help calculation and optimization of pcrit.

        '''

        alpha = np.zeros((self.nclasses,self.nclasses))
        svector = np.zeros(self.nclasses)

        for segm0 in range(self.nclasses):
            #svector0 = np.zeros(nclasses)
            #svector0[segm0] = -nways
            for segm1 in range(self.nclasses):
                value=0 #Default result value
                if segm0 != segm1 :
                    svector = np.zeros(self.nclasses)
                    svector[segm1]= self.nways
                    svector[segm0] = -self.nways
                    #svector should be s1 - s0 (vectors) 
                    
                    #print(f"hvector={hvector} ; svector={svector}")
                    
                    value = np.dot(np.array(hvector), svector )
                

                alpha[segm0,segm1] = value
        
        return alpha


    def __init__(self, nclasses, nways, useBckgnd = False):
        #Initialises table of ProjMatrices for speed calculations


        self.nclasses = nclasses
        self.nways= nways

        #self.p_ProjMatrix=None
        #self.CSegmProjMatrix0= self.CSegmProjMatrix(nclasses, nways)

        self.pcrit = None
        self.from_hvect_to_segm_dict = None

        self.useBckgnd = useBckgnd

    def isValidHvector(self, hvector):
        '''
        Check if a given hvector is valid
        '''
        h0 = np.array(hvector)
        if self.nways == h0.sum() and h0.ndim==1 and h0.shape[0]==self.nclasses:
            return True
        
        return False

    def set_pcrit(self, pcrit):
        '''
        Sets the hypervector pcrit.
        This is the criteria that is used to decide which class a given hvector belongs to.
        This pcrit is a hvector itself.

        '''
        
        #self.p_ProjMatrix= self._cProjMatrices.getProjMatrixForHypervector(pcrit)
         #this will be useful to get the matrices for evaluating the class

        #Rather than doing this above create a dictionary that relates any possible hvector to
        #the calculated segmentation

        if not self.isValidHvector(pcrit):
            logging.error(f"pcrit = {pcrit} is not a valid hvector")
            self.pcrit=None
            return

        self.pcrit = pcrit

        #Gets all possible values of hvector
        all_hvectors= self.getCombinations()  #Returns a standard python array

        self.from_hvect_to_segm_dict = {}

        for hvector0 in all_hvectors:
            _class0 = self.getClassNumber(hvector0) #Will automatically add all entries to the dictionary

            #Adds new element to dictionary
            #self.from_hvect_to_segm_dict[hvector0] = class0

    
    def getClassNumber(self, hvector_tuple):
        '''
        Given a (hyper)vector v gets the class number
        Uses the dictionary if available
        If value not available then calculate new one
        
        '''

        nclass0=None

        if not self.from_hvect_to_segm_dict is None:
            #Check if -hvector v is available
            if hvector_tuple in self.from_hvect_to_segm_dict:
                #get the class value
                nclass0 = self.from_hvect_to_segm_dict[hvector_tuple]

        if  nclass0 is None:
            #Calculate class and
            #Add new element to dictionary
            Pm = self.getSegmentationProjMatrix(self.pcrit)
            Vm = self.getSegmentationProjMatrix(hvector_tuple)

            Cm = Vm - Pm #Compare matrix

            row_maxs = np.amax(Cm , axis=1) #Need to check the axis is correct
            #print(f"row_maxs = {row_maxs}")
            
            #print(f"len(row_maxs) = {len(row_maxs)}")
            #First row that <=0 sets as the identified class
            for rown in range(len(row_maxs)):
                #print(f"rown = {rown}")
                if row_maxs[rown]<=0 :
                    #res=rown
                    #print(f"res = {res}")
                    #return rown
                    nclass = rown
                    break
            
            self.from_hvect_to_segm_dict[hvector_tuple] = nclass

        return nclass0

    def identifiyClassFromVols(self, vols):
        '''
        Identifies the class for each voxel in the volumes using the pcrit established.
        Assumes that:
            vols first index corresponds to the segmentation class nway score
            Each voxel has numbers 0 to nways

        '''

        logging.info("identifiyClassFromVols()")

        vols_shape = vols.shape
        
        if len(vols_shape) != 4:
            print("vols is not 4-dimensional. exiting with None")
            return None
        
        #nways= p.sum() #infers nways
        #print(f"nways from p_criteria = {nways}")
        #nclasses = vols_shape[0] #infers nclasses from the shape of the first index
        #print(f"nclasses from vols_shape[0] = {nclasses}")
        
        #Initialise values to -1 (=no class identified for each voxel)
        classid_vol = np.full( (vols.shape[1], vols.shape[2], vols.shape[3]) , -1 , dtype=np.int8)
        
        for iz in range(vols_shape[1]):
            #logging.info(f"iz={iz} / {vols_shape[1]}")
            for iy in range(vols_shape[2]):
                #logging.info(f"iy={iy} / {vols_shape[2]}")
                for ix in range(vols_shape[3]):
                    #logging.info(f"ix={ix}")
                    v = vols[:, iz,iy,ix]

                    #logging.info(f"v={v}")

                    #print(f"v= {v}")
                    v0 = tuple(v) #Convert to tuple
                    classid_vol[iz,iy,ix] = self.getClassNumber(v0)
                    #logging.info(f"classid_vol[iz,iy,ix]= {classid_vol[iz,iy,ix]}")
        
        return classid_vol

    METRICACCURACY=0
    METRICDICE=1


    def getMetricScoreWithPcritFromClassVols(self, a_all, gt_rnd , metric=METRICDICE):
        '''
        Get the score metric considering the p-criteria pgrad given
        and between two vols-with-all-classes a_all and ground-truth gt_rnd

        Returns the value of the metric and the 'classed' volume
        '''
        logging.info("getMetricScoreWithPcritFromClassVols()")

        classed_vol = self.identifiyClassFromVols(a_all)
        
        score=None

        if metric == self.METRICDICE:
            logging.info(f"metric= Dice , useBckgnd = {self.useBckgnd}")
            score = MetricScoreOfVols_Dice(classed_vol, gt_rnd, self.useBckgnd)
        elif metric == self.METRICACCURACY:
            logging.info("metric= Accuracy")
            score = MetricScoreOfVols_Accuracy(classed_vol, gt_rnd)
        

        return score , classed_vol


    def getCombinations(self):
        '''
        Gets all the possible integer-value combinations for pcriteria value
        from nways and nclasses.
        Returns a list with all the possible values
        '''
        def _getCombinations(inextdim,pbase):
            #plist_ret = np.zeros(nclasses)
            plist_ret = [] #python list (not numpy)
            #print(f"inextdim = {inextdim}")
            if inextdim==self.nclasses-1:
                #Last index
                #plist_ret.append( nways - pbase.sum())
                pbase1= np.copy(pbase)
                pbase1[inextdim] = self.nways - pbase.sum()
                #plist_ret= [pbase1]
                plist_ret= [tuple(pbase1)]
                #print(f"Last index; plist_ret={plist_ret}")
            else:
                for i in range(0, int(self.nways-pbase.sum()+1) ):
                    #print(f"for inextdim = {inextdim} ; i={i} ")
                    pbase1= np.copy(pbase)
                    pbase1[inextdim] = i
                    inextdim0 = inextdim+1
                    plist = _getCombinations(inextdim0, pbase1)
                    #print(f"for inextdim = {inextdim} ; i={i} ; plist={plist}")

                    plist_ret.extend(plist)
                    #plist_ret.append(plist)

            #print (f"plist_ret = {plist_ret}")
            return plist_ret
        
        pbase0 = np.zeros(self.nclasses, dtype=int)
        pcomb0= _getCombinations(0, pbase0)
        
        return pcomb0
    
    def getPCritForMaxMetric(self, a_all0, gt_rnd0, metric=METRICDICE, savemetricdatafile=None):
        '''
        Determines which value of pcrit gives the best score Accuracy.
        Returns the pcrit hypervector and the maximumly determined metric value 

        Parameters
            a_all0: array (4D) with first index corresponding to the class segment index and the remaining 3
                indexes for the volume (int format)
            gt_rnd0: single 3D volume with the ground truth. Contains int values that represent segmentation class
            nways: number of ways that the prediction was made
            nclases: number of segmentation classes
        '''
        logging.info("getPCritForMaxMetric()")

        #Do for all interger combinations of pgrad
        pvalues0= self.getCombinations()
        logging.info(f"pvalues0 = {pvalues0}")

        #vfunc_getAcc = np.vectorize( getMetricAccuracyFromClassVolsAndPcrit , excluded=['a_all', 'gt_rnd'] )
        #accvalues= vfunc_getAcc(a_all=a_all0 , gt_rnd=gt_rnd0 , pvalues0)
        #Does not work
        
        npvalues = len(pvalues0)
        logging.info(f"npvalues = {npvalues}")

        metricvalues = np.zeros(npvalues)
        
        #savetext=""
        
        for i in range(npvalues):
            logging.info(f"i = {i} , pvalue = {pvalues0[i]}")

            self.set_pcrit(pvalues0[i])

            #metricvalues[i] =  self.getMetricScoreWithPcritFromClassVolsAndPcrit(a_all0, gt_rnd0 , pvalues0[i], metric)
            metricvalues[i] , _ =  self.getMetricScoreWithPcritFromClassVols(a_all0, gt_rnd0 , metric)

            logging.info(f"metricvalue = {metricvalues[i]}")

            if savemetricdatafile is not None:
                #savetext = savetext+ f"{pvalues0[i]} , {metricvalues[i]} \n"
                saveline= f"i={i} , p={pvalues0[i]} , metric={metricvalues[i]}\n"

                with open(savemetricdatafile, "a") as myfile:
                    myfile.write(saveline)

        #print (accvalues)
        
        imax = np.argmax(metricvalues)
        
        p_maxscore= pvalues0[imax]

        #gets the volume that gives max metric
        self.set_pcrit(p_maxscore)
        max_metric_score, classedvol_pmax= self.getMetricScoreWithPcritFromClassVols(a_all0, gt_rnd0 , metric)
        
        if savemetricdatafile is not None:
            #savetext = savetext+ f"{pvalues0[i]} , {metricvalues[i]} \n"
            saveline= f"pvalue for max metric= {p_maxscore} , with score = {max_metric_score}"

            with open(savemetricdatafile, "a") as myfile:
                myfile.write(saveline)

        return p_maxscore, max_metric_score , classedvol_pmax


    def getOptimizedHvectorsToClassForMaxDiceMetric(self, a_all0, gt_rnd0):
        '''

        Uses a different method to calculate the hvector_to_class by doing a per-hvector optimization
        Each h-vector will be counted in the appearence in the data volumes.
        This is counting will be separated by-per ground-truth class.

        Parameters
            a_all0: input arrays (4D) with first index corresponding to the class segment index and the remaining 3
                indexes for the volume (int format)
            gt_rnd0: single 3D volume with the ground truth. Voxels are int values that represent segmentation class
            nways: number of ways that the prediction was made
            nclasses: number of segmentation classes
        '''

        logging.info("getOptimizedHvectorsToClassForMaxDiceMetric()")

        #Do for all integer combinations of pgrad
        hvectors0= self.getCombinations()
        logging.info(f"hvectors0 = {hvectors0}")

        #Initialise the counter
        hvect_gndclass_counter = {}
        for h0 in hvectors0:
            for c0 in range(self.nclasses):
                hvect_gndclass_counter[ (h0,c0) ] = 0

        vols_shape = a_all0.shape
        #Count voxel-by-voxel
        for iz in range(vols_shape[1]):
            for iy in range(vols_shape[2]):
                for ix in range(vols_shape[3]):
                    v = a_all0[:, iz,iy,ix]
                    h1 = tuple(v) #Convert to tuple to get the h-vector

                    gnd0 = gt_rnd0[iz,iy,ix]

                    #increment dict counter
                    hvect_gndclass_counter[ (h1,gnd0) ] += 1

        print("hvect_gndclass_counter")
        print(hvect_gndclass_counter)
        #TODO

                    
        return None