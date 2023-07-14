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

import h5py
import dask.array as da
import os
import logging

class PredictedData:
    '''
    Class to handle multiply-predicted combined data typically 12-way combined binary predictions (Olly's 2D unet output)
    '''
    def __init__(self , filename):
        self.filename = filename
        self.data_da = None #Default
        self.vmax = None
        self.vmin = None
        self.weightedData_da = None
        self.weightedDataAvgValue = None

        #Open file and do some processing
        
        #Check file has h5 extension
        fileroot , fileext = os.path.splitext(filename)
        
        if fileext == ".h5":
            #Try to open the file as hdf5 to a dask array object
            #Opens file
            fx= h5py.File(filename,'r')
            self.data_da = da.from_array(fx['data'], chunks='auto')
            logging.info("File " + filename + " opened successfully as hdf5 file.")

        if self.data_da is not None :
            #Gets vmax and vmin
            self.vmax = da.max( self.data_da ).compute()
            self.vmin = da.min( self.data_da ).compute()
            logging.info("vmax=" + str(self.vmax) + ", vmin=" + str(self.vmin))
    

    WEIGHTMETHOD_MAXMINSQUARE = 'MaxMinSquare'
    WEIGHTMETHOD_MAXZEROSQUARE = 'MaxZeroSquare'
    WEIGHTMETHOD_NONE = 'None'

    def setWeightedDataMethod(self, method='MaxMinSquare'):
        '''
        Weights the data.
        Typically used to get a consistency score from combined predicted data.
        It also returns the average value over the whole volume.
        It may be useful for reporting a score for the whole data.
        '''

        #Default return values
        self.weightedData_da = None
        self.weightedDataAvgValue = None

        #data_da_weighted=None
        if self.data_da is not None:
            #Use square function with minimum at the value halfway between vmax and vmin
            if method == self.WEIGHTMETHOD_MAXMINSQUARE:
                vaverage = 0.5*( self.vmax - self.vmin)
                self.weightDataSquareAtX(vaverage)
                #self.weightedData_da = da.square(self.data_da - vaverage) #weighting values
                #print ("data_da_weighted.shape = ", data_da_weighted.shape)
                #self.weightedDataAvgValue  = da.average( self.weightedData_da ).compute()
            elif method == self.WEIGHTMETHOD_MAXZEROSQUARE:
                x0 = self.vmax / 2.0
                self.weightDataSquareAtX(x0)
            elif method == 'None':
                self.weightedData_da = self.data_da
            
            self.weightedDataAvgValue  = da.average( self.weightedData_da ).compute()
            
            return self.weightedData_da , self.weightedDataAvgValue

    def weightDataSquareAtX(self, x0):
        self.weightedData_da = da.square(self.data_da - x0)
        self.weightedDataAvgValue  = da.average( self.weightedData_da ).compute()
        return self.weightedData_da , self.weightedDataAvgValue

    def getWeightedValueAverageOfVolume(self, coordsTuple):
        zmin, zmax , ymin,ymax, xmin,xmax  = coordsTuple
        #It does not use the score data from the average pooling
        #But uses the actual combined data (predicted data)
        vol_da = self.weightedData_da[zmin:zmax, ymin:ymax , xmin:xmax]
        wavg = da.average(vol_da).compute()
        return wavg
        
    @staticmethod
    def SelfTest():
        #TODO: Write some code here that tests this class
        return 0
