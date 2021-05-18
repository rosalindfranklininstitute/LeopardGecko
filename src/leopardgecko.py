#Leopard Gecko module with functions and classes for
# processing, analysing and reporting 'predicted' data

# To be run from the command line or jupyter

#Code based in notebooks
# AvgPooling3DConsistencyData.ipynb and AnalyseAvgPoolResults.ipynb


import torch
import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import time #Needed here?
import dask.array as da
import h5py
import os
import math
import logging
from scipy import optimize


#For showing nested loop progress in notebook
#from IPython.display import clear_output

def lizzie():
    '''
    No code is proper without an Easter Egg
    '''
    print("Lizzie, The greatest Leopard Gecko in the Gecko world. If you ever met her, you would even say she glows.", \
        "She has geckifing powers from her petrifying stare.")

class PredictedData:

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



class ScoreData:
    '''
    Class to help analysing of score data.
    It has also functions to determine and report regions of interest
    '''

    def __init__(self , data3d , zVolCentres, yVolCentres, xVolCentres, origfilename =""):
        self.data3d = data3d
        self.zVolCentres = zVolCentres
        self.yVolCentres = yVolCentres
        self.xVolCentres = xVolCentres
        self.origfilename = origfilename

        self.data3d_vmin= np.amin(data3d)
        self.data3d_vmax= np.amax(data3d)

        self.histogram = None
        self.histogram_bins = None
        #Automatically generate histogram?

    @staticmethod
    def fromFile(filename):
        f = h5py.File(filename,'r')
        data3d =  f['data']
        #print (datahdf5.shape)
        xVolCentres = f['X']
        yVolCentres = f['Y']
        zVolCentres = f['Z']

        origfilename = None
        fkeys = list (f.keys())
        if 'origfilename' in fkeys:
            origfilename = f['origfilename'] #This may throw an error as older versions did not have this parameter
        
        newscoredata= ScoreData(data3d , zVolCentres, yVolCentres, xVolCentres, origfilename)

        return newscoredata

    def saveToFile(self, filename):
        with h5py.File(filename ,'w') as f:
            f['/data']= self.data3d
            f['/Z']= self.zVolCentres
            f['/Y']= self.yVolCentres
            f['/X']= self.xVolCentres
            f['origfilename'] = self.origfilename
    
    def GetIndicesFromOrigDataWithScoreBetween(self, cscore_low, cscore_high):
        ''' Get indices of original data where the consistency score is betwen cscore_low and cscore_high
        '''
        
        #This is needed to prevent errors when using slider (native Python float values to numpy.float)
        cscore_low_np = np.float64(cscore_low)
        cscore_high_np = np.float64(cscore_high)
        
        data = self.data3d

        indexes = np.where( (data >= cscore_low_np) & (data <= cscore_high_np ) )
        #Values will be in format (example)
        # (array([52, 53, 54, 55, 56]), array([0, 0, 0, 0, 0]), array([283, 283, 283, 283, 283]))
        
        #Convert these indices to the original data indices
        i_zip = list(zip(indexes[0],indexes[1],indexes[2]))
        #This will convert to format (example)
        # [(52, 0, 283), (53, 0, 283), (54, 0, 283), (55, 0, 283), (56, 0, 283)]
        
        #print (i_zip)
        i_conv=[]
        c_score = []
        #Convert indices to the ones from original data
        for e1 in i_zip:
            #print (e1)
            iz = self.zVolCentres[ e1[0] , e1[1] , e1[2] ]
            iy = self.yVolCentres[ e1[0] , e1[1] , e1[2] ]
            ix = self.xVolCentres[ e1[0] , e1[1] , e1[2] ]

            i_conv.append( (iz,iy,ix))
            c_score.append( data[ e1[0] , e1[1] , e1[2] ] ) #The actual cons-score is also returned
        
        
        return i_conv, c_score

    def GetIndicesFromOrigDataWithScoreNear(self, cscore, cwidth= 0.1):
        ''' Get indices of original data where the consistency score is between
        cscore-cwidth and and cscore-cwidth
        '''
        cscore_low = cscore-cwidth/2
        cscore_high = cscore+cwidth/2
        
        return self.GetIndicesFromOrigDataWithScoreBetween( cscore_low, cscore_high )

    def getHistogram(self):
        self.histogram , binsedge = np.histogram(self.data3d , bins='auto')
        self.histogram_bins = binsedge[:-1]
        return self.histogram , self.histogram_bins


    def SelectRegionsOfInterest_V0(self , showPlot=True , printReport = True, k_width=256 ):
        if self.histogram is None:
            _a, _b = self.getHistogram()
        
        hist_vmax = np.amax(self.histogram)
        hist_x_vmax = self.histogram_bins [ np.where(self.histogram == hist_vmax)[0] ]
        
        hist_xmin = self.data3d_vmin
        hist_xmax = self.data3d_vmax

        #Choose regions based in the histogram shape
        #Approximate peak with gaussian
        def fgaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp( -0.5 * ((x - mean) / stddev)**2 )

        aguess= np.max(self.histogram)
        mguess = hist_x_vmax[0]
        popt, pcov = optimize.curve_fit(fgaussian, self.histogram_bins , self.histogram, [aguess, mguess, 1.0])

        print("Gaussian fit to peak, parameters amplitude={} , mean={} ,stdev={}".format(popt[0],popt[1],popt[2]))

        hx = []

        hx.append ( (hist_xmax - hist_xmin)/4 + hist_xmin )
        hx.append ( (hist_xmax + hist_xmin)/2.0 )
        hx.append ( popt[1] - 3* popt[2] )  #At mean - 3*stdev (3 sigma)
        hx.append ( popt[1] - popt[2]*2.35482 /2 ) # FHWM = 2.35482*stdev, At FWHM going up
        hx.append ( popt[1] ) #Commonest consistency

        def getClosestlVoxelFromListOfIndices(listindicesandscores, Zc,Yc,Xc):
            #From the list of voxel indices, get the closest voxel to point (Xc, Yc, Zc)
            #Also returns the distance and its score

            listindices = listindicesandscores[0]
            listcscores = listindicesandscores[1]
            
            #First element sets the return result (default)
            if len(listindices) >0:
                voxelresult = listindices[0]
                voxelresult_dist = (voxelresult[0]-Zc)**2 + (voxelresult[1]-Yc)**2 + (voxelresult[2] - Xc)**2
                voxel_cscore = listcscores[0]
                
                if len(listindices)>1:
                    for j0 in range(1, len(listindices)):
                        i0 =  listindices[j0]
                        #print( "i0= " , i0)
                        thisdist = math.sqrt( (i0[0]-Zc)**2 + (i0[1]-Yc)**2 + (i0[2] - Xc)**2 )
                        if thisdist<voxelresult_dist:
                            voxelresult = i0
                            voxelresult_dist = thisdist
                            voxel_cscore = listcscores[j0]
                
                #print ("Closest voxel is ", voxelresult , " with distance ", voxelresult_dist)
                return voxelresult, voxelresult_dist, voxel_cscore
        
        Zcenter = int( (np.amax( self.zVolCentres ) - np.amin( self.zVolCentres )) /2 )
        Ycenter = int( (np.amax( self.yVolCentres ) - np.amin( self.yVolCentres )) /2 )
        Xcenter = int( (np.amax( self.xVolCentres ) - np.amin( self.xVolCentres )) /2 )

        pointsOfInterest=[]

        for hx0 in hx:
            pointsOfInterest.append( \
                getClosestlVoxelFromListOfIndices ( \
                self.GetIndicesFromOrigDataWithScoreNear( hx0, popt[2] / 4 ), \
                Zcenter , Ycenter, Xcenter) )

        # #TODO: This can be simplified in a for loop
        # point5 = 

        # point4 = getClosestlVoxelFromListOfIndices ( \
        #     self.GetIndicesFromOrigDataWithScoreNear( hx4, popt[2] / 4 ), \
        #     Zcenter , Ycenter, Xcenter)
        
        # point3 = getClosestlVoxelFromListOfIndices ( \
        #     self.GetIndicesFromOrigDataWithScoreNear( hx3, popt[2] / 4 ), \
        #     Zcenter , Ycenter, Xcenter)

        # point2 = getClosestlVoxelFromListOfIndices ( \
        #     self.GetIndicesFromOrigDataWithScoreNear( hx2, popt[2] / 4 ), \
        #     Zcenter , Ycenter, Xcenter)

        # point1 = getClosestlVoxelFromListOfIndices ( \
        #     self.GetIndicesFromOrigDataWithScoreNear( hx1, popt[2] / 4 ), \
        #     Zcenter , Ycenter, Xcenter)

        # pointsOfInterest = (point1, point2, point3, point4, point5)

        #Consider removal
        # if printReport:
        #     def point_ReportResultAs_Vol_Center_CScore(p3d , width):
        #         w2 = int(width/2)
                
        #         p = p3d[0] #First element is the voxel coordinates
        #         point_dist = p3d[1] #Second element is the distance to...
        #         point_cscore = p3d[2] #Second element is the consistency score
                
        #         s = "[ " + str(p[0]-w2) + ":" + str(p[0]+w2) + " , " + \
        #             str(p[1]-w2) + ":" + str(p[1]+w2) + " , " + \
        #             str(p[2]-w2) + ":" + str(p[2]+w2) + \
        #             " ]"
        #         s +=  " centered at " + str(p)
        #         s += " cscore=" + str(point_cscore)
        #         return s

        #     for i in range( len(pointsOfInterest) ):
        #         print ("Point {}".format(i) , " , ", point_ReportResultAs_Vol_Center_CScore( pointsOfInterest[i] , k_width) )

        #Consider removal
        # if showPlot:
        #     plt.plot(self.histogram_bins, self.histogram)
        #     plt.xlabel('Score')
        #     plt.ylabel('Counts')

        #     for i in range( len(pointsOfInterest) ):
        #         x0=pointsOfInterest[i][2] #gets the point score
        #         plt.axvline( x0, color='red')
        #         # plt.axvline(point2[2], label='point2', color='red')
        #         # plt.axvline(point3[2], label='point3', color='red')
        #         # plt.axvline(point4[2], label='point4', color='red')
        #         # plt.axvline(point5[2], label='point5', color='red')

        #         #Annotate lines
        #         plt.annotate( 'point{}'.format(i), ( x0, fgaussian(x0, popt[0], popt[1], popt[2]) ) )
        #         #uses the gaussian approximation to get a rough location of the point
        #         #TODO: this is not ingenious enough and prone to errors. Try to use raw score data for this

        #         if extraX is not None:
        #             plt.axvline( extraX, color='green')

        return pointsOfInterest

    def SelectRegionsOfInterest_V1(self , showPlot=True , printReport = True, k_width=256 ):
        if self.histogram is None:
            _a, _b = self.getHistogram()
        
        hist_vmax = np.amax(self.histogram)
        hist_x_vmax = self.histogram_bins [ np.where(self.histogram == hist_vmax)[0] ]
        
        hist_xmin = self.data3d_vmin
        hist_xmax = self.data3d_vmax

        #Choose regions based in the histogram shape
        #Approximate peak with gaussian
        def fgaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp( -0.5 * ((x - mean)**2 / stddev**2 ) )

        aguess= np.max(self.histogram)
        mguess = hist_x_vmax[0]
        popt, pcov = optimize.curve_fit(fgaussian, self.histogram_bins , self.histogram, [aguess, mguess, 1.0])

        #If negative, make positive
        stdev = abs(popt[2])

        print("Gaussian fit to peak, parameters amplitude={} , mean={} ,stdev={}".format(popt[0],popt[1],stdev ))

        hx = []

        hx.append ( (hist_xmax - hist_xmin)/4 + hist_xmin )
        hx.append ( (hist_xmax + hist_xmin)/2.0 )
        hx.append ( popt[1] - 3* stdev )  #At mean - 3*stdev (3 sigma) #MB
        hx.append ( popt[1] - (3* stdev + stdev*2.35482 /2)/2 ) #Olly
        hx.append ( popt[1] - stdev*2.35482 /2 ) # FHWM = 2.35482*stdev, At FWHM going up #Neville
        hx.append ( popt[1] - stdev*2.35482 /4 ) # MD
        hx.append ( popt[1] ) #Commonest consistency

        def getClosestlVoxelFromListOfIndices(listindicesandscores, Zc,Yc,Xc):
            #From the list of voxel indices, get the closest voxel to point (Xc, Yc, Zc)
            #Also returns the distance and its score

            listindices = listindicesandscores[0]
            listcscores = listindicesandscores[1]
            
            #First element sets the return result (default)
            if len(listindices) >0:
                voxelresult = listindices[0]
                voxelresult_dist = (voxelresult[0]-Zc)**2 + (voxelresult[1]-Yc)**2 + (voxelresult[2] - Xc)**2
                voxel_cscore = listcscores[0]
                
                if len(listindices)>1:
                    for j0 in range(1, len(listindices)):
                        i0 =  listindices[j0]
                        #print( "i0= " , i0)
                        thisdist = math.sqrt( (i0[0]-Zc)**2 + (i0[1]-Yc)**2 + (i0[2] - Xc)**2 )
                        if thisdist<voxelresult_dist:
                            voxelresult = i0
                            voxelresult_dist = thisdist
                            voxel_cscore = listcscores[j0]
                
                #print ("Closest voxel is ", voxelresult , " with distance ", voxelresult_dist)
                return voxelresult, voxelresult_dist, voxel_cscore
        
        Zcenter = int( (np.amax( self.zVolCentres ) - np.amin( self.zVolCentres )) /2 )
        Ycenter = int( (np.amax( self.yVolCentres ) - np.amin( self.yVolCentres )) /2 )
        Xcenter = int( (np.amax( self.xVolCentres ) - np.amin( self.xVolCentres )) /2 )

        pointsOfInterest=[]

        for hx0 in hx:
            pointsOfInterest.append( \
                getClosestlVoxelFromListOfIndices ( \
                self.GetIndicesFromOrigDataWithScoreNear( hx0, stdev / 4 ), \
                Zcenter , Ycenter, Xcenter) )

        return pointsOfInterest


    def SelectRegionsOfInterest_V2(self ):
        if self.histogram is None:
            _a, _b = self.getHistogram()
        
        hist_vmax = np.amax(self.histogram)
        hist_x_vmax = self.histogram_bins [ np.where(self.histogram == hist_vmax)[0] ]
        
        hist_xmin = self.data3d_vmin
        hist_xmax = self.data3d_vmax

        #Choose regions based in the histogram shape
        #Approximate peak with gaussian
        def fgaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp( -0.5 * ((x - mean)**2 / stddev**2 ) )

        aguess= np.max(self.histogram)
        mguess = hist_x_vmax[0]
        popt, pcov = optimize.curve_fit(fgaussian, self.histogram_bins , self.histogram, [aguess, mguess, 1.0])

        #If negative, make positive
        stdev = abs(popt[2])

        print("Gaussian fit to peak, parameters amplitude={} , mean={} ,stdev={}".format(popt[0],popt[1],stdev ))

        hx = []

        pos_meanminus3sigma = popt[1] - 3* stdev

        hx.append ( (pos_meanminus3sigma-hist_xmin)/3 + hist_xmin )
        #print("hx[0]= {}".format(hx[0]))

        hx.append ( (pos_meanminus3sigma-hist_xmin)/3*2 + hist_xmin )
        hx.append ( pos_meanminus3sigma )  #At mean - 3*stdev (3 sigma) #MB
        hx.append ( popt[1] - (3* stdev + stdev*2.35482 /2)/2 ) #Olly
        hx.append ( popt[1] - stdev*2.35482 /2 ) # FHWM = 2.35482*stdev, At FWHM going up #Neville
        hx.append ( popt[1] - stdev*2.35482 /4 ) # MD
        hx.append ( popt[1] ) #Commonest consistency

        def getClosestlVoxelFromListOfIndices(listindicesandscores, Zc,Yc,Xc):
            #From the list of voxel indices, get the closest voxel to point (Xc, Yc, Zc)
            #Also returns the distance and its score

            listindices = listindicesandscores[0]
            listcscores = listindicesandscores[1]
            
            
            if len(listindices) >0:
                #First element sets the return result (default)
                voxelresult = listindices[0]
                voxelresult_dist = (voxelresult[0]-Zc)**2 + (voxelresult[1]-Yc)**2 + (voxelresult[2] - Xc)**2
                voxel_cscore = listcscores[0]
                
                if len(listindices)>1:
                    for j0 in range(1, len(listindices)):
                        i0 =  listindices[j0]
                        #print( "i0= " , i0)
                        thisdist = math.sqrt( (i0[0]-Zc)**2 + (i0[1]-Yc)**2 + (i0[2] - Xc)**2 )
                        if thisdist<voxelresult_dist:
                            voxelresult = i0
                            voxelresult_dist = thisdist
                            voxel_cscore = listcscores[j0]
                
                #print ("Closest voxel is ", voxelresult , " with distance ", voxelresult_dist)
                return voxelresult, voxelresult_dist, voxel_cscore
        
        Zcenter = int( (np.amax( self.zVolCentres ) - np.amin( self.zVolCentres )) /2 )
        Ycenter = int( (np.amax( self.yVolCentres ) - np.amin( self.yVolCentres )) /2 )
        Xcenter = int( (np.amax( self.xVolCentres ) - np.amin( self.xVolCentres )) /2 )

        pointsOfInterest=[]

        for hx0 in hx:
            scorewidth= stdev / 4
            #If no indices are found then widens search. Tested, ok
            while True:
                indices0 = self.GetIndicesFromOrigDataWithScoreNear( hx0, scorewidth )
                #print("len(indices0[0]) = {}".format(len(indices0[0])))
                #print(indices0)
                if len(indices0[0])==0:
                    scorewidth += stdev / 4
                else:
                    break



            poi0 = getClosestlVoxelFromListOfIndices ( indices0, \
                Zcenter , Ycenter, Xcenter)

            #print(poi0)
            pointsOfInterest.append( poi0 )
                
        return pointsOfInterest


#This class is not being used
# class GroundTruthData:

#     def __init__(self , filename):
#         self.filename = filename
#         self.data_da = None #Default

#         #Check file has h5 extension
#         fileroot , fileext = os.path.splitext(filename)
        
#         if fileext == ".h5":
#             #Try to open the file as hdf5 to a dask array object
#             #Opens file
#             fx= h5py.File(filename,'r')
#             self.data_da = da.from_array(fx['data'], chunks='auto')
#             logging.info("File " + filename + " opened successfully as hdf5 file.")

#         if self.data_da is not None :
#             #Gets vmax and vmin
#             self.vmax = da.max( self.data_da ).compute()
#             self.vmin = da.min( self.data_da ).compute()
#             logging.info("vmax=" + str(self.vmax) + ", vmin=" + str(self.vmin))



def AvgPool3D_LargeData(data3d, w_avg = 512, k_width=256 , s_stride=8 ):
    #This function will do the avarage pooling in 3D using PyTorch AvgPool3D
    #It splits data into chunks automatically
    #and then combines the data automaticaly
    #It returns a ScoreData object


    def AvgPool3DPytorch(data3d_np , kwidth=8 , stride0=1):
        '''
        Applies Pytorch AvgPool3D on the numpy data object data3d_np with the width and stride parameters given
        Automatically uses GPU if available.
        This function will not check if the data is too large. Use this with caution
        It is recommended that data3d_np has a maximum of 512x512x512 size, to keep GPU usage acceptably low
        The function returns the average-pooled data as a numpy 3D array objects
        '''
        #Generic. It will use the GPU if available

        if torch.cuda.is_available():
                dev="cuda:0"
        else:
                dev="cpu"
        device = torch.device(dev)
        
        #convert to torch objects, and to gpu using cuda()
        data3d_torch = torch.unsqueeze( torch.unsqueeze( torch.from_numpy(data3d_np),0),0 ).to(device)
        
        #setup torch calculation
        torchc3d = torch.nn.AvgPool3d(kwidth, stride0)
        
        #Run the calculation
        result = torchc3d(data3d_torch)
        
        return result.cpu().detach().numpy()[0][0]

    def Get3DAvgPoolOfChunkWithCornerAt( data3d, iz,iy,ix , w_avg , k_width, s_stride ):
        '''
        Averages the 3D data3d (dask array), from corner (iz, iy, ix)
        and with windows of w_avg in all directions.
        So, it does AvgPool3D at region [ iz : iz+w_avg , iy : iy+w_avg , iy : iy+w_avg ]
        and with a kernel size of k_width x k_width x k_width , and stride (jump) of s_stride.
        If the desired window exceeds the limits of the data3d , it will adjust the index limits in order to fit
        Because it may change the index limits, the actual limits are also returned.
        datavol_avg
        '''

        #Check all is ok
        assert ( iz >=0 and iy>=0 and ix>=0 ) , "Error, indexes cannot be < 0 ."
        
        assert ( iz < data3d.shape[0] and
                iy < data3d.shape[1] and
                ix < data3d.shape[2]) , "Error, invalid indexes."
        
        #Adjust limits
        iz_da_min = iz
        iz_da_max = iz_da_min + w_avg
        if iz_da_max > data3d.shape[0] :
            iz_da_max = data3d.shape[0]
            iz_da_min = iz_da_max- w_avg
        
        iy_da_min = iy
        iy_da_max = iy_da_min + w_avg
        if iy_da_max > data3d.shape[1] :
            iy_da_max = data3d.shape[1]
            iy_da_min = iy_da_max- w_avg
        
        ix_da_min = ix
        ix_da_max = ix_da_min + w_avg
        if ix_da_max > data3d.shape[2] :
            ix_da_max = data3d.shape[2]
            ix_da_min = ix_da_max- w_avg
        
        logging.info( "iz_da_min=", iz_da_min,", iz_da_max=", iz_da_max,
            ", iy_da_min=", iy_da_min,", iy_da_max=", iy_da_max,
            ", ix_da_min=", ix_da_min,", ix_da_max=", ix_da_max
            )
        
        #Get volume and convert to numpy array
        datavol_da = data3d [ iz_da_min:iz_da_max , iy_da_min:iy_da_max , ix_da_min:ix_da_max ]

        #print("datavol_da.shape = ", datavol_da.shape)
        #convert to numpy
        datavol_np = datavol_da.compute()
        #print("datavol_np.shape = ", datavol_np.shape)

        #Calculate here the AvgPooling (big calculation)
        #datavol_avg = AvgPool3DPytorchGPU(datavol_np , k_width , s_stride )
        datavol_avg = AvgPool3DPytorch(datavol_np , k_width , s_stride )

        logging.info("AvgPool3D calculation complete")
        #logging.info("datavol_avg.shape = ",datavol_avg.shape)
        
        torch.cuda.empty_cache()
        
        return datavol_avg, (iz_da_min , iz_da_max , iy_da_min , iy_da_max , ix_da_min , ix_da_max)


    res=None

    assert (w_avg > k_width), "w_avg (window width average) should be higher than kwidth"
    
    # if (do_weighting):
    #     setWeightedData('MaxMinSquare')
    
    # data3d = self.weightedData_da

    if data3d is not None :
        result_avg_of_vols = np.zeros( ( int( (data3d.shape[0]-k_width)/s_stride )+1 , 
                            int( (data3d.shape[1]-k_width)/s_stride )+1  ,
                            int( (data3d.shape[2]-k_width)/s_stride )+1  ))

        logging.info ("result_avg_of_vols.shape = " , result_avg_of_vols.shape)
        
        # BIG CALCULATION
        
        #Nested iterations of w_avg x w_avg x w_avg volumes
        #step0 = int( (w_avg - k_width) / s_stride )
        step0 = int(w_avg - k_width)
        
        niter = 0 #Count the number of ierations
        time0 = time.perf_counter()
        time1 = time0
        
        ntotaliter = int(data3d.shape[0]/step0) * int(data3d.shape[1]/step0)* int(data3d.shape[2]/step0)
        
        logging.info ("ntotaliter  = ", ntotaliter)
        
        time.sleep(2) #A little pause to see print output
        
        for iz_da in range(0 , data3d.shape[0] , step0):
            for iy_da in range(0 , data3d.shape[1] , step0):
                for ix_da in range(0 , data3d.shape[2] , step0):
                
                    #Show progress
                    #clear_output(wait=True)
                    
                    if (niter>0):
                        logging.info("niteration = ", niter , "/", ntotaliter)
                        logging.info ("Estimated time to finish (s) = ",
                            str( round( (ntotaliter-niter)*(time1-time0)/niter) ) )
                    
                    logging.info("iz_da=", iz_da , "/" , data3d.shape[0] ,
                        " , iy_da=", iy_da , "/" , data3d.shape[1] ,
                        " , ix_da=", ix_da , "/" , data3d.shape[2]
                        )

                    datavol_avg , index_limits = Get3DAvgPoolOfChunkWithCornerAt(data3d, iz_da,iy_da,ix_da , w_avg , k_width, s_stride )
                    
                    #clear_output(wait=True)
                    
                    time1 = time.perf_counter()
                    niter += 1
                    
                    #With data collected, store it in appropriate array 
                    #print("index_limits = ", index_limits)
                    iz = int(index_limits[0] / s_stride)
                    iy = int(index_limits[2] / s_stride)
                    ix = int(index_limits[4] / s_stride)

                    #print ("Start indexes to store at result_avg_of_vols: " , iz , iy , ix)

                    result_avg_of_vols[ iz : (iz + datavol_avg.shape[0]) ,
                                    iy : (iy + datavol_avg.shape[1]) ,
                                    ix : (ix + datavol_avg.shape[2]) ] = datavol_avg

        logging.info("Completed.")      
        
        
        #Create the respective indexes
        #Indexes are the midpoints of the respective averaging volume
        # (No index should have a value of 0)
        result_avg_of_vols_x_range = np.arange( int(k_width/2) , data3d.shape[2]-int(k_width/2)+1, s_stride )
        result_avg_of_vols_y_range = np.arange( int(k_width/2) , data3d.shape[1]-int(k_width/2)+1 , s_stride )
        result_avg_of_vols_z_range = np.arange( int(k_width/2) , data3d.shape[0]-int(k_width/2)+1 , s_stride )

        #Attention, order of x,y,z has to be in this way
        #otherwise the vales will not correspond to the averaging point volumes.
        #In a 3D array, first index is zz, 2nd is yy, and 3rd is xx
        result_avg_of_vols_z , result_avg_of_vols_y , result_avg_of_vols_x = np.meshgrid( result_avg_of_vols_z_range ,
                                                                                        result_avg_of_vols_y_range,
                                                                                        result_avg_of_vols_x_range,
                                                                                        indexing='ij')
        #Resulting meshgrids should have the same shape as result_avg_of_vols
        logging.info ("result_avg_of_vols_x.shape = ", result_avg_of_vols_x.shape )
        
        #Create a ScoreData object containing all the data
        res = ScoreData(result_avg_of_vols , result_avg_of_vols_z , result_avg_of_vols_y , result_avg_of_vols_x )

    else:
        logging.error("No data3d.")

    return res


# def SorensenDiceCoefficientCalculator3DPool (data1_da, data1_thresh, data2_da , data2_thresh , w_avg , k_width , s_stride ):

#     #This will not check whether the data is boolean or not.
#     logging.info("SorensenDiceCoefficientCalculator3DPool")

#     #check shapes of data1 and data2 are the same
#     if (data1_da.shape == data2_da.shape ):
#         #Both data has the same shape

#         #threshold data and convert to values 0.0 and 1.0 only
#         data1_th_bool = da.where(data1_da < data1_thresh , False ,True)
#         data2_th_bool = da.where(data2_da < data2_thresh , False ,True)

#         logging.info ("Calculating elementwise d1==d2")
#         neq_d1d2_bool= da.equal(data1_th_bool , data2_th_bool)

#         #Convert
#         neq_d1d2 = neq_d1d2_bool.astype('float')
#         #Do calculation similar to average pooling, but get the sum back by multiplying by the volume (number of elements)
#         logging.info ("AvgPool3D_LargeData of neq_d1d2")
#         SDnumerator_SC = AvgPool3D_LargeData( neq_d1d2 , w_avg , k_width , s_stride )

        
#         #data1_np_th= data1_np_th_bool.astype('uint8')
#         data1_th = data1_th_bool.astype('float')
#         logging.info ("AvgPool3D_LargeData of data1_th")
#         SDden_d1_SC = AvgPool3D_LargeData( data1_th , w_avg , k_width , s_stride )

#         data2_th = data2_th_bool.astype('float')
#         logging.info ("AvgPool3D_LargeData of data1_th_int")
#         SDden_d2_SC = AvgPool3D_LargeData( data2_th , w_avg , k_width , s_stride )

#         #Calculates the SD coefficient on the pooled data, pixelwise
#         SDCoeffRes = ScoreData( 2*SDnumerator_SC.data3d /( SDden_d1_SC.data3d + SDden_d2_SC.data3d ) , \
#             SDnumerator_SC.zVolCentres, SDnumerator_SC.yVolCentres , SDnumerator_SC.xVolCentres )
        
#         return SDCoeffRes



def SorensenDiceCoefficientCalculator3DPool (data1_da, data1_thresh, data2_da , data2_thresh , w_avg , k_width , s_stride ):
    '''
    Calculates Sorensen-Dice coefficient by using the formula:

    SDC = 2* (data1==data2).sum / (data1.volume + data2.volume)

    where (data1==data2) is volume where voxels have values for 1 when d1(z,y,x) = d2(z,,y,x) and zero otherwise

    Since data1.volume = data2.volume, and using that sum/volume = average, then is simply given by
    SDC = (data1==data2).average
    '''
    #This will not check whether the data is boolean or not.
    logging.info("SorensenDiceCoefficientCalculator3DPool")

    #check shapes of data1 and data2 are the same
    if (data1_da.shape == data2_da.shape ):
        #Both data has the same shape

        #threshold data and convert to values 0.0 and 1.0 only
        data1_th_bool = da.where(data1_da < data1_thresh , False ,True)
        data2_th_bool = da.where(data2_da < data2_thresh , False ,True)

        logging.info ("Calculating elementwise d1==d2")
        neq_d1d2_bool= da.equal(data1_th_bool , data2_th_bool)

        #Convert
        neq_d1d2 = neq_d1d2_bool.astype('float')
        #Do calculation similar to average pooling, but get the sum back by multiplying by the volume (number of elements)
        logging.info ("AvgPool3D_LargeData of neq_d1d2")
        SDnumerator_SC = AvgPool3D_LargeData( neq_d1d2 , w_avg , k_width , s_stride )

        SDCoeffRes = SDnumerator_SC
        
        return SDCoeffRes

def SorensenDiceCoefficientCalcWholeVolume (data1_da, data1_thresh, data2_da , data2_thresh ):
    '''
    Calculates Sorensen-Dice coefficient by using the formula:

    SDC = 2* (data1==data2).sum / (data1.volume + data2.volume)

    where (data1==data2) is volume where voxels have values for 1 when d1(z,y,x) = d2(z,,y,x) and zero otherwise

    Since data1.volume = data2.volume, and using that sum/volume = average, then is simply given by
    SDC = (data1==data2).average
    '''
    #This will not check whether the data is boolean or not.
    logging.info("SorensenDiceCoefficientCalcWholeVolume")

    #check shapes of data1 and data2 are the same
    if (data1_da.shape == data2_da.shape ):
        #Both data has the same shape

        #threshold data and convert to values 0.0 and 1.0 only
        data1_th_bool = da.where(data1_da < data1_thresh , False ,True)
        data2_th_bool = da.where(data2_da < data2_thresh , False ,True)

        logging.info ("Calculating elementwise d1==d2")
        neq_d1d2_bool= da.equal(data1_th_bool , data2_th_bool)

        #Convert
        neq_d1d2 = neq_d1d2_bool.astype('float')

        #Calculates the Sorensen-Dice coefficient as the mean of the whole array
        sdc_wholevol = da.mean(neq_d1d2).compute()
        
        return sdc_wholevol
    else:
        return None

