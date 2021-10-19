import numpy as np
import h5py
import math
from scipy import optimize

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
        '''
        Saves ScoreData to a hdf5 file, containing the data in /data, and also the respective
        X,Y and Z indices
        '''
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
        '''
        Gets an histogram of the score values
        Returns tuple (histrogram,bins), but also stores it locally in the class
        '''
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

        pointsOfInterest=None
        
        try:
            popt, pcov = optimize.curve_fit(fgaussian, self.histogram_bins , self.histogram, [aguess, mguess, 1.0])
        except:
            print("Error trying to fit gaussian to histogram.")
        else:
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


        pointsOfInterest=None
        
        try:
            popt, pcov = optimize.curve_fit(fgaussian, self.histogram_bins , self.histogram, [aguess, mguess, 1.0])
        except:
            print("Error trying to fit gaussian to histogram.")
        else:

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

        pointsOfInterest=None

        try:
            popt, pcov = optimize.curve_fit(fgaussian, self.histogram_bins , self.histogram, [aguess, mguess, 1.0])
        except:
            print("Error trying to fit gaussian to histogram.")
        else:
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
