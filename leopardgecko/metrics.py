
import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import dask.array as da
import logging

def MetricAccuracyWholeVolume ( data1_da, data2_da ):
    '''
    Calculates Accuracy metric of the whole volume by using the formula:

    Accuracy = (data1==data2).sum / data1(2).volume

    where (data1==data2) is volume where voxels have values for 1 when d1(z,y,x) = d2(z,,y,x) and zero otherwise

    Parameters:
    data1_da , data2_da : Volume data to compare, often with integer values per voxel representing class/segmentation.
    Data type is daskarray format

    Returns:
    A single value corresponding to the Accuracy score between the two data sets across the whole volume

    '''
    #This code will not check whether the data is boolean or not.

    logging.info("MetricAccuracyWholeVolume")

    #check shapes of data1 and data2 are the same
    if (data1_da.shape == data2_da.shape ):
        #Both data has the same shape

        logging.info ("Calculating elementwise d1==d2")
        neq_d1d2_bool= da.equal(data1_da , data2_da)

        #Convert
        neq_d1d2 = neq_d1d2_bool.astype('float')

        #Calculates the Accuracy-metric coefficient as the mean of the whole array
        ret = da.mean(neq_d1d2).compute()
        
        return ret
    else:
        return None

def MetricSorensenDiceCoefficientWholeVolume (data1bool_da, data2bool_da ):
    '''
    Calculates Sorensen-Dice coefficient by using the formula:

    SDC = 2* (data1bin_da * data2bin_da).sum / (data1bin_da.sum() + data2bin_da.sum)

    Parameters:
    data1bool_da , data2bool_da : Volume data with zeros (background) and ones (mask) per voxel. Data type is daskarray

    Returns:
    A single value corresponding to the Sorensen-Dice score between the two data sets across the whole volume

    '''

    logging.info("MetricSorensenDiceCoefficientWholeVolume")

    #This will not check whether the data is boolean or not.
    #logging.info("SorensenDiceCoefficientCalcWholeVolume")

    #Assumes data has values 0 or 1
    #It will not work with multiple segmented images

    #check shapes of data1 and data2 are the same
    if (data1bool_da.shape == data2bool_da.shape ):
        #Both data has the same shape

        im_intersection = (data1bool_da * data2bool_da).sum().compute()
        im_sum = (data1bool_da.sum() + data2bool_da.sum()).compute()

        if im_sum==0:
            return None

        #Calculates the Sorensen-Dice coefficient as the mean of the whole array
        sdc_wholevol =  2.0 * im_intersection / im_sum
        
        return sdc_wholevol
    else:
        return None
