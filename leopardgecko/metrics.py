import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import dask.array as da
import logging

def MetricScoreOfVols_Accuracy(vol0, vol1):
    '''
    Get the whole-volume Accuracy metric between two volumes that have been segmented
    '''
    equalvol = np.equal(vol0,vol1).astype(np.float32)
    
    res = equalvol.mean()
    return res

def MetricScoreOfVols_Dice(vol0, vol1, useBckgnd=False):
    '''
    Get the whole-volume Multiclass metric between two volumes that have been segmented
    '''
    
    logging.info("MetricScoreOfVols_Dice()")
    #Check both volumes dtype is int

    if not ( np.issubdtype(vol0.dtype, np.integer) and np.issubdtype(vol1.dtype, np.integer) ):
        logging.info("Volumes are not integer type. Try to convert them")
        vol0 = vol0.astype(np.int8)
        vol1= vol1.astype(np.int8)

        #equalvol = np.equal(vol0,vol1).astype(np.float32)
        #res = equalvol.mean()

    #Number of segmentations
    nseg0 = np.max(vol0)
    nseg1 = np.max(vol1)
    logging.info(f"Number of class segmentations in first volume {nseg0+1}")
    logging.info(f"Number of class segmentations in second volume {nseg1+1}")

    if nseg0 != nseg1:
        logging.warning ("Number of segmentations between volumes is different.")
    
    nseg = max(nseg0, nseg1)

    isegstart=1
    if useBckgnd: isegstart=0
    
    #Calculate dice of each metric
    #Similar to code in https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L343
    dicescores=np.array([])
    for iseg in range(isegstart,nseg+1): #include last value , discards background (iseg=0)
        p = np.where(vol0 == iseg, 1, 0)
        t = np.where(vol1 == iseg, 1, 0)
        c_inter = (p*t).astype(np.float32).sum()
        c_union = (p+t).astype(np.float32).sum()

        #dicescore= 2.*self.inter[c]/self.union[c] if self.union[c] > 0 else np.nan
        dicescore= 2.*c_inter/c_union if c_union > 0 else np.nan

        dicescores = np.append(dicescores, dicescore)

    dicescore_all = np.nanmean(dicescores)

    return dicescore_all