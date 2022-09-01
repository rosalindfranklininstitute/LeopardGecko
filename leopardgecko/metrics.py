import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import dask.array as da
import logging

def MetricScoreOfVols_Accuracy(vol0, vol1, loginfosum=False):
    '''
    Get the whole-volume Accuracy metric between two volumes that have been segmented
    Uses dask array by default
    '''
    logging.info("MetricScoreOfVols_Accuracy()")
    vol0_da = da.array(vol0)
    vol1_da = da.array(vol1)
    #equalvol = da.equal(vol0_da,vol1_da).astype(np.float32)
    equalvol = da.equal(vol0_da,vol1_da).astype(np.uint8) #reduce memory footprint
    
    if loginfosum:
        sum = da.sum(equalvol, dtype=np.float64).compute() #compute it float64 accumulator
        logging.info(f"For interest sum = {sum}, volsize = {equalvol.size}")

    res = da.mean(equalvol, dtype=np.float64).compute() #Compute mean suing float64 accumulator precision

    return res


def MetricScoreOfVols_Accuracy1(vol0, vol1, showinfosum=False):
    '''
    Alternative
    Get the whole-volume Accuracy metric between two volumes that have been segmented
    '''
    logging.info("MetricScoreOfVols_Accuracy1()")
    vol0_da = da.array(vol0)
    vol1_da = da.array(vol1)
    vol_sub = vol0_da-vol1_da
    equalvol = da.where(vol_sub==0, 1.0, 0.0)

    logging.info("equalvol calculated")

    if showinfosum:
        sum0 = da.sum(equalvol)
        sum = sum0.compute()
        logging.info(f"For interest sum = {sum}, volsize = {equalvol.size}")
    
    res = equalvol.mean().compute()
    return res

def MetricScoreOfVols_Dice(vol0, vol1, useBckgnd=False):
    '''
    Get the whole-volume Multiclass metric between two volumes that have been segmented
    '''

    logging.info("MetricScoreOfVols_Dice()")

    #Check arrays have the same shape
    shape1 = vol0.shape
    shape2 = vol1.shape
    if not np.array_equal( np.array(shape1), np.array(shape2)):
        logging.info("Arrays have different shapes. Exiting with None")
        return None

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

    #Use dask arrays is arrays are too large >1Gb
    usedask=False
    if vol0.size> 1e9:
        logging.info("Arrays too large >1Gb, will use dask.")
        usedask=True
    
    if not usedask:
        for iseg in range(isegstart,nseg+1): #include last value , discards background (iseg=0)
            p = da.where(vol0 == iseg, 1, 0)
            t = da.where(vol1 == iseg, 1, 0)
            # c_inter = (p*t).astype(np.float32).sum()
            # c_union = (p+t).astype(np.float32).sum()
            c_inter = np.sum( (p*t) , dtype=np.float64 )
            c_union = np.sum( (p+t) , dtype=np.float64 )

            #dicescore= 2.*self.inter[c]/self.union[c] if self.union[c] > 0 else np.nan
            dicescore= 2.*c_inter/c_union if c_union > 0 else np.nan
            dicescores = np.append(dicescores, dicescore)

    else:
        dicescores=np.array([])
        for iseg in range(isegstart,nseg+1):
            vol0_da = da.from_array(vol0)
            vol1_da = da.from_array(vol1)

            p = da.where(vol0_da == iseg, 1, 0)
            t = da.where(vol1_da == iseg, 1, 0)

            # c_inter = (p*t).astype(np.float32).sum()
            # c_union = (p+t).astype(np.float32).sum()
            c_inter = np.sum( (p*t) , dtype=np.float64 )
            c_union = np.sum( (p+t) , dtype=np.float64 )

            dicescore_da= 2.*c_inter/c_union if c_union > 0 else np.nan
            logging.info(f"iseg={iseg}, compute()")
            dicescore = dicescore_da.compute()

            dicescores = np.append(dicescores, dicescore)

    logging.info(f"dicescores = {dicescores}")
    dicescore_all = np.nanmean(dicescores)

    return dicescore_all