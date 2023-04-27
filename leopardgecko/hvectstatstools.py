import logging
import numpy as np
import dask.array as da
import tqdm

def hvect_gndclass_counter(data_all , data_class_gnd , hvectors):
    '''
    Does some counting statistics of hvector,gndclass.
    This is useful to analyse the quality of predictions from Unet, and may help with assignment of hvector-to-classlabel

    Parameters:
        data_all: Several data volumes, with the first dimension being the index of each of the data volumes
        data_class_gnd: data volume with the ground truth class assignement for respective voxels
        hvectors: a list of tuples with all the possible hvectors
    Returns:
        hvect_gndclass_counter: dictionary with each entry being hvector: [c0,c1,c2]
            with c0 c1 and c2 being the counts for each class
    '''
    nclasses = len(hvectors[0])

    #Initialise the counter
    #Another way to handle data
    hvect_gndclass_counter1 = {}
    for h0 in hvectors:
        d0= []
        for c0 in range(nclasses):
            d0.append(0)
        hvect_gndclass_counter1[ h0 ] = d0

    vols_shape = data_all.shape
    #Count voxel-by-voxel
    for iz in range(vols_shape[1]):
        for iy in range(vols_shape[2]):
            for ix in range(vols_shape[3]):
                v = data_all[:, iz,iy,ix]
                h1 = tuple(v) #Convert to tuple to get the h-vector

                gnd0 = data_class_gnd[iz,iy,ix]

                #increment dict counter
                hvect_gndclass_counter1[ h1 ][gnd0] += 1
    
    return hvect_gndclass_counter1

# def gethvect_gndclass_counter_in_data_counted(data, data_gnd, hvectors):
#     '''
#     Does some counting statistics of hvector,gndclass.
#     This is useful to analyse the quality of predictions from Unet, and may help with assignment of hvector-to-classlabel

#     Parameters:
#         data: volume where each voxeld has the counted number of volumes that gave non-background
#         data_class_gnd: data volume with the ground truth class assignement for respective voxels
#         hvectors: a list of tuples with all the possible hvectors
#     Returns:
#         hvect_gndclass_counter: dictionary with each entry being hvector: [c0,c1,c2]
#             with c0 c1 and c2 being the counts for each class
#     '''
#     nclasses = len(hvectors[0])
#     nways = np.sum(np.array(hvectors[0],dtype=np.uint8))
#     #Initialise the counter
#     hvect_gndclass_counter0 = {}
#     for h0 in hvectors:
#         d0= []
#         for c0 in range(nclasses):
#             d0.append(0)
#         hvect_gndclass_counter0[ h0 ] = d0

#     vols_shape = data.shape
#     #Count voxel-by-voxel

#     #This code is really slow
#     #TODO: try to make it run faster
#     for iz in range(vols_shape[0]):
#         print(f"iz={iz} / {vols_shape[0]}")
#         for iy in range(vols_shape[1]):
#             for ix in range(vols_shape[2]):
#                 v = data[iz,iy,ix]
                
#                 #Create the hvector from the count
#                 h1 = (nways-v,v)

#                 gnd0 = data_gnd[iz,iy,ix]

#                 #increment dict counter
#                 hvect_gndclass_counter0[ h1 ][gnd0] += 1

#     return hvect_gndclass_counter0


# TODO: REQUIRES REVISION. IT APPEARS IT WORKS ONLY WITH 2-CLASS DATA
# def gethvect_gndclass_counter_in_data_counted(countdata_np, data_gnd, hvectors):
#     '''
#     Does some counting statistics of hvector,gndclass.
#     This is useful to analyse the quality of predictions from Unet, and may help with assignment of hvector-to-classlabel

#     Parameters:
#         data: volume where each voxeld has the counted number of volumes that gave non-background
#         data_class_gnd: data volume with the ground truth class assignement for respective voxels
#         hvectors: a list of tuples with all the possible hvectors
#     Returns:
#         hvect_gndclass_counter: dictionary with each entry being hvector: [c0,c1,c2]
#             with c0 c1 and c2 being the counts for each class
    
#     This version is much faster than the previous version
#     See: scripts/developing_hvectstatstools.ipynb
#     '''

#     nclasses = len(hvectors[0])
#     nways = int(np.sum(np.array(hvectors[0],dtype=np.uint8)))

#     #Initialise the counter
#     hvect_gndclass_counter0 = {}
#     for h0 in hvectors:
#         d0= []
#         for c0 in range(nclasses):
#             d0.append(0)
#         hvect_gndclass_counter0[ h0 ] = d0

#     data_gnd_binary = (data_gnd!=0)

#     #iterate calculation per vector basis
#     for i0 in range(nways+1):
#         hv0 = (i0, nways-i0)#I am not sure what is happening here, this seems to be only applicable to 2-class data

#         #get matches
#         d_match = (data==nways-i0)
#         match_count = np.count_nonzero(d_match)

#         #Compare with gntruth
#         count_compare_to_gnd = np.count_nonzero(np.logical_and(d_match,data_gnd_binary))

#         hvect_gndclass_counter0[hv0] = [ match_count- count_compare_to_gnd, count_compare_to_gnd]


#     return hvect_gndclass_counter0

def hvect_count_in_data(data_all, hvectors):
    '''
    Does some counting statistics of hvector appearences in the data_all volumes.
    Parameters:
        data_all: Several data volumes, with the first dimension being the index of each of the data volumes
        hvectos: a list of tuples with all the possible hvectors
    Returns:
        hvect_counter: dictionary with each entry being hvector: count
        hvect_counter_max: maximum count value. It can be useful for setting up plots
    '''
    logging.info("hvect_countstats")
    #Initialise the counter
    hvect_counter2 = {}
    hvect_counter2_max=0

    for h0 in hvectors:
        hvect_counter2[h0] = 0

    zsize = data_all.shape[1]

    #TODO: WARNING THIS ONLY SUPPORTS 3-CLASS
    for iz in range(zsize):
        logging.info(f"iz = {iz} / {zsize}")
        zslice = data_all[:,iz,:,:]
        for ih,h0 in enumerate(hvectors):
            #print(f"ih:{ih} , h0:{h0}")
            match = np.where(zslice[0,:,:]==h0[0], 1, 0 ) * np.where( zslice[1,:,:]==h0[1], 1, 0 ) * np.where( zslice[2,:,:] == h0[2],1,0 )
            hvect_counter2[h0] += np.sum(match)
            #update max here
            hvect_counter2_max = max(hvect_counter2_max, hvect_counter2[h0])

    return hvect_counter2, hvect_counter2_max


# def hvect_count_in_data_counted(countdata_np, hvectors):
#     '''
#     Does some counting statistics of values inside the data_np
#     Parameters:
#         countdata_np: volume with the counted occurences of non-background prediction per voxel
#         hvectors: a list of tuples with all the possible hvectors
#     Returns:
#         hvect_count_in_data0 : dictionary with each entry being {hvector:count}
#         hvect_count_in_data0_max : maximum count value. It can be useful for setting up plots
#     '''
#     #Initialise the counter
#     hvect_count_in_data0={} #dictionary hvector to counts
#     hvect_count_in_data0_max=0

#     for h0 in hvectors:
#         v0 = h0[1]
#         count = np.count_nonzero(countdata_np == v0)
#         print(f"hvector: {h0} , count:{count}")
#         hvect_count_in_data0[h0]=count
#         hvect_count_in_data0_max = max(hvect_count_in_data0_max, count )

#     return hvect_count_in_data0, hvect_count_in_data0_max


def get_hvect_combinations(nclasses, nways):
    '''
    Gets all the possible hvectors
    from nways and nclasses.
    Returns a list with all the possible values in tuples
    '''
    def _getCombinations(inextdim,pbase):
        #plist_ret = np.zeros(nclasses)
        plist_ret = [] #python list (not numpy)
        #print(f"inextdim = {inextdim}")
        if inextdim==nclasses-1:
            #Last index
            #plist_ret.append( nways - pbase.sum())
            pbase1= np.copy(pbase)
            pbase1[inextdim] = nways - pbase.sum()
            #plist_ret= [pbase1]
            plist_ret= [tuple(pbase1)]
            #print(f"Last index; plist_ret={plist_ret}")
        else:
            for i in range(0, int(nways-pbase.sum()+1) ):
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
    
    pbase0 = np.zeros(nclasses, dtype=int)
    pcomb0= _getCombinations(0, pbase0)
    
    return pcomb0


def testme():
    print("testme()")


# Fast metrics from hvectors

def get_fast_metrics(hvect_gnd_class_counter, hvect_to_class):
    '''
    Returns: dictionary with metrics. Currently accuracy and dice scores
    '''
    
    #Accuracy
    totalvol=0
    accmatch = 0

    #Dice
    ndims = len(list(hvect_to_class.keys())[0])
    print(f"ndims:{ndims}")
    totalinters = np.zeros(ndims, dtype=np.float32)
    totalpred = np.zeros(ndims, dtype=np.float32)
    totalgnd = np.zeros(ndims, dtype=np.float32)

    for h0, v0 in hvect_gnd_class_counter.items():
        v= np.array(v0)
        segm0 = hvect_to_class[h0]

        #Accuracy
        totalvol+= np.sum(v) #Acc
        accmatch += v[segm0] #Add only the segment predicted match is found

        #Dice
        totalgnd+=v #Dice
        totalinters[segm0] += v[segm0]
        vsum = np.sum(v)
        totalpred[segm0] += vsum
    
    accuracy_value = float(accmatch) / totalvol
    print (f"accmatch={accmatch} , totalvol={totalvol}")
    print(f"accuracy_value={accuracy_value}")

    print (f"totalinters={totalinters} , totalpred={totalpred} , totalgnd={totalgnd} ")
    dicescores = 2*np.array(totalinters, dtype=np.float32) / (totalpred+totalgnd)
    print(f"dicescore={dicescores}")
    dicescore_mean_with_bkg = np.mean(dicescores) #If want to use background
    dicescore_mean = np.mean(dicescores[1:])
    print(f"dicescore_mean_with_bkg={dicescore_mean_with_bkg} , dicescore_mean={dicescore_mean}")

    return {
        "accuracy_score": accuracy_value,
        "dice_scores": dicescores
        }


def hvect_gndclass_counter1(data_all , data_class_gnd , hvectors):
    '''
    Prototyping

    Does some counting statistics of hvector,gndclass.
    This is useful to analyse the quality of predictions from Unet, and may help with assignment of hvector-to-classlabel

    Parameters:
        data_all: Several data volumes, with the first dimension being the index of each of the data volumes
        data_class_gnd: data volume with the ground truth class assignement for respective voxels
        hvectors: a list of tuples with all the possible hvectors
    Returns:
        hvect_gndclass_counter: dictionary of hvector:tuple with the tuple being the number of occurences of
            the indexed class of that h-vector in the ground truth volume
    '''
    nclasses = len(hvectors[0])

    #Initialise the counter
    #Another way to handle data
    hvect_gndclass_counter1 = {}
    for h0 in hvectors:
        d0= []
        for c0 in range(nclasses):
            d0.append(0)
        hvect_gndclass_counter1[ h0 ] = d0

    #For each h-vector, get locations where the h-vector appears
    for h0 in hvectors:
        print("hvector:", h0)
        d_h0_locs=None
        for i, h0_el in enumerate(h0):
            if i==0: # initialise for the first class
                d_h0_locs = (data_all[i,:,:,:]==h0_el)
            else:
                d_h0_locs = ( ( data_all[i,:,:,:]==h0_el) & d_h0_locs )

        #By this point d_h0_locs should be boolean volume where hvector appears        
        #Compare with ground truth

        # match= np.where(d_h0_locs, 1,np.nan)

        # masked_gnd = match*data_class_gnd #Mask in such a way that outside the mak will be NaN

        # for cl0 in range(nclasses):
        #     count_in_gnd_for_cls = np.sum(np.where(masked_gnd==cl0,1,0))
        #     hvect_gndclass_counter1[ h0 ][cl0] += count_in_gnd_for_cls

        #Try to use masked array
        masked_gnd = np.ma.masked_where(np.logical_not(d_h0_locs), data_class_gnd)
        for cl0 in range(nclasses):
            count_in_gnd_for_cls = np.ma.sum(np.ma.where(masked_gnd==cl0,1,0))
            if count_in_gnd_for_cls is np.ma.masked:
                count_in_gnd_for_cls=0
            hvect_gndclass_counter1[ h0 ][cl0] += count_in_gnd_for_cls

    return hvect_gndclass_counter1

def get_n2c_from_hvect_count_in_data_max(hvect_gndclass_count_in_data0_l):
    '''
    From a hvect_gndclass_count_in_data data type ( a dictionary with tuples to a list of counts in ground truth), gets the
    class that gives the maximum count for each hvector.

    Example: if dictionary has entry (3,4,5):[35,64,20] , class index 1 has maximum counts, then the corresponding result in the dictionary returned will be
    (3,4,5):1

    In some cases, the counts are all zero, and there is not enough information to work out the maximum.
    (11,1,0):[30,0,0] will then return (11,1,0):-1

    '''

    hvect_to_class_dict = {}
    for hvect, d0 in hvect_gndclass_count_in_data0_l.items():
        #Get max wedge
        d0_np = np.array(d0)
        max = np.max(d0_np)
        if max>0:  #Check if there is any data that is enough for classification based on highest counts
            imax = np.argmax(d0_np) #This will be the class that will be selected
            hvect_to_class_dict[hvect] = imax
        else:
            hvect_to_class_dict[hvect] = -1

    return hvect_to_class_dict

def identifiy_class_in_dataall_using_n2c(data_all, n2c, show_progress_bar=False):
    '''
    Identifies the class for each voxel in the volumes using the N2C provided.
    Assumes that:
        data_all first index corresponds to the segmentation class nway score
        Each voxel has numbers 0 to nways

    '''

    logging.info("identifiy_class_in_dataall_using_n2c()")

    data_all_shape = data_all.shape
    
    if len(data_all_shape) != 4:
        print("data_all is not 4-dimensional. exiting with None")
        return None
    
    #Initialise values to -1 (=no class identified for each voxel)
    classid_vol = np.full( (data_all.shape[1], data_all.shape[2], data_all.shape[3]) , -1 , dtype=np.int8)

    for iz in tqdm.trange(data_all_shape[1], disable=not show_progress_bar):
        for iy in range(data_all_shape[2]):
            for ix in range(data_all_shape[3]):
                v = data_all[:, iz,iy,ix]
                h0=tuple(v)
                cl0=-1 #default, couldn't identify
                if h0 in n2c:
                    cl0= n2c[h0]
                classid_vol[iz,iy,ix] = cl0

    #This could probably be vectorized but i tested it and
    # found that non-vevctorized version is faster
    
    return classid_vol