import logging
import numpy as np

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


def hvect_count_in_data(data_all, hvectors):
    '''
    Does some counting statistics of hvector appearences in the data_all volumes.
    Parameters:
        data_all: Several data volumes, with the first dimension being the index of each of the data volumes
        hvectos: a list of tuples with all the possible hvectors
    Returns:
        hvect_counter: dictionary with each entry being hvector: count
    '''
    logging.info("hvect_countstats")
    #Initialise the counter
    hvect_counter2 = {}
    hvect_counter2_max=0

    for h0 in hvectors:
        hvect_counter2[h0] = 0

    zsize = data_all.shape[1]

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


def testme():
    print("testme()")