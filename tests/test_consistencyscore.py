import numpy as np
import dask as da
import pytest
import leopardgecko.ConsistencyScore as lgcs

shape0 = (2,32,32)

@pytest.fixture
def get_pdata_ways():
    shape1 = (*shape0,3) # z,y,x, class

    pdata_ways=np.zeros((12,*shape1), dtype=np.float32) # ways, z,y,x, class

    #Default. certainty for class 0
    pdata_ways[:,:,:,:,0]=1
    pdata_ways[:,:,:,:,1]=0
    pdata_ways[:,:,:,:,2]=0

    #Top bar, certainty for background in all ways
    # no change needed

    #Second bar, certainty for class 1
    pdata_ways[:,:,4:8,:,0]=0
    pdata_ways[:,:,4:8,:,1]=1
    pdata_ways[:,:,4:8,:,2]=0

    #Third bar, certainty for class 2
    pdata_ways[:,:,8:12,:,0]=0
    pdata_ways[:,:,8:12,:,1]=0
    pdata_ways[:,:,8:12,:,2]=1

    #Fourth bar, 1/3 for all
    pdata_ways[:,:,12:16,:,0]=1.0/3.0
    pdata_ways[:,:,12:16,:,1]=1.0/3.0
    pdata_ways[:,:,12:16,:,2]=1.0/3.0

    # Fifth bar, varying across ways, certainty depending on mod(ways)

    for i in range(12):
        pdata_ways[i,:,16:20,:,0]= ((i%3)==0)*1
        pdata_ways[i,:,16:20,:,1]= (((i+2)%3) ==0)*1
        pdata_ways[i,:,16:20,:,2]= (((i+1)%3) ==0)*1
    
    return pdata_ways


def test_consistencyscore_labels():
    #Generate some data for testing
    #shape0 = (2,32,32)

    data_ways=np.zeros((12,*shape0), dtype=np.uint8) # ways, z,y,x

    for i in range(12):
        data0 = np.zeros(shape0,dtype=np.uint8)

        data0[:,:, 4:i*2+8 ]=1

        data0[:,0:4,:]=0
        data0[:,4:8,:]=1
        data0[:,8:12,:]=2

        #bar with most inconsistency
        data0[:,12:16,:]= i%3

        data_ways[i,...]=data0

    #run consistency score calculation
    cslabels = lgcs.getCScoreFromMultipleWayLabelsPred(data_ways)

    assert np.allclose(cslabels[:,0:12,:], 1.0)
    assert np.allclose(cslabels[:,12:16,:], 0.0)
    
        
    exp_res_profile = np.array([1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 0.7708333 , 0.7708333 ,
       0.5833333 , 0.5833333 , 0.4375    , 0.4375    , 0.33333334,
       0.33333334, 0.27083334, 0.27083334, 0.25      , 0.25      ,
       0.27083334, 0.27083334, 0.33333334, 0.33333334, 0.4375    ,
       0.4375    , 0.5833333 , 0.5833333 , 0.7708333 , 0.7708333 ,
       1.        , 1.        ]
    )

    assert np.allclose( exp_res_profile, cslabels[0,20,:])

def test_consistencyscore_probs(get_pdata_ways):
    pdata_ways=get_pdata_ways

    csprobs = lgcs.getCScoreFromMultipleWayProbsPred(pdata_ways)

    assert np.allclose(csprobs[:,0:12,:], 1.0)
    assert np.allclose(csprobs[:,12:20,:], 0.0, atol=1e-7)
    assert np.allclose(csprobs[:,20:,:], 1.0)

def test_consistencyscore_probs_accum(get_pdata_ways):
    pdata_ways=get_pdata_ways

    cs_cl = lgcs.cConsistencyScoreMultipleWayProbsAccumulate()
    for iways in range(12):
        cs_cl.accumulate(pdata_ways[iways,...])

    csprobs_acc = cs_cl.getCScore()

    assert np.allclose(csprobs_acc[:,0:12,:], 1.0)
    assert np.allclose(csprobs_acc[:,12:20,:], 0.0, atol=1e-7)
    assert np.allclose(csprobs_acc[:,20:,:], 1.0)
    