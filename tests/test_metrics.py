import numpy as np
import dask as da
import pytest
import leopardgecko.metrics as lgm

@pytest.fixture
def get_two_annot_volumes():
    datashape = (256,256,256)


    data0 = np.zeros(datashape, dtype=np.uint8)
    data1 = np.zeros(datashape, dtype=np.uint8)

    #Create annotations with some overlap along z

    data0[0:64,:,:] = 1
    data1[32:96,:,:] = 1
    """
    inters = 32*256*256
    union = 64*256*256 + 64*256*256 = 128*256*256
    dice = 2*(inters)/union = 64/128 = 0.5
    """
    
    data0[128:132,:,:]=2
    data1[130:136,:,:] = 2
    """
    inters = 2*256*256
    union = 4*256*256 + 6*256*256 = 10*256*256
    dice = 2*(inters)/union = 4/10 = 0.4
    """

    return data0, data1

def test_dice(get_two_annot_volumes):
    #data0,data1 = get_two_annot_volumes()
    dice= lgm.MetricScoreOfVols_Dice(*get_two_annot_volumes)
    print(dice) #use -s in pytest to see print statements

    diceavg, dicescores = dice

    assert diceavg== 0.45

    assert dicescores[1]==0.5 
    assert dicescores[2]==0.4
    

def test_dice_dask(get_two_annot_volumes):
    #data0,data1 = get_two_annot_volumes()
    dice= lgm.MetricScoreOfVols_Dice(*get_two_annot_volumes, use_dask=True)
    print(dice) #use -s in pytest to see print statements

    diceavg, dicescores = dice

    assert diceavg== 0.45

    assert dicescores[1]==0.5 
    assert dicescores[2]==0.4

def test_accuracy(get_two_annot_volumes):
    #data0,data1 = get_two_annot_volumes()
    acc= lgm.MetricScoreOfVols_Accuracy(*get_two_annot_volumes)
    print(acc) #use -s in pytest to see print statements

    assert round(acc,3)==0.727