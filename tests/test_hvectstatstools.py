'''
Useful functions to get statistics of hvectors in data volumes

'''

import leopardgecko as lg

#lg.hvectstatstools.testme()

def test_get_hvect_combinations():
    #Get hvector combinations in a 2-class, 12 way system
    hvects2 = lg.hvectstatstools.get_hvect_combinations(2,12)

    assert len(hvects2) == 12+1
    