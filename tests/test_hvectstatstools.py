'''
Copyright 2022 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Useful functions to get statistics of hvectors in data volumes

'''

import leopardgecko as lg
import random

#lg.hvectstatstools.testme()

def test_get_hvect_combinations():
    #Get hvector combinations in a 2-class, 12 way system
    hvects2 = lg.hvectstatstools.get_hvect_combinations(2,12)

    assert len(hvects2) == 12+1

    #random numbe of ways
    nways = random.randint(3,20)
    hvects2 = lg.hvectstatstools.get_hvect_combinations(2,nways)
    assert len(hvects2) == nways+1


    #Tests for 3-class
    
    hvects3 = lg.hvectstatstools.get_hvect_combinations(3,12)

    assert len(hvects3) == (13*14)/2
    # The combinations are a triangle.
    # Number of combinations, from base is
    # 13+12+11+...+1 = n(n+1)/2 with n=13

    #random number of ways
    nways = random.randint(5,20)
    hvects3 = lg.hvectstatstools.get_hvect_combinations(3,nways)

    assert len(hvects3) == ((nways+1)*(nways+2))/2


