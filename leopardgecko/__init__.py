
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

#Leopard Gecko module with functions and classes for
# processing, analysing and reporting 'predicted' data

# To be run from the command line or jupyter

#Code based in notebooks
# AvgPooling3DConsistencyData.ipynb and AnalyseAvgPoolResults.ipynb


# For showing nested loop progress in notebook
#from IPython.display import clear_output

#from .leopardgecko import *
# from .PredictedData import *
# from .ScoreData import *
# from .Pcrit import *
# from .metrics import *

# #Make the AvgPool optional?
# from .AvgPool import *

#from . import hvectstatstools #OK
# #To use the functions use import leopardgecko as lg ; lg.hvectstatstools.function()

# from . import PredictedData
from . import ScoreData, Pcrit, metrics, AvgPool
from . import hvectstatstools
from . import hvect_plot_2class, hvect_plot_3class

def lizzie():
    '''
    No code is proper without an Easter Egg
    '''
    print("Lizzie, The greatest Leopard Gecko in the Gecko world. If you ever met her, you would even say she glows.", \
        "She has geckifing powers from her petrifying stare.")
