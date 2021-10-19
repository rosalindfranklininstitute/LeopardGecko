
#Leopard Gecko module with functions and classes for
# processing, analysing and reporting 'predicted' data

# To be run from the command line or jupyter

#Code based in notebooks
# AvgPooling3DConsistencyData.ipynb and AnalyseAvgPoolResults.ipynb


# For showing nested loop progress in notebook
#from IPython.display import clear_output

#from .leopardgecko import *
from .PredictedData import *
from .ScoreData import *
from .Pcrit import *

#Make the AvgPool optional?
from .AvgPool import *

def lizzie():
    '''
    No code is proper without an Easter Egg
    '''
    print("Lizzie, The greatest Leopard Gecko in the Gecko world. If you ever met her, you would even say she glows.", \
        "She has geckifing powers from her petrifying stare.")
