#Script to generate pdf report of prediction data consistency
#This python script must runs the notebook lg-genpredcsreport.ipynb

#Accepts command line arguments
#Generates a configuration file that will be used by jupyter notebook g-genpredcsreport.ipynb
# that will generate the report, in pdf file format.
#The pdf file that will be generated will be renamed to an appropriate filename


import argparse
import yaml
import os
import sys

parser = argparse.ArgumentParser()

#Positional arguments
parser.add_argument("datafilename", help="Input data filename, combined result from prediction. File must be in hdf5 format.")

#optional arguments
parser.add_argument("--out", help="Output filename (default:, datafilename.pdf)")

#average pooling arguments
parser.add_argument("--avgpwidth", type=int, default=512, \
    help="Chunk width (in index units) for the average pooling of consistency score calculation. Default: 512." )

parser.add_argument("--avgpkwidth" , type=int, default=256, \
    help="Kernel width (in index units) for the average pooling of consistency score calculation. Default: 256." )

parser.add_argument("--avgpstride" , type=int, default=128, \
    help="Stride (jumps in index units) for the average pooling of consistency score calculation. Default: 128." )

#Note that action='store_true' will default to False if not specified
parser.add_argument("--avgpforcerecalc" , action='store_true' ,\
    help="If calculated average pooled data already exists it will skip calculation of the average pool data unless this flag is set to true." )

parser.add_argument("--csweightingmethod" , default="MaxMinSquare" , \
    help="Sets the weigthing method used to calculate the consistency score from the data. Default: MaxMinSquare. Other options: MaxZeroSquare , None" )

parser.add_argument("--csvolscalculatefile", \
    help='''Filename containing user defined volumes (index units) where additional reporting of the
        consistency score is desired.
        The file must be in text format with values separated by commas
        and volumes sparated by newlines.
        Values should be written in order zmin, zmax, ymin, ymax, xmin, xmax
        ''')

parser.add_argument("--csroimethod", default="v1" , \
    help="Regions of interest reporting settings by version. Default: v1. Other options:TODO")

parser.add_argument("--configonly", action='store_true' , \
    help="Specify whether only config file is desired, hence not running the notebook and not producing th pdf file")

parser.add_argument("--dataoutfolder", \
    help="Define here where output intermediate files should be saved. Default is current folder.")

parser.add_argument("--dosdccalculation", \
    help="Do Sorensen-Dice coefficient calculation using filename given as the labels (in binary format)")

#Process arguments

args = parser.parse_args()

#check datafilename exists
if not os.path.exists(args.datafilename) :
    print("File could not be found. Exiting.")
    sys.exit()

outputfile=args.out
if outputfile is None:
    pathhead, pathtail = os.path.split(args.datafilename)
    pathname , ext = os.path.splitext(pathtail)
    outputfile = pathname + ".pdf"

print ( "Report will be saved in file {}".format(outputfile) )

#Create the configuration file
#Saves a dictionary object containing all the settings

#If no datafolder defined, use the current path as the folder
dataoutfolder = args.dataoutfolder
if dataoutfolder is None:
    dataoutfolder = os.getcwd()

configuration =  {"datafilename" : args.datafilename , \
    "outputfilepdf" : outputfile , \
    "avgpwidth" : args.avgpwidth ,\
    "avgpkwidth" : args.avgpkwidth , \
    "avgpstride" : args.avgpstride , \
    "avgpforcerecalc" : args.avgpforcerecalc, \
    "csweightingmethod" : args.csweightingmethod, \
    "csvolscalculatefile" : args.csvolscalculatefile , \
    "csroimethod" : args.csroimethod ,
    "dataoutfolder" : dataoutfolder, \
    "dosdccalculation": args.dosdccalculation \
}

# print( yaml.dump (configuration) )

#we are going to use the jupyter notebook and the configuration file in
#the same folder as this python code
thisfolder = os.path.dirname(__file__)
configfile = thisfolder + "/lg-genpredcsreport.yaml"
notebookfile = thisfolder + "/lg-genpredcsreport.ipynb"

#Create shared config file only
with open(configfile, 'w') as f:
    data = yaml.dump(configuration, f)
print("Config file saved : {} ".format(configfile) )

if not args.configonly:
    #Run jupyter nbconvert on lg-genpredcsreport.ipynb
    command_string = "jupyter nbconvert --no-input --execute --to pdf " + notebookfile
    os.system ( command_string )

    #When completed, rename the file to output filename
    command_rename = "mv " + thisfolder + "/lg-genpredcsreport.pdf " + outputfile
    os.system ( command_rename )
    print("Just as leopard geckos really like doing, the report {} was created.".format(outputfile) )

else:
    print("As specified, only config file was created, without running notebook and generating report" )