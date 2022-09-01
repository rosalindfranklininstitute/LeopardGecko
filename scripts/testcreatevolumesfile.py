import yaml
import os
import sys

mySegmentedVolumes = []

mySegmentedVolumes.append( (897,1153 , 185,441 , 1750,2006) )

mySegmentedVolumes.append( (512,768 , 1280,1536 , 1536,1792) )
mySegmentedVolumes.append( (0,256 , 1536,1792 , 1792,2048) )
mySegmentedVolumes.append( (0,256 , 1024,1280 , 512,768) )
mySegmentedVolumes.append( (1536,1792 , 256,512 , 0,256) )

#mySegmentedVolumes.append( (684,1068 , 2012,2396 , 457,841) )
mySegmentedVolumes.append( (684,1068 , 2012,2396 , 455,839) ) 

#mySegmentedVolumes.append( (1067,1991 , 1074,1458 , 1354,1738) )
mySegmentedVolumes.append( (1067,1991 , 1074,1458 , 1352,1736) )

with open('testcreatevolumesfile.yaml', 'w') as f:
    data = yaml.dump(mySegmentedVolumes, f)