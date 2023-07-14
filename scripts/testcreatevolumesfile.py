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