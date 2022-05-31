#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:40:08 2022

@author: Carl Johan Danbjørg

Code to distribute images and json into train-, validation- and testset
"""
import os
import shutil
import json
from PIL import Image
import pandas as pd
import random

i=0
path='/Users/Data/Blue_ocean/Dataset/Labelled_images 4/Test/'
with os.scandir(path) as dirs:      #skanning the folder for images
    for entry in dirs:              #iterating through the list of files
        if entry.is_file():         #avoiding manipulation of folders
            try:    
                b=os.path.basename(entry)
                #print(f'B is: {b}')
                split=os.path.splitext(b)
                #print(f'Split is: {split}')
                
                if split[1]=='.jpg':
                    #print(f'Entry is: {entry}')
                    #establish segments    
                    pools = ['train','valid','test']
                    for p in pools:
                        os.makedirs(path+r'/'+p,exist_ok=True)
                    #setting weights for distribution of files
                    distribution = [80,20,0]
                    #assign a pool to the specific file
                    pool = random.choices(pools, weights = distribution,k=1)
                    #separate according to labels
                    jsonpath=path+split[0]+'.json'
                    #print(jsonpath)
                    if os.path.isfile(jsonpath):
                        oldpath=path+split[0]+'.jpg'
                        newpath=path+r'/'+pool[0]+r'/'+split[0]+'.jpg'
                        shutil.move(oldpath,newpath)
                        oldpath=path+split[0]+'.json'
                        newpath=path+r'/'+pool[0]+r'/'+split[0]+'.json'
                        shutil.move(oldpath,newpath)
                        #print('ok')
                    else:
                        print(f'Err in {jsonpath}')
                        i+=1
                
                else:
                    #print('Else')
                    continue
            except:
                #print(f'Err in:{entry}')
                continue
print(i)
            