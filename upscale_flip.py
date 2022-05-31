#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:51:49 2022

@author: Carl Johan Danbjørg

Code to upscale the category: "sink" in the robotdataset

Pseudo:
- Directory is scanned for images and labels
- Image with the label "sink" are selected
- Image and .json are copied
- The copied image is flipped and saved with suffix: "aug"
- In the copied json the bounding box is flipped and the imagereference updated

"""

from PIL import Image, ImageOps
import os
import json
import shutil

all_img_1={}

def json_iterator(file_dir):
    global all_img
    files=os.scandir(file_dir)
    #with os.scandir(img_dir) as dirs:
    #print(dirs)
    i=0
    for entry in files:
            split=os.path.splitext(entry)
            print(f'Split: {split}')
            #Tjek for previously augmented to prevent continous augmentation
            if 'augm' in str(split):
                print(f'Augm in: {split}')
                continue
            else:
                if split[1]=='.json':
                #print(entry)
                #filetype=split[1]
                    with open(entry.path) as json_file:
                        one_img=json.load(json_file)
                        en=entry.name
                        key=en[0:len(en)-5]
                        for shape in one_img['shapes']:
                            if shape['label']=="sink":
                                print('Found')
                                print(key)
                                flip_image(file_dir, key)
                                flip_json(file_dir, key)
                                break
                            else:
                                continue
                        #print(one_img)
                        all_img_1[en]=one_img
                        json_file.close()
                else:
                    continue
            i+=1
            print(i)
    return

def flip_image(file_dir,key):
    src=file_dir+'/'+key+'.jpg'
    dst=file_dir+'/'+key+'_augm.jpg'
    #print(f'Src: {src}')
    print(f'Dst: {dst}')
    shutil.copyfile(src,dst)
    #with open(dst,'w') as file:      
    img=Image.open(dst)
    flip=img.transpose(Image.FLIP_LEFT_RIGHT)
    #flip.show()
    #img.show()
    flip=flip.save(dst)
    
    return

def flip_json(file_dir,key):
    src=file_dir+'/'+key+'.json'
    dst=file_dir+'/'+key+'_augm.json'
    
    #print(f'Src: {src}')
    print(f'Dst: {dst}')
    shutil.copyfile(src,dst)
    with open(dst) as json_file:
        jsonfile=json.load(json_file)
        jsonfile['imagePath']=key+'_augm.jpg'
        #en=entry.name$
        for shape in jsonfile['shapes']:
            #print(shape)
            for points in shape['points']:
                #print(points)
                #for coordinates in points():
                #print(f'coordinat pre: {points}')
                points[0]=640-points[0]
                #print(f'coordinat post: {points}')
        #json_object=json.dumps(jsonfile)
    with open (dst,'w') as outfile:
        json.dump(jsonfile,outfile)

    return


#remember to update all folders in set: train, valid, test....
json_iterator('/Users/Data/Blue_ocean/Dataset/Labelled_images 3/valid')