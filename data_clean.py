#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:55:03 2022

@author: Carl Johan Danbjørg


Code to allign imagedata:
    - from .png to .jpg
    - rename s A=austria and I=Italy
    
Analyze for label frequency

Distribute labels equally to Train, Validation and Test

"""

import os
import shutil
import json
from PIL import Image
import pandas as pd
import random

#dictionary for all images
all_img={}

#get all files from Portugal and rename
def rename(directory):
    a=0    
    for x in os.listdir(directory):
        if x.startswith("image"):
            a+=1
            z=x[5:len(x)]
            prefix='I'
            os.rename(directory+x,directory+prefix+z)
    print(f'{a} images in {directory} renamed')
    return
        
#get all files with format .png and replace with a .jpg version
def convert_png(directory,error_directory):
    #make err directory if not available
    os.makedirs(error_directory,exist_ok=True)
    b=0; c=0; d=0
    for y in os.listdir(directory):
        if y.endswith('.png'):
            b+=1
            #beneath will err if there is no .json associated with the image
            #i.e. images without labels
            try:
                #b+=1
                image_png = Image.open(directory+y)
                
                #get the filename without suffix
                v=directory+y[0:len(y)-4]
                
                #correct the imagefilename in the json file
                linebyline=[]
                with open(v+'.json', 'r') as j:
                    #path={'imagePath':y+".jpg"}
                    js=json.load(j)
                    #js['image_id']=y[0:len(y)-4]+'.jpg'
                    linebyline.append(js)
                    #js['imagePath']=y[0:len(y)-4]+'.jpg'
                    for line in linebyline:
                        #print(line['imagePath'])
                        line['imagePath']=y[0:len(y)-4]+'.jpg'
                        line['image_id']=y[0:len(y)-4]+'.jpg'
                
                with open(v+'.json', 'w') as j:
                    json.dump(linebyline[0],j)
                
                #save a .jpg version
                image_png.save(v+'.jpg')
                
                #delete the .png file
                os.remove(directory+y)
                d+=1
            except:
                c+=1
                shutil.move(directory+y,error_directory+y)
                print(f'Convertionerror with image{y}')
    print(f'{b} png-files found')
    print(f'{c} png-files had err')
    print(f'{d} png-files was converted')
    return


#build dictionary for images
def json_collection(img_dir):
    global all_img
    files=os.scandir(img_dir)
    #with os.scandir(img_dir) as dirs:
    #print(dirs)
    for entry in files:
        split=os.path.splitext(entry)
        if split[1]=='.json':
            #print(entry)
            one_img={}
            #filetype=split[1]
            with open(entry.path) as json_file:
                    one_img=json.load(json_file)
                    #print(one_img)
                    en=entry.name
                    all_img[en]=one_img
                    json_file.close()
        else:
            continue
    return

#Function to compile all labels in dataframe for analytical purposes
all_img_df=pd.DataFrame(columns=['Image', 'Label', 'Points'])
def labelsdataframe():
    global all_img_df
    #Iterating all json listed in dictionary of images
    for image in all_img:
        img = all_img[image]['imagePath']
        #Iterating all shapes/labels for each image
        for shape in all_img[image]['shapes']:
            i={}
            i['Image']=img
            i['Label']= shape['label']
            i['Points']= [shape['points']]
            df_temp=pd.DataFrame(i)
            all_img_df=pd.concat([all_img_df,df_temp],ignore_index=True)
    return

def statistics():
    all_img_df['Country']=all_img_df['Image'].str.split('_',expand=True)[0]
    labels=all_img_df.groupby(['Country','Label'])['Image'].count()
    print(labels)
    
    return

def filesplit(path):
    with os.scandir(path) as dirs:      #skanning the folder for images
        for entry in dirs:              #iterating through the list of files
            if entry.is_file():         #avoiding manipulation of folders
                try:    
                    b=os.path.basename(entry)
                    print(f'B is: {b}')
                    split=os.path.splitext(b)
                    print(f'Split is: {split}')
                    
                    if split[1]=='.jpg':
                        print(f'Entry is: {entry}')
                        #establish segments    
                        pools = ['train','valid','test']
                        for p in pools:
                            os.makedirs(path+r'/'+p,exist_ok=True)
                        #setting weights for distribution of files
                        distribution = [80,20,0]
                        #assign a pool to the specific file
                        pool = random.choices(pools, weights = distribution,k=1)
                        #separate according to labels
                        oldpath=path+split[0]+'.jpg'
                        newpath=path+r'/'+pool[0]+r'/'+split[0]+'.jpg'
                        shutil.move(oldpath,newpath)
                        oldpath=path+split[0]+'.json'
                        newpath=path+r'/'+pool[0]+r'/'+split[0]+'.json'
                        shutil.move(oldpath,newpath)
                    
                    else:
                        print('Else')
                        continue
                except:
                    continue
        return

#folder that contains full imageset                 
allimagefolder='/Users/Data/Blue_ocean/Dataset/Labelled_images 5/'
#folder to caontain images that err (due to laking notations)
errfolder=allimagefolder+'Err/'

rename(allimagefolder)
convert_png(allimagefolder,errfolder)
json_collection(allimagefolder)
labelsdataframe()
filesplit(allimagefolder)

print(statistics())