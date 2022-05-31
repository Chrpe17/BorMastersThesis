#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:56:21 2022

@author: Carl Johan Danbjørg

Code to analyze an imageset to:
    - Summarize labels available in images and pools
    - Summarize labels in ground truth and compare with inferences
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

#build dictionary for images in directory: img_dir
def json_collection(img_dir):
    #global all_img
    all_img={}
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
    return all_img


#Function to compile all labels in dataframe for analytical purposes
all_img_df=pd.DataFrame(columns=['Pool','Dataset','Image', 'Label', 'Points'])
def labelsdataframe(directory):
    all_img=json_collection(directory)
    global all_img_df
    #Iterating all json listed in dictionary of images
    for image in all_img:
        img = all_img[image]['imagePath']
        #Iterating all shapes/labels for each image
        for shape in all_img[image]['shapes']:
            i={}
            i['Pool']=directory.split('/')[-1]
            i['Image']=img
            i['Label']= shape['label']
            i['Points']= [shape['points']]
            if img[0] =='I':
                i['Dataset']='Portugal'
            elif img[0] == 'A':
                i['Dataset']='Austria'
                
            df_temp=pd.DataFrame(i)
            all_img_df=pd.concat([all_img_df,df_temp],ignore_index=True)
    return

#Iterate through the subfolders of the directory and pool images
def iterator(path):
    #sets=['train','valid','test']
    sets=['train','test','valid']
    for s in sets:
        direct=path+s
        labelsdataframe(direct)
    return

#to analyze prior to split in training, validation and test
def iterator_without_split(path):
    sets=['']
    for s in sets:
        direct=path+s
        labelsdataframe(direct)
    return

#line to activate above functions, with reference to dataset-folder
iterator('/Users/Data/Blue_ocean/Dataset/Final_Dataset/')
#iterator_without_split('/Users/Data/Blue_ocean/Dataset/Labelled_images_1')

#%% section with code to build various dataframes

#def table():
all_img_multi=all_img_df.set_index(['Dataset','Pool'])
labels_by_dataset=all_img_df.groupby(['Label']).count()
Images_by_dataset=all_img_df.groupby(['Image']).count()
Images_by_dataset2=all_img_multi.groupby(['Pool','Image']).count()
Images_by_dataset3=Images_by_dataset2.filter(['Dataset','Image']).count()

img_grouped=all_img_df.groupby(['Dataset','Label','Pool']).size().unstack()

#%% Collecting inferences from different models

def inferencematrix(inferenceset):
    #get the validation set from the set of all images
    validation_set=all_img_df[all_img_df['Pool']=='test']
    
    #count labels by image and adapt dataframe
    labels_by_img_val=validation_set.groupby(['Image','Label']).count()
    labels_by_img_val.rename(columns={'Pool':'Ground'},inplace=True)
    labels_by_img_val=labels_by_img_val.drop(columns=['Dataset','Points'])
    
    #get the inferences
    def import_inteferences(inferenceset):
        confusion=labels_by_img_val
        for key in inferenceset.keys():
            #read the json file
            print(key)
            try:
                print(inferenceset[key])
                inference=pd.read_json(inferenceset[key],orient='records',lines=False)
                #modify the full textstring of image to the imagename
                inference[['image_id','Image']]=inference['image_id'].str.rsplit("/", n=1, expand=True)
                #remove irrelevant columns
                inference=inference.drop(columns=['image_id','bbox'])
                #rename column with label for consistency
                inference.rename(columns={'category_id':'Label'},inplace=True)
                #convert label from numeric reference to actual label
                inference.replace({'Label':{0:'sink',1:'door',2:'bed',3:'screen',4:'socket'}},inplace=True)
                #convert individual labels to groups pr image
                inference=inference.groupby(['Image','Label']).count()
                #rename column with score to inferencekey
                inference.rename(columns={'score':key},inplace=True)
                inference[key]=inference[key].astype(int) 
                #add the modelinference to the table with groundtruth to build confusionmatrix
                confusion=confusion.join(inference,how='outer')
                print(key)
                print('Try')
        
            except:
                inference=pd.read_json(inferenceset[key],orient='records',lines=True)
                #modify the full textstring of image to the imagename
                inference[['image_id','Image']]=inference['image_id'].str.rsplit("/", n=1, expand=True)
                #remove irrelevant columns
                inference=inference.drop(columns=['image_id','bbox'])
                #rename column with label for consistency
                inference.rename(columns={'category_id':'Label'},inplace=True)
                #convert label from numeric reference to actual label
                inference.replace({'Label':{0:'sink',1:'door',2:'bed',3:'screen',4:'socket'}},inplace=True)
                #convert individual labels to groups pr image
                inference=inference.groupby(['Image','Label']).count()
                #rename column with score to inferencekey
                inference.rename(columns={'score':key},inplace=True)
                inference[key]=inference[key].astype(int)
                
                #add the modelinference to the table with groundtruth to build confusionmatrix
                confusion=confusion.join(inference,how='outer')
                #print(confusion)     
                print(key)
                print('Except')
        return confusion
     
    confusionmatrix=import_inteferences(inferenceset)
    #get list of models for naming of columns
    models=list(inferenceset.keys())
    #calculate meanvalue for labelpredictions
    confusionmatrix['Mean']=confusionmatrix[models].mean(axis=1)
    #compare mean to Ground to identify images with errors
    measure=[confusionmatrix['Ground']==confusionmatrix['Mean']]
    result=['Full congruens']
    confusionmatrix['Congruens']=np.select(measure,result,'Err')
    confusionmatrix=confusionmatrix.fillna(0)
    
    labels_correct=confusionmatrix['Congruens'].value_counts()['Full congruens']
    labels_incorrect=confusionmatrix['Congruens'].value_counts()['Err']
    print(f'Labels correct: {labels_correct}, labels incorrect: {labels_incorrect}')
    
    
    return confusionmatrix

#Dictionary to hold references to the models which should be analyzed
inferenceset={'Final r50':'/Users/Data/Blue_ocean/Hyperparameters_r101/Final model/0.05_0.5_00025/coco_instances_results.json',
              #'Final r101':'/Users/Data/Blue_ocean/Hyperparameters_r101/Final model/0.07_0.1_0.0005/coco_instances_results.json',
              'Final r101':'/Users/Data/Blue_ocean/Hyperparameters_r101/Final model/custom_predict_1/coco_instances_results.json',
              'Base r101':'/Users/Data/Blue_ocean/Hyperparameters_r101/Final model/r101_base/coco_instances_results.json',
              'Base r50':'/Users/Data/Blue_ocean/Hyperparameters_r101/Final model/r50_base/coco_instances_results.json'}

confusionmatrix=inferencematrix(inferenceset)

#plot the distribution of labels labels
sns.countplot(x='Pool', hue='Label',data=all_img_df)
sns.catplot(x='Pool', hue='Label',col='Dataset',kind='count',data=all_img_df)

#%%
uniqueimages=all_img_df['Image'].unique()
uniquedf=all_img_df.copy(deep=True)
uniquedf=uniquedf.drop_duplicates(subset=['Image'])
uniquedf_original=uniquedf[~uniquedf['Image'].str.contains('augm')]
uniquedf_augmented=uniquedf[uniquedf['Image'].str.contains('augm')]

#function to identify if image has been augmented or is an original
def augmented(row):
    
    if 'augm' in row['Image']:
        return 'Augmented'
    else:
        return 'Original'
    return 'L'


data = all_img_df
all_img_df['Augmented'] = data.apply(lambda row: augmented(row), axis=1)
print(data)