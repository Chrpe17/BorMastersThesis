#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:10:59 2022

@author: Carl Johan Danbjørg

Kode to import all metrices.json in a directory (subset of models) and
identify the iterations registered, that
either has lowest loss, highest accuracy or highest bbox_AP - and based on
that compare different models
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

#metric=pd.DataFrame()
path='/Users/Data/Blue_ocean/Hyperparameters_r101/Final model'

#target to control if lowest loss or highest accuracy are defining measure,
#valid keys = 'loss, 'accuracy', 'bbox_AP'
target='bbox_AP'

best_df=pd.DataFrame()

for f in os.scandir(path):
    if f.is_dir():
            namesplit=f.name.split('_')
            lr=namesplit[0]
            gamma=namesplit[1]
            #print(lr)
            #print(gamma)
            metricpath=path+'/'+f.name+'/'
            print(metricpath)
            #mdf.drop_duplicates()
            try:
                mdf=pd.read_json(metricpath+'metrics.json', orient='records',lines=False)
                mdf.columns=mdf.columns.str.replace('[/]','_')
                mdf=mdf[~mdf['validation_loss'].isna()]
                if target == 'loss':                    
                    metrics_df_target=mdf[mdf.validation_loss == mdf.validation_loss.min(axis=0)]
                if target == 'accuracy':
                    metrics_df_target=mdf[mdf.fast_rcnn_cls_accuracy == mdf.fast_rcnn_cls_accuracy.max(axis=0)]
                if target == 'bbox_AP':
                    metrics_df_target=mdf[mdf.bbox_AP == mdf.bbox_AP.max(axis=0)]
                metrics_df_target.insert(0,'model',f.name)
                #metrics_df_target=metrics_df_target.iloc[:,[0,11,19,20,21,22,23,24,25,26,27,28,29,30]]
                best_df=pd.concat([best_df, metrics_df_target],ignore_index=True, sort=False)
                print('IF true')
                
                
            except:
                mdf=pd.read_json(metricpath+'metrics.json', orient='records',lines=True)
                mdf.columns=mdf.columns.str.replace('[/]','_')
                mdf=mdf[~mdf['validation_loss'].isna()]
                if target == 'loss':                    
                    metrics_df_target=mdf[mdf.validation_loss == mdf.validation_loss.min(axis=0)]
                if target == 'accuracy':
                    metrics_df_target=mdf[mdf.fast_rcnn_cls_accuracy == mdf.fast_rcnn_cls_accuracy.max(axis=0)]
                if target == 'bbox_AP':
                    metrics_df_target=mdf[mdf.bbox_AP == mdf.bbox_AP.max(axis=0)]
                metrics_df_target.insert(0,'model',f.name)
                #metrics_df_target=metrics_df_target.iloc[:,[0,11,19,20,21,22,23,24,25,26,27,28,29,30]]
                best_df=pd.concat([best_df, metrics_df_target],ignore_index=True, sort=False)
                print('Else true')

    else:
        continue
    
outpath=path+'/'+target+'.csv'
best_df.to_csv(outpath,sep=';')

#best_df=best_df.iloc[0:0]            
            


#path='/Users/Data/Blue_ocean/Hyperparameters/Metrices'
#iterator(path)