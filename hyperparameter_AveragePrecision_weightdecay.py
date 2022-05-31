#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:10:59 2022

@author: Carl Johan Danbjørg
Code to plot individual columns from metric.json (typically loss or precision)
against iterations. Also used with Learning rate plotted on secondary scale.

This version uses weight decay as a parameter in the plot title and filename, for
reference.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

#metric=pd.DataFrame()
path='/Users/Data/Blue_ocean/Hyperparameters_r101/other'

   
for f in os.scandir(path):
    if f.is_dir():
            w=['0.00025', '0.0005']
            namesplit=f.name.split('_')
            lr=namesplit[0]
            gamma=namesplit[1]
            if len(namesplit)>2:
                wd=namesplit[2]
                if wd in w:
                    print(namesplit[2])
                    
                    #print(lr)
                    #print(gamma)
                    metricpath=path+'/'+f.name+'/'
                    print(metricpath)
                    #mdf.drop_duplicates()
                    fig, ax1 =plt.subplots()
                    ax1.set_title(f'r50, learning rate: {lr}, gamma: {gamma}, weight decay: {wd}')
                    ax1.set_ylim(0.9,1)
                    ax1.grid(which='major', axis='y', linestyle='--',linewidth='0.5')
                    
                    #color2='C2'
                    #ax2 =ax1.twinx()
                    #ax2.set_yscale('log')
                    #ax2.tick_params(axis='y', labelcolor=color2)
                    #ax2.set_ylim(0,0.1)
                    #ax2.yscale('log')
                    #ax1.set_ylim(0,3)
                    
                    try:             
                        metrics_df=pd.read_json(metricpath+'metrics.json', orient='records',lines=True)
                        mdf=metrics_df.sort_values('iteration')
                        if "validation_loss" in mdf.columns:
                            print('accuracy loss OK')
                            mdf1 = mdf[~mdf["validation_loss"].isna()]
                            ax1.plot(mdf1["iteration"], mdf1["fast_rcnn/cls_accuracy"], c="C1", label="Validation accuracy")
                            
                        if "total_loss" in mdf.columns:
                            print('total loss OK')
                            join=mdf.append(mdf1)
                            mdf2 = join.drop_duplicates(keep=False, ignore_index=True)
                            ax1.plot(mdf2['iteration'], mdf2['fast_rcnn/cls_accuracy'],c='C0', label='Train Accuracy')
                        
        # =============================================================================
        #                 if "lr" in mdf.columns:
        #                     print('lr OK')
        #                     mdf2 = mdf[~mdf['lr'].isna()]
        #                     ax2.plot(mdf2["iteration"], mdf2["lr"],color=color2, label='learning rate')
        #                     #ax2.set_ylim(0.00001,0.06)
        #                     fig.tight_layout()
        # 
        # =============================================================================
                    except:
                        metrics_df=pd.read_json(metricpath+'metrics.json', orient='records',lines=False)
                        mdf=metrics_df.sort_values('iteration')
                        if "validation_loss" in mdf.columns:
                            print('accuracy loss OK')
                            mdf1 = mdf[~mdf["validation_loss"].isna()]
                            ax1.plot(mdf1["iteration"], mdf1["fast_rcnn/cls_accuracy"], c="C1", label="Validation accuracy")
                            
                        if "total_loss" in mdf.columns:
                            print('total loss OK')
                            join=mdf.append(mdf1)
                            mdf2 = join.drop_duplicates(keep=False, ignore_index=True)
                            ax1.plot(mdf2['iteration'], mdf2['fast_rcnn/cls_accuracy'],c='C0', label='Train Accuracy')
                            
        # =============================================================================
        #                 if "lr" in mdf.columns:
        #                     print('lr OK')
        #                     mdf2 = mdf[~mdf['lr'].isna()]
        #                     ax2.plot(mdf2["iteration"], mdf2["lr"],color=color2, label='learning rate')
        #                     ax2.set_ylim(0.00001,0.06)
        #                     fig.tight_layout()
        # =============================================================================
        
                    # ax.set_ylim([0, 0.5])               
                    
                    #ax.legend()
                    #ax2.ylim(0,0.1)            
                    ax1.set_ylabel('Class accuracy')
                    #ax2.set_ylabel('learning rate', color=color2)
                    handles,labels = [],[]
                    for ax in fig.axes:
                        for h,l in zip(*ax.get_legend_handles_labels()):
                            handles.append(h)
                            labels.append(l)
                    
                    ax1.legend(handles,labels, loc='lower right')
                    plt.savefig(metricpath+"detailed_accuracy.png")
                    plt.show()
                    #if lr=='0.00025':
                    #    break
