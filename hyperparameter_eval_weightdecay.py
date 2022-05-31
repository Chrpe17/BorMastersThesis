#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:10:59 2022

@author: Carl Johan Danbjørg
Code to build evaluationplot based on loss.

To be used with models with weight decay as hyperparameter, as wd-level is
placed in title

"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

#metric=pd.DataFrame()
path='/Users/Data/Blue_ocean/Hyperparameters_r101/hypothesis2'

   
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
                metricpath=path+'/'+f.name+'/'
                fig, ax1 =plt.subplots()
                ax1.set_title(f'r50, learning rate: {lr}, gamma: {gamma}, weight decay: {wd}')
                ax1.set_ylabel('loss')
                ax1.grid(which='major', axis='y', linestyle='--',linewidth='0.5')
          
                try:             
                    metrics_df=pd.read_json(metricpath+'metrics.json', orient='records',lines=True)
                    mdf=metrics_df.sort_values('iteration')
                    if "total_loss" in mdf.columns:
                        print('total loss OK')
                        mdf1 = mdf[~mdf["total_loss"].isna()]
                        ax1.plot(mdf['iteration'], mdf['total_loss'],c='C0', label='train')
                    
                    if "validation_loss" in mdf.columns:
                        print('accuracy loss OK')
                        mdf1 = mdf[~mdf["validation_loss"].isna()]
                        ax1.plot(mdf1["iteration"], mdf1["validation_loss"], c="C1", label="validation")
                        
                    if "lr" in mdf.columns:
                        print('lr OK')
                        mdf2 = mdf[~mdf['lr'].isna()]
                        color2='C2'
                        ax2 =ax1.twinx()
                        ax2.set_yscale('log')
                        #ax2.set_ylim(0,0.1)
                        ax2.plot(mdf2["iteration"], mdf2["lr"],color=color2, label='learning rate')
                        #ax2.set_ylim(0,0.06)
                        #fig.tight_layout()
        
                except:
                    metrics_df=pd.read_json(metricpath+'metrics.json', orient='records',lines=False)
                    mdf=metrics_df.sort_values('iteration')
                    if "total_loss" in mdf.columns:
                        print('total loss OK')
                        mdf1 = mdf[~mdf["total_loss"].isna()]
                        ax1.plot(mdf['iteration'], mdf['total_loss'],c='C0', label='train')
                    
                    if "validation_loss" in mdf.columns:
                        print('val loss OK')
                        mdf1 = mdf[~mdf["validation_loss"].isna()]
                        ax1.plot(mdf1["iteration"], mdf1["validation_loss"], c="C1", label="validation")
                        
                    if "lr" in mdf.columns:
                        print('lr OK')
                        mdf2 = mdf[~mdf['lr'].isna()]
                        color2='C2'
                        ax2 =ax1.twinx()
                        ax2.set_yscale('log')
                        ax2.plot(mdf2["iteration"], mdf2["lr"],color=color2, label='learning rate')
                        #ax2.set_ylim(0,0.06)
                        #fig.tight_layout()
        
                # ax.set_ylim([0, 0.5])               
                
                #ax.legend()
                ax1.set_ylim(0,0.8)
                ax2.set_ylim(0.0000001,0.01)
                #ax2.ylim(0,0.1)            
                #ax1.set_ylabel('loss')
                ax2.set_ylabel('learning rate', color=color2)
                handles,labels = [],[]
                for ax in fig.axes:
                    for h,l in zip(*ax.get_legend_handles_labels()):
                        handles.append(h)
                        labels.append(l)
                
                ax1.legend(handles,labels, loc='upper left')
                plt.savefig(metricpath+"detailed_loss.png", dpi=300)
                plt.show()
 