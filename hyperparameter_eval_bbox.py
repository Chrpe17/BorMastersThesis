#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:10:59 2022

@author: Carl Johan Danbjørg
Code to plot the bbox_AP with loss (or any other column) across the iterations
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

#path to models that are to be analyzed (will scan directory)
path='/Users/Data/Blue_ocean/Hyperparameters_r101/Final model'
#Modelfamily to reference in title
model='r101'


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
            fig, ax1 =plt.subplots()
            ax1.set_title(f'model: {model} , learning rate: {lr}, gamma: {gamma}')
            ax1.grid(which='major', axis='y', linestyle='--',linewidth='0.5')
            #ax1.set_title(f)
            ax1.set_ylabel('loss')
            
            #color2='C2'
            #ax2 =ax1.twinx()
            #ax2.set_yscale('log')
            #ax2.set_ylim(0,0.02)
            #ax2.tick_params(axis='y', labelcolor=color2)
            #ax2.yscale('log')
            #ax1.set_ylim(0,3)
            
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

                if "bbox/AP" in mdf.columns:
                    print('bbox_AP OK')
                    mdf2 = mdf[~mdf['bbox/AP'].isna()]
                    color2='C2'
                    ax2 =ax1.twinx()
                    #ax2.set_yscale('log')
                    ax2.plot(mdf2["iteration"], mdf2["bbox/AP"],color=color2, label='bbox AP')

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
                    
                if "bbox/AP" in mdf.columns:
                    print('bbox/AP OK')
                    mdf2 = mdf[~mdf['bbox/AP'].isna()]
                    color2='C2'
                    ax2 =ax1.twinx()
                    #ax2.set_yscale('log')
                    ax2.plot(mdf2["iteration"], mdf2["bbox/AP"],color=color2, label='bbox AP')
                    #ax2.set_ylim(0,0.06)
                    #fig.tight_layout()

            # ax.set_ylim([0, 0.5])               
            
            #ax.legend()
            ax1.set_ylim(0,3)
            ax2.set_ylim(0,65)
            #ax2.ylim(0,0.1)            
            #ax1.set_ylabel('loss')
            ax2.set_ylabel('bbox AP', color=color2)
            ax2.tick_params(axis='y',labelcolor=color2)
            handles,labels = [],[]
            for ax in fig.axes:
                for h,l in zip(*ax.get_legend_handles_labels()):
                    handles.append(h)
                    labels.append(l)
            
            ax1.legend(handles,labels, loc='upper right')
            plt.savefig(metricpath+model+'_'+f.name+'_'+"detailed_bbox_2.png",dpi=300)
            plt.show()
