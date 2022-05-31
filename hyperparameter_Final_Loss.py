#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:10:59 2022

@author: Carl Johan Danbjørg
Code to evaluate the final models with respect to loss, accuracy (or any other column in 
metrics.json)

"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

#metric=pd.DataFrame()
path='/Users/Data/Blue_ocean/Hyperparameters_r101/Final model'

   
for f in os.scandir(path):
    if f.is_dir():
            #w=['0.00025', '0.0005']
            namesplit=f.name.split('_')
            frame=namesplit[0]
            version=namesplit[1]
            if frame.startswith('r'):
                    metricpath=path+'/'+f.name+'/'
                    print(metricpath)
                    #mdf.drop_duplicates()
                    fig, ax1 =plt.subplots()
                    ax1.set_title(f'{frame}, version: {version}')
                    ax1.set_ylim(0,1)
                    ax1.grid(which='major', axis='y', linestyle='--',linewidth='0.5')
                    
                    color2='C2'
                    ax2 =ax1.twinx()
                    #ax2.set_yscale('log')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    ax2.set_ylim(0.8,1.05)
                    #ax2.yscale('log')
                    #ax1.set_ylim(0,3)
                    
                    try:             
                        metrics_df=pd.read_json(metricpath+'metrics.json', orient='records',lines=True)
                        mdf=metrics_df.sort_values('iteration')
                        if "total_loss" in mdf.columns:
                            print('total loss OK')
                            mdf1 = mdf[~mdf["total_loss"].isna()]
                            ax1.plot(mdf1["iteration"], mdf1["total_loss"], c="C1", label="Total loss")
                            
                        if "fast_rcnn/cls_accuracy" in mdf.columns:
                            print('fast_rcnn/cls_accuracy OK')
                            join=mdf.append(mdf1)
                            mdf2 = join.drop_duplicates(keep=False, ignore_index=True)
    
                            ax2.set_ylim()
                            fig.tight_layout()
                            ax2.plot(mdf1['iteration'], mdf1['fast_rcnn/cls_accuracy'],c='C2', label='fast_rcnn/cls_accuracy')
                        
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

                        if "total_loss" in mdf.columns:
                            print('total loss OK')
                            mdf1 = mdf[~mdf["total_loss"].isna()]
                            ax1.plot(mdf1["iteration"], mdf1["total_loss"], c="C1", label="Total loss")
                        if "total_loss" in mdf.columns:
                            print('total loss OK')
                            join=mdf.append(mdf1)
                            mdf2 = join.drop_duplicates(keep=False, ignore_index=True)
                            ax2.plot(mdf1["iteration"], mdf1["fast_rcnn/cls_accuracy"],color=color2, label='Fast_rcnn/cls_accuracy')
                            ax2.set_ylim()
                            fig.tight_layout()
                            
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
                    ax1.set_ylabel('Total loss')
                    ax2.set_ylabel('Fast_rcc/cls_accuracy', color=color2)
                    handles,labels = [],[]
                    for ax in fig.axes:
                        for h,l in zip(*ax.get_legend_handles_labels()):
                            handles.append(h)
                            labels.append(l)
                    
                    ax1.legend(handles,labels, loc='upper left')
                    plt.savefig(metricpath+"loss_accuracy.png",dpi=300)
                    plt.show()
  