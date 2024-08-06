#!/usr/bin/env python
# coding: utf-8

"""
Purpose: 
Run classification ML on map data from TAG.  
Seabed massiv sulfide case

Created on Fri Mar 01 2021
Programmed: Ketil Hokstad, 
            modified to fit ML classification model by Stine Ekrheim (summer 2021)
"""
#------------------------------------------------------
#  Imports
#------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
import json
import pickle
import matplotlib.cm as cm
import os

# KetilH stuff
import ml.greg as greg
from gravmag.meta import MapData
import khio

#From Stine (summer intern 2021)
import ml.gclas as gclas

#--------------------------
# Folders
#--------------------------

cdir = '../data_SMS/'

pdir = 'png_SMS/'
if not os.path.isdir(pdir): os.mkdir(pdir)

block = True

#---------------------------------------------------
# Read input data
#---------------------------------------------------

# Polygons
[poly_tag] =  pickle.load(open(cdir + 'TAG_polygons.pkl','rb'))

# Samples: features and target
in_file = cdir + 'TAG_AUV_Target_and_Features_Repick.xlsx'
df_smp_in = pd.read_excel(pd.ExcelFile(in_file))

# Features on grid for prediction:
in_file_map = cdir + 'AUV_MBES_Mag_UTM23N_decim.pkl'
with open(in_file_map, 'rb') as fid: [auv] = pickle.load(fid)

feat_list_all = ['bat', 'bs', 'slope', 'crv', 'crv_g', 'bss', 
                 'tma', 'tma_lo_k','tma_hi_k']
feat_unit_all = [' [m]',' [???]',' [m/m]',' [1/m2]',' [1/m2]',' [???/m]',
                 ' [nT]',' [nT]',' [nT]']

# Column keys and units:
key_x, unit_x = 'x', ' [m]'
key_y, unit_y = 'y', ' [m]'
key_t, unit_t = 'sms', ' [-]'

# Fix some shit
df_grd_in = pd.DataFrame(columns=[key_x,  key_y] + feat_list_all)
jnd = np.isfinite(auv.grd[0])
df_grd_in[key_x] = auv.gx[jnd]
df_grd_in[key_y] = auv.gy[jnd]
for jj, key in enumerate(feat_list_all):
    df_grd_in[key] = auv.grd[jj][jnd]

#----------------------------------------
#   Set job parameters
#---------------------------------------

# Which test to run:
krun, version = 1, 'b'   # version = empty or a, b, c etc
print(f'### Run {krun}{version}')
if   krun ==  0: # No good
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
     split, var = 'Random',  'MLE'
elif krun ==  1: #  
     ind_feat_use = [0, 1, 2, 3, 6]
     split, var = 'Random',  'MLE'
elif krun ==  2: #  
     ind_feat_use = [1, 2, 3, 6]
     split, var = 'Random',  'MLE'
elif krun ==  3: #  Spatial
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
     split, var = 'Spatial',  'MLE'
elif krun ==  4: #  
     ind_feat_use = [0, 1, 2, 3]
     split, var = 'Random',  'MLE'

# Train model without MIR?
trash_MIR = version == 'b'

# Out put files
crun = 'run' + str(krun) + version
out_file_model = '_TAG_AUV_ML_regression_model.pkl'
out_file_pred  = '_TAG_AUV_Predicted_Maps.pkl' 

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [feat_unit_all[jj] for jj in ind_feat_use]

# ML pars:
abu_scale = 1.0
targ_lo, targ_hi = 0, 2 # prospectivity flag is 0,1,2 = none, active, inactive
test_size = 0.20
n_halluc  = 0
scaling = 'Standardization' #Standardization, Normalization or None

#------------------------------------------------------------
#  Thrash some sample data?
#------------------------------------------------------------

#df_smp_in = df_smp_in[(df_smp_in[key_t] >= targ_lo) &  (df_smp_in[key_t] <= targ_hi)]

# Thrash MIR (use as validation)
if trash_MIR:
    print('Trash MIR')
    df_smp_in = df_smp_in[~df_smp_in['name'].str.contains('MIR')]

# Get the features of interest:
df_smp = df_smp_in[[key_x, key_y, key_t] + feat_list].copy()
df_grd = df_grd_in[[key_x, key_y] + feat_list].copy()

#------------------------------------------------------------
#   Add some hallucinatory evidence for the failure case
#------------------------------------------------------------

if n_halluc > 0:
    
    df_smp = greg.add_halluc(df_grd, df_smp, n_halluc, targ_lo, key_t, 
                             verbose=1, kplot=True)

#------------------------------------------------------------
#   Feature engineering
#------------------------------------------------------------

# Impute nans
df_grd.fillna(0, inplace=True)

if scaling == 'Standardization':
    mean = np.mean(df_smp[feat_list],axis = 0)
    std = np.std(df_smp[feat_list],axis = 0)
    df_smp[feat_list] = (df_smp[feat_list]-mean)/std
    df_grd[feat_list] = (df_grd[feat_list]-mean)/std
elif scaling == 'Normalization':
    minvalue = np.min(df_smp[feat_list],axis = 0)
    maxvalue = np.max(df_smp[feat_list],axis = 0)
    df_smp[feat_list] = (df_smp[feat_list]-minvalue)/(maxvalue-minvalue)
    df_grd[feat_list] = (df_grd[feat_list]-minvalue)/(maxvalue-minvalue)

#------------------------------------------------------------
#   Train ML models for classification:
#     o etc : model for estimating the class (extra trees classifier)
#     o random or spatial train/test split
#------------------------------------------------------------

# Init classifier objects:
n_est = 500
etc  = ens.ExtraTreesClassifier(n_estimators=n_est)

# Cross validation train/test
etc, test = gclas.fit_cv(etc, df_smp, key_t, 
                         key_x=key_x, key_y=key_y, split=split, 
                         verbose=1, kplot=True, qc_cv=True)

# Fix the title of the cluster plot
try:  
    ax = test['fig_clu'].get_axes()[0]
    ax.set_title(f'run {krun}{version}: kMeans clustering')
    test['fig_clu'].savefig(pdir + crun + '-kMeans_clusters.png')            
except: 
    pass

#------------------------------------------------------------
#   Predict target from feature maps
#------------------------------------------------------------

df_pred = gclas.predict_ml(etc, df_grd, key_t,
                          key_x=key_x, key_y=key_y, verbose=1)

#Get different colors (Problems with the heatmap otherwise)
numb_dict = {}
for ii, el in enumerate(np.unique(df_smp[key_t])):
    numb_dict[el] = ii

print(numb_dict)
# Put the datafram back in a regular grid (df_pred is a pd.DatFrame)
grd_pred = MapData(auv.x, auv.y, auv.z, 
                   [np.empty_like(auv.grd[0],dtype=object)])

grd_pred.grd[0].fill('nan')
grd_pred.grd[0][jnd] = df_pred['pred']

#------------------------------------------------------------
#   Write results
#------------------------------------------------------------

# Map prediciton of nodule abundance
with open(cdir + crun + out_file_pred, 'wb') as wrt:
    pickle.dump([grd_pred], wrt)

# ML regression models for mean and variance:
with open(cdir + crun + out_file_model,'wb') as wrt:
    pickle.dump([etc, test], wrt)

#-----------------------------------------------------------
#   Some awkward shit to get nans back in the grid
#   Plot functions need dataframes as input (fix later)
#------------------------------------------------------------

# ML predictions
df_pred_nan = pd.DataFrame(columns=df_pred.columns)
df_pred_nan[key_x] = grd_pred.gx.flatten()
df_pred_nan[key_y] = grd_pred.gy.flatten()
df_pred_nan['pred'] = grd_pred.grd[0].flatten()

# Feature maps
df_grd_nan = pd.DataFrame(columns=[key_x, key_y] + feat_list_all)
df_grd_nan[key_x] = auv.gx.flatten()
df_grd_nan[key_y] = auv.gy.flatten()
for ii, key in enumerate(feat_list_all):
    df_grd_nan[key] = auv.grd[ii].flatten()
        
#------------------------------------------------------------
#   Plot inputs 
#------------------------------------------------------------

# Zoom on maps
xmin, xmax, ymin, ymax, zoom =  auv.x[1],  auv.x[-1], auv.y[1],  auv.y[-1], '_zoom3'
print(zoom)

# Just for plotting labels:
clas_name, srun = str(etc.__str__).split()[3], crun + ': '

# TAG pseudo-samples
title = 'SMS abundance [-]'

fig_ts = gclas.plot_targ_samp(df_smp, key_t,numb_dict=numb_dict, key_x=key_x, key_y=key_y,
                             vmin=0, vmax=2, title=title, poly=poly_tag,
                             unit_x=unit_x, unit_y=unit_y, unit_t=unit_t)

# Some clipping to avoid fucked up color scales
df_grd_nan['bs'].clip(lower=50, upper=200, inplace=True)
df_grd_nan['slope'].clip(lower=0, upper=1.5, inplace=True)
df_grd_nan['crv'].clip(lower=-0.04, upper=0.04, inplace=True)
df_grd_nan['crv_g'].clip(lower=-0.2e-3, upper=0.2e-3, inplace=True)
df_grd_nan['bss'].clip(lower=0, upper=1, inplace=True)
df_grd_nan['tma'].clip(lower=-3500, upper=3500, inplace=True)
df_grd_nan['tma_lo_k'].clip(lower=-3500, upper=3500, inplace=True)
df_grd_nan['tma_hi_k'].clip(lower=-1000, upper=1000, inplace=True)

# Plot feature maps:
title = 'Feature maps'
fig_fm = greg.plot_feat_maps(df_grd_nan, key_x=key_x, key_y=key_y,
                             nx = auv.nx, ny=auv.ny, poly=poly_tag,
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                             unit_x=unit_x, unit_y=unit_y)

# Plot correlation heatmap:
title = 'Feature correlation'
fig_cr = gclas.plot_correl(df_smp, key_t, key_x=key_x, key_y=key_y, title=title)

#------------------------------------------------------------
#   Plot results 
#------------------------------------------------------------

# Plot cross-validation scores:
scores, title = ['accu', 'prec', 'recall','f1'], srun + clas_name
fig_sc = gclas.plot_scores(test, scores, title=title)

# PLot feature importance:
title = srun + 'CV importante features'
fig_fi = greg.plot_feat_importance(etc.feature_importances_, 
                                  feat_name=feat_list, title=title)

# Plot confusion:
kbest = test['kbest']
title = srun + 'Confusion matrix'
fig_cf = gclas.plot_confusion(test['pred'][kbest], test['targ'], 
                              title=title, key_t=key_t, unit_t=unit_t, 
                              label = numb_dict.keys())

plt.show()

# CRS colormap:    
'''crs_wrk = [[1,0,0,1], [1,0,0,1], [1,0,0,1], [1,1,0,1], 
           [1,1,0,1], [1,1,0,1], [0,1,0,1], [0,1,0,1]]'''

crs_wrk = [[1,1,0,1], [1,1,0,1], [1,0,0,1], [1,0,0,1],[0,1,0,1], [0,1,0,1]]
crs = ListedColormap(crs_wrk,len(crs_wrk))

# Deleta all labels except MIR
poly_no_label = poly_tag.copy()
for jj in range(len(poly_no_label)-1): poly_no_label[jj]['label'] = ''

numb_dict['nan'] = np.nan   
    
# Plot map predictions:
title=srun + 'SMS predictions map '
fig_pm = gclas.plot_pred_maps(df_pred_nan, key_t, numb_dict=numb_dict ,key_x=key_x, key_y=key_y,
                             nx = auv.nx, ny=auv.ny, title=title, 
                             poly=poly_no_label, ticks = numb_dict.keys(), 
                             pred_perc=['pred'],
                             vmin=0, vmax=2, unit_t=unit_t, cmap=crs,
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                             unit_x=unit_x, unit_y=unit_y, verbose=1)

plt.show(block=block)

#-------------------------------------------------------------------
#   Dump figs on png files
#-------------------------------------------------------------------

fig_ts.savefig(pdir + crun + '_TAG_AUV_Pseudo_Sample_Picks.png')
fig_fm.savefig(pdir + crun + zoom + '_TAG_AUV_Feature_Maps.png')
fig_cr.savefig(pdir + crun + '_TAG_AUV_Feature_Correlation.png')
fig_sc.savefig(pdir + crun + '_TAG_AUV_CV_Scores.png')
fig_fi.savefig(pdir + crun + '_TAG_AUV_Feature_Importance.png')
fig_cf.savefig(pdir + crun + '_TAG_AUV_Confusion.png')
fig_pm.savefig(pdir + crun + zoom + '_TAG_AUV_Prospectivity_Map_Prediction.png')

