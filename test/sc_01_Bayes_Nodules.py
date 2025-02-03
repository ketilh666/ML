# -*- coding: utf-8 -*-
"""
Purpose: 
Run Naive Bayes ML with histogram ditricutions 
on map data from CCZ. Dense grid (6min x 6min).

Testscript, to check that the code works

Created on Tue Oct 20 2020
Programmed: Ketil Hokstad
"""
#------------------------------------------------------
#  Imports
#------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import os

# KetilH stuff
import ml.bayes  as bml
import ml.screen as sml
import ml.greg as greg

# Move to function later:
#from sklearn.model_selection import train_test_split, cross_validate
#from matplotlib.colors import ListedColormap
#import matplotlib.cm as cm

#---------------------------------
# Folders
#---------------------------------


# Data directory
cdir = '../data_nodules/'

pdir = 'png_nodules/'
if not os.path.isdir(pdir): os.mkdir(pdir)

block = True

#---------------------------------------------------
# Read input data
#---------------------------------------------------

# Read AOI:
aoi = json.load(open(cdir + 'CCZ_AOI_Dense.json')) 

# Read polygons and unpack
# polygons = pd.read_pickle(cdir + 'CCZ_polygons.pkl') # old file, not compatible with pandas 2x
polygons =  pickle.load(open(cdir + 'CCZ_polygons_pandas_2x.pkl','rb')) # 
poly_isa,  poly_spc, poly_eur, poly_aoi = polygons

# Samples: features and target
in_file, sheet = cdir + 'CCZ_Picks_Abundance_withFeatures.xlsx', 'Features'
df_smp_in = pd.read_excel(pd.ExcelFile(in_file), sheet)

# Features on grid for prediction:
in_file, sheet = cdir + 'CCZ_Grids_Features.xlsx', 'Grids'
df_grd_in = pd.read_excel(pd.ExcelFile(in_file), sheet)

#----------------------------------------
#   Set job parameters
#---------------------------------------

# Features and target keys and units
feat_list_all = ['bat', 'slope', 'cph', 'vcl', 'vca', 'ccd', 'dist',
                 'grav', 'gzz', 'giso', 'mag']
feat_unit_all = [' [m]',' [m/km]',' [mg/m3]',' [-]',' [-]',' [m]',' [km]',
                 ' [mGal]',' [Eo]',' [mGal]',' [nT]']

# Which test to run:
krun, version = 666, ''   # version = empty or a, b, c etc
if   krun ==  0: # No good
    pass
elif krun == 666: #  Test ScreeningBayes method
     ind_feat_use = [2,5,6,7]
     split = 'Random'
     estimator = 'Mean'
elif krun == 667: #  Test ScreeningBayes method
     ind_feat_use = [2,5,6,7]
     split = 'Spatial'     
     estimator = 'Mean'
elif krun == 668: #  Test ScreeningBayes method
     ind_feat_use = [2,5,6,7]
     split = 'Random'     
     estimator = 'MAP'

n_estimator = 100

# Some pars for HistBayesRegresor:
bal = 0.0 # bal=0.5 works OK, bal =1 is too much
boost, wgt = 1, [1.0, 1.0, 1.0]
resfac = 4.0
kplot, krl, kkk = True, True, 200

# Out put files
crun = 'run' + str(krun) + version
out_file_model = '_CCZ_NaiveBayes_regression_model.pkl'
out_file_pred  = '_CCZ_NaiveBayes_Predicted_Maps.xlsx' 

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [feat_unit_all[jj] for jj in ind_feat_use]

# ML pars:
abu_scale = 1.0
targ_lo, targ_hi = -5, 40 # To capture all samples and get regular binning
test_size = 0.20
n_halluc  = 0

# Column keys and units:
key_x, unit_x = 'lon', ' [deg]'
key_y, unit_y = 'lat', ' [deg]'
key_t, unit_t = 'abu', ' [kg/m2]'

#------------------------------------------------------------
#  Thrash some sample data?
#------------------------------------------------------------

ind = list(df_smp_in[df_smp_in[key_t] < 1].index)
df_smp_in.loc[ind,key_t] = -2.5 # To get binning correct

df_smp = df_smp_in[(df_smp_in[key_t] >= targ_lo)] # &  (df_smp_in[key_t] <= targ_hi)]

# Feature maps OK
df_grd = df_grd_in

#------------------------------------------------------------
#   Add some hallucinatory evidence for the failure case
#------------------------------------------------------------

if n_halluc > 0:

    df_smp = greg.add_halluc(df_grd, df_smp, n_halluc, targ_lo, key_t, 
                              verbose=1, kplot=True)
    
#------------------------------------------------------------
#   Feature engineering
#------------------------------------------------------------

### TODO

#------------------------------------------------------------
#   Test screening Bayes (the old simple method)
#------------------------------------------------------------

# Initiate object:
key_p = 'pos'
reg_kh = sml.ScreeningBayesRegressor(n_estimator=100, verbose=1)

# Get the "success data" 
fiasco = 5  
train_kh = df_smp[df_smp[key_t]> fiasco][feat_list]

# Train the model
reg_kh, test_kh = sml.fit_sb(reg_kh, train_kh, key_p=key_p, 
                    feat_unit=feat_unit, kplot=True, verbose=1)

# Predict
col_list = [key_x, key_y] + feat_list
maps_kh = sml.predict_sb(reg_kh, df_grd[col_list], key_p=key_p, verbose=1)

#------------------------------------------------------------
#   Test histogram naive Bayes
#------------------------------------------------------------

# Number of target and feature bins for histograms
nbin_t, nbin_f = np.unique(df_smp[key_t]).shape[0]-1, 10

# Histogram and bagging parameters set in __init__
bal = 0.0    # bal=0.5 works OK, bal =1 is too much
boost, wgt = 1, [1.0, 1.0, 1.0]
resfac = 4.0

# FOr QC plotting:
kplot, krl, kkk = True, True, 200

# Initiate object:
reg_nb = bml.HistBayesRegressor(nbin_t=nbin_t, nbin_f=nbin_f, 
                                n_estimator=n_estimator, estimator=estimator, 
                                resfac=resfac, boost=boost, boost_wgt=wgt,
                                verbose=1)

# Get the target and feature data 
train_nb = df_smp[[key_x, key_y, key_t] + feat_list]

# Train the model
reg_nb, test_nb = bml.fit_nb(reg_nb, train_nb, key_t, split='Random',
                             feat_unit=feat_unit,  test_size=test_size,
                             kplot=True, kkk=200, krl=True, verbose=1)

# Predict
col_list = [key_x, key_y] + feat_list
maps_nb = bml.predict_nb(reg_nb, df_grd[col_list], key_t=key_t, verbose=1)

df_pred = maps_nb

#------------------------------------------------------------
#   Write results
#------------------------------------------------------------

# ML regression models for mean and variance:
with pd.ExcelWriter(pd.ExcelWriter(cdir + crun + out_file_pred)) as writer:
    df_pred.to_excel(writer)

# Map prediciton of nodule abundance
with open(cdir + crun + out_file_model,'wb') as writer:
    pickle.dump([reg_nb, test_nb], writer)

#------------------------------------------------------------
#   Plot inputs
#------------------------------------------------------------

# Just for plotting labels:
reg_name, srun = 'HistBayesRegressor', crun + ': '

# Grid dimensions:
lon1, lon2, lat1, lat2 = [aoi[key] for key in ['lon1', 'lon2', 'lat1', 'lat2']]
nlon, nlat, dlon, dlat = [aoi[key] for key in ['nlon', 'nlat', 'dlat', 'dlat']]

#------------------------------------------------------------
#  PLot Screening Bayes results
#------------------------------------------------------------

# Plot map predictions (move to plot section at the end)
title=srun + 'Posterior probability '
fig = sml.plot_post_map(maps_kh, key_p, nx=nlon, ny=nlat,
                        poly=poly_isa, title=title,
                        xmin=lon1, xmax=lon2, ymin=lat1, ymax=lat2,
                        unit_x=unit_x, unit_y=unit_y, verbose=1)

fig.savefig(pdir + crun + '_CCZ_Posterior_Probability.png')

#------------------------------------------------------------
#   Plot Naive Bayes results 
#------------------------------------------------------------

# Plot cross-validation scores:
scores, title = ['r2', 'rmse'], srun + reg_name
fig = greg.plot_scores(test_nb, test_nb, scores, title=title)
fig.savefig(pdir + crun + '_CCZ_NB_Scores.png')

# Plot confusion:
title = srun + 'Confusion'
fig = greg.plot_confusion(test_nb['mu'], test_nb['targ'], 
                          title=title, key_t=key_t, unit_t=unit_t)
fig.savefig(pdir + crun + '_CCZ_Confusion.png')

# Plot map predictions:
title=srun + 'Predicted nodule abundance '
fig = greg.plot_pred_maps(df_pred, key_t, nx=nlon, ny=nlat,
                          poly=poly_spc, title=title,
                          vmin=0, vmax=40, unit_t=unit_t,
                          xmin=lon1, xmax=lon2, ymin=lat1, ymax=lat2,
                          unit_x=unit_x, unit_y=unit_y, verbose=1)
fig.savefig(pdir + crun + '_CCZ_Abundance_Map_Prediction.png')

plt.show(block=block)

