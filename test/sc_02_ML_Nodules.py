# -*- coding: utf-8 -*-
"""
Purpose: 
Run regression ML on map data from CCZ. 
Dense grid (6min x 6min).
Testscript, the check that the code works

Created on Fri Apr 17 09:23:37 2020
Programmed: Ketil Hokstad
"""
#------------------------------------------------------
#  Imports
#------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
import json
import pickle
import os

# KetilH stuff
import ml.greg as greg

##---------------------------------
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
lon1, lon2, nlon, dlon = [aoi[key] for key in ['lon1', 'lon2', 'nlon', 'dlon']]
lat1, lat2, nlat, dlat = [aoi[key] for key in ['lat1', 'lat2', 'nlat', 'dlat']]

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
krun, version = 8, 'a'   # version = empty or a, b, c etc
print('### Run{}{}'.format(krun,version))
if   krun ==  0: # No good
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
     split, var = 'Random',  'RMSE'
elif krun ==  5: # Best
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
     split, var = 'Random',  'MLE'
elif krun ==  6: #  
     ind_feat_use = [0,2,5,6,7,9,10]
     split, var = 'Random',  'MLE'
elif krun ==  7: #  
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
     split, var = 'Spatial', 'MLE'
elif krun ==  8: #  
     ind_feat_use = [0,2,5,6,7,9,10]
     split, var = 'Spatial', 'MLE'

# Out put files
crun = 'run' + str(krun) + version
out_file_model = '_CCZ_ML_regression_model.pkl'
out_file_pred  = '_CCZ_Predicted_Maps.xlsx' 

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [feat_unit_all[jj] for jj in ind_feat_use]

# ML pars:
abu_scale = 1.0
targ_lo, targ_hi = 0, 40 # To capture all samples and get regular binning
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
df_smp_in.loc[ind,key_t] = -2.5     # To get binning correct
df_smp_in[key_t] = abu_scale*df_smp_in[key_t]
df_smp_in = df_smp_in[(df_smp_in[key_t] >= targ_lo) &  (df_smp_in[key_t] <= targ_hi)]

# Get the features of interest:
df_smp = df_smp_in[[key_x, key_y, key_t] + feat_list]
df_grd = df_grd_in[[key_x, key_y] + feat_list]

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
#   Train ML models for mean and varaince:
#     o reg_mu : model for estimating mean
#     o reg_sig: model for estimating variance
#     o random or spatial train/test split
#------------------------------------------------------------

# Init regressor objects:
n_est = 500
reg_mu  = ens.RandomForestRegressor(n_estimators=n_est)
reg_sig = ens.RandomForestRegressor(n_estimators=n_est)

# Cross validation train/test
reg_mu, reg_sig, test_mu, test_sig = greg.fit_cv(reg_mu, reg_sig, 
                                     df_smp, key_t, split=split,
                                     verbose=1, kplot=True, qc_cv=True)

# Fix the title of the cluster plot
try:  
    ax = test_mu['fig_clu'].get_axes()[0]
    ax.set_title('run{}{}: kMeans clustering'.format(krun, version))            
except: 
    pass

#------------------------------------------------------------
#   Predict target from feature maps
#------------------------------------------------------------

df_pred = greg.predict_ml(reg_mu, reg_sig, df_grd, key_t, 
                          key_x=key_x, key_y=key_y, verbose=1)

#------------------------------------------------------------
#   Write results
#------------------------------------------------------------

# Map prediciton of nodule abundance
with pd.ExcelWriter(pd.ExcelWriter(cdir + crun + out_file_pred)) as wrt:
    df_pred.to_excel(wrt, index=False)

# ML regression models for mean and variance:
with open(cdir + crun + out_file_model,'wb') as writer:
    pickle.dump([reg_mu, reg_sig, test_mu, test_sig], writer)

#------------------------------------------------------------
#   Plot inputs 
#------------------------------------------------------------

# Just for plotting labels:
reg_name, srun = str(reg_mu.__str__).split()[3], crun + ': '

# Box core samples
title = 'Box-core nodule abundance [kg/m2]'
fig_ts = greg.plot_targ_samp(df_smp, key_t, key_x=key_x, key_y=key_y,
                             vmin=0, vmax=40, poly=poly_isa, title=title,
                             xmin=lon1, xmax=lon2, ymin=lat1, ymax=lat2,
                             unix_x=unit_x, unit_y=unit_y, unit_t=unit_t)

# Plot feature maps:
title = 'Feature maps'
fig_fm = greg.plot_feat_maps(df_grd, key_x=key_x, key_y=key_y,
                             nx = nlon, ny=nlat, poly=poly_isa,
                             xmin=lon1, xmax=lon2, ymin=lat1, ymax=lat2,
                             unit_x=unit_x, unit_y=unit_y)

# Plot correlation heatmap:
title = 'Feature correlation'
fig_cr = greg.plot_correl(df_smp, key_t, key_x=key_x, key_y=key_y, title=title)

#------------------------------------------------------------
#   Plot results 
#------------------------------------------------------------

# Plot cross-validation scores:
scores, title = ['r2', 'ev', 'mase'], srun + reg_name
fig_sc = greg.plot_scores(test_mu, test_sig, scores, title=title)

# PLot feature importance:
title_mu, title_sig = srun + 'CV1 mean', srun + 'CV2 variance'
fig_f1 = greg.plot_feat_importance(reg_mu.feature_importances_, 
                                  feat_name=feat_list, title=title_mu)
fig_f2 = greg.plot_feat_importance(reg_sig.feature_importances_, 
                                  feat_name=feat_list, title=title_sig)

# Plot confusion:
kbest = test_mu['kbest']
title = srun + 'Confusion'
fig_cf = greg.plot_confusion(test_mu['pred'][kbest], test_mu['targ'], 
                                    title=title, key_t=key_t, unit_t=unit_t)

# Plot map predictions:
title=srun + 'Predicted nodule abundance '
fig_pm = greg.plot_pred_maps(df_pred, key_t, nx=nlon, ny=nlat,
                            poly=poly_isa, title=title,
                            vmin=0, vmax=40, unit_t=unit_t,
                            xmin=lon1, xmax=lon2, ymin=lat1, ymax=lat2,
                            unit_x=unit_x, unit_y=unit_y, verbose=1)

plt.show(block=block)

#-------------------------------------------------------------------
#   Dump figs on png files
#-------------------------------------------------------------------

fig_ts.savefig(pdir + crun + '_CCZ_Box_Core_Abundance_Picks.png')
fig_fm.savefig(pdir + crun + '_CCZ_Feature_Maps.png')
fig_cr.savefig(pdir + crun + '_CCZ_Feature_Correlation.png')
fig_sc.savefig(pdir + crun + '_CCZ_CV_Scores.png')
fig_f1.savefig(pdir + crun + '_CCZ_Feature_Importance_Mean.png')
fig_f2.savefig(pdir + crun + '_CCZ_Feature_Importance_Variance.png')
fig_cf.savefig(pdir + crun + '_CCZ_Confusion.png')
fig_pm.savefig(pdir + crun + '_CCZ_Abundance_Map_Prediction.png')
