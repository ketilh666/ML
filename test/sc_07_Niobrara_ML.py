# -*- coding: utf-8 -*-
"""
Purpose: 
Run regression ML on map data from TAG.  
Seabed massiv sulfide case

Created on Fri Mar 01 2021
Programmed: Ketil Hokstad
"""
#------------------------------------------------------
#  Imports
#------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
import sklearn.linear_model as lin
import json
import pickle
import matplotlib.cm as cm
import os

# KetilH stuff
import ml.greg as greg
from gravmag.meta import MapData
from gravmag.common import gridder

#----------------------------
# Folders
#----------------------------

# Data directory
cdir = '../data_Niobrara/'

pdir = 'png_Niobrara/'
if not os.path.isdir(pdir): os.mkdir(pdir)

block = True

#---------------------------------------------------
# Read data
#---------------------------------------------------   

# Labeled samples: Target and features for train and test 
in_file = cdir + 'Niobrara_dre_cleaned.xlsx'
sheet  = 'WellData'
df_samp_in = pd.read_excel(pd.ExcelFile(in_file),sheet)

# Features: map data for prediction
in_file = in_file = cdir + 'Niabrara_MapData_withIsopach.xlsx'
sheet  = 'SI units'
df_feat_in = pd.read_excel(pd.ExcelFile(in_file),sheet)

#---------------------------------------
# Convert feet to SI units
#---------------------------------------

f2m = 0.3048
df_samp_in['true_vert_depth'] = f2m*df_samp_in['true_vert_depth'] 
df_samp_in['tot_depth'] = f2m*df_samp_in['tot_depth'] 
df_samp_in['lat_length'] = f2m*df_samp_in['lat_length'] 
df_samp_in['dist_closest_well_2D'] = f2m*df_samp_in['dist_closest_well_2D'] 
df_samp_in['dist_closest_well_3D'] = f2m*df_samp_in['dist_closest_well_3D'] 

#------------------------------------------
# Target and feature keys in dataframe
#------------------------------------------

# Features and target keys and units
feat_list_all = ['isopach','topNio_surf','surf_elev',
                 'slope_angle','temp_gradient','topNio_temp']
feat_unit_all = [' [m]',' [m]',' [m]',
                 ' [deg]',' [degC/km]',' [degC]']

# Column keys and units:
key_x, unit_x = 'longitude', ' [deg]'
key_y, unit_y = 'latitude', ' [deg]'
key_t, unit_t = 'firstyear_prod', ' [bbl]'

#----------------------------------------
#   Set job parameters
#---------------------------------------

# ML pars:
test_size = 0.20
n_halluc  = 0

# Random Forest: Init regressor objects:
n_est, msl, md = 500, 1, 16
reg_mu  = ens.RandomForestRegressor(n_estimators=n_est, min_samples_leaf=msl, max_depth=md)
reg_sig = ens.RandomForestRegressor(n_estimators=n_est, min_samples_leaf=msl, max_depth=md)

# Which test to run:  
krun, version = 1, ''   # version = empty or a, b, c etc
print('### Run{}{}'.format(krun,version))
if   krun ==  0: # No good
    ind_feat_use = [ii for ii in range(len(feat_list_all))]
    split, var = 'Random',  'MLE'
    reg_mu  = lin.LinearRegression()
    reg_sig = lin.LinearRegression()
elif krun ==  1: #  
    ind_feat_use = [ii for ii in range(len(feat_list_all))]
    split, var = 'Random',  'MLE'
elif krun ==  2: #  
    ind_feat_use = [0, 1, 3, 4, 5]
    split, var = 'Random',  'MLE'
elif krun ==  3: #  
    ind_feat_use = [1, 3, 4, 5]
    split, var = 'Random',  'MLE'
elif krun ==  4: #  
    ind_feat_use = [ii for ii in range(len(feat_list_all))]
    split, var = 'Spatial',  'MLE'
elif krun == 11:
    ind_feat_use = [ii for ii in range(len(feat_list_all))]
    split, var = 'Random',  'MLE'
    n_halluc = 1000
    

# Output files
crun = 'run' + str(krun) + version
out_file_model = '_Niobrara_ML_regression_model.pkl'
out_file_pred  = '_Niobrara_Predicted_Maps.xlsx' 

#------------------------------------------------------------
#  Get features to be used in current run
#------------------------------------------------------------

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [feat_unit_all[jj] for jj in ind_feat_use]

# Get the features of interest:
df_samp = df_samp_in[[key_x, key_y, key_t] + feat_list].copy()
df_feat = df_feat_in[[key_x, key_y] + feat_list].copy()

#------------------------------------------------------------
#   Add some hallucinatory evidence for the failure case
#------------------------------------------------------------

if n_halluc > 0:
    
    val_hal = 0
    df_samp = greg.add_halluc(df_feat, df_samp, n_halluc, val_hal, key_t,
                              key_x='longitude', key_y='latitude',
                              verbose=1, kplot=True)
    
#------------------------------------------------------------
#   Feature engineering
#------------------------------------------------------------

#  Thrash some sample data?
targ_lo, targ_hi = 0, 1e12 
df_samp = df_samp[(df_samp[key_t] >= targ_lo) &  (df_samp[key_t] <= targ_hi)]

# Impute nans
df_feat.fillna(0, inplace=True)

### OK Hit

#------------------------------------------------------------
#   Train ML models for mean and varaince:
#     o reg_mu : model for estimating mean
#     o reg_sig: model for estimating variance
#     o random or spatial train/test split
#------------------------------------------------------------

# Cross validation train/test
reg_mu, reg_sig, test_mu, test_sig = greg.fit_cv(reg_mu, reg_sig, df_samp, 
                                     key_t, key_x=key_x, key_y=key_y, split=split, 
                                     verbose=1, kplot=True, qc_cv=True)

# Fix the title of the cluster plot
try:  
    ax = test_mu['fig_qc'].get_axes()[0]
    if split.lower()[0] == 's':
        ax.set_title('run{}{}: kMeans clustering'.format(krun, version))
    else:
        ax.set_title('run{}{}: Random split'.format(krun, version))        
    test_mu['fig_qc'].savefig(pdir + crun + 'test_train_split.png')            
except: 
    pass

#------------------------------------------------------------
#   Predict target from feature maps
#------------------------------------------------------------

df_pred = greg.predict_ml(reg_mu, reg_sig, df_feat, key_t, 
                          key_x=key_x, key_y=key_y, verbose=1)

#-----------------------------------------------------------------
#  Get feature and target maps on a rectangular grid for plotting
#-----------------------------------------------------------------

nx, ny = len(np.unique(df_feat[key_x])), len(np.unique(df_feat[key_y]))
x = np.linspace(np.min(df_feat[key_x]), np.max(df_feat[key_x]), nx)
y = np.linspace(np.min(df_feat[key_y]), np.max(df_feat[key_y]), ny)
gx, gy = np.meshgrid(x,y)

# Make new dataframes for the rectangular grids
df_feat_r  = pd.DataFrame(columns=df_feat.columns)
df_pred_r = pd.DataFrame(columns=df_pred.columns)
# Put coordinates of the rectangular grid into dataframes
df_feat_r[key_x], df_feat_r[key_y] = gx.flatten(), gy.flatten()
df_pred_r[key_x], df_pred_r[key_y] = gx.flatten(), gy.flatten()

# Interpolate feature maps
for ff in feat_list:
    print('Interpolate {} grid'.format(ff))
    gval = gridder(df_pred[key_x], df_pred[key_y], df_feat[ff], gx, gy)
    df_feat_r[ff] = gval.flatten()

# Interpolate P10/P50/P90 maps
prob_list = list(df_pred.columns)[2:4+1]
for pp in prob_list:
    print('Interpolate {} grid'.format(pp))
    gval = gridder(df_pred[key_x], df_pred[key_y], df_pred[pp], gx, gy)
    df_pred_r[pp] = gval.flatten()

#------------------------------------------------------------
#   Write results
#------------------------------------------------------------

# Write map predictions to Excel file:
with pd.ExcelWriter(cdir + crun + out_file_pred) as wrt:
    df_pred_r.to_excel(wrt, index=False)

# ML regression models for mean and variance:
with open(cdir + crun + out_file_model,'wb') as wrt:
    pickle.dump([reg_mu, reg_sig, test_mu, test_sig], wrt)

#------------------------------------------------------------
#   Plot inputs 
#------------------------------------------------------------

# Zoom on maps
xmin, xmax = df_pred[key_x].min(), df_pred[key_x].max()
ymin, ymax = df_pred[key_y].min(), df_pred[key_y].max()
vmin, vmax = 0, 1.6e5

# Just for plotting labels:
reg_name, srun = str(reg_mu.__str__).split()[3], crun + ': '

# Well samples
title = 'Well data: first-year prod [bbl]'
fig_ts = greg.plot_targ_samp(df_samp, key_t, key_x=key_x, key_y=key_y,
                             title=title, vmin=vmin, vmax=vmax,
                             unix_x=unit_x, unit_y=unit_y, unit_t=unit_t)

## Clip to avoid fucked up color scales
#df_feat_nan['bs'].clip(lower=50, upper=200, inplace=True)
# Plot feature maps:
title = 'Feature maps'
fig_fm = greg.plot_feat_maps(df_feat_r, key_x=key_x, key_y=key_y,
                             nx = nx, ny=ny,
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                             unit_x=unit_x, unit_y=unit_y)

# Plot correlation heatmap:
title = 'Feature correlation'
fig_cr = greg.plot_correl(df_samp, key_t, key_x=key_x, key_y=key_y, title=title)

#------------------------------------------------------------
#   Plot results 
#------------------------------------------------------------

# Plot cross-validation scores:
scores, title = ['r2', 'ev', 'mase'], srun + reg_name
fig_sc = greg.plot_scores(test_mu, test_sig, scores, title=title)

# PLot feature importance (Random Forest only)
try:
    title_mu, title_sig = srun + 'CV1 mean', srun + 'CV2 variance'
    fig_f1 = greg.plot_feat_importance(reg_mu.feature_importances_, 
                                      feat_name=feat_list, title=title_mu)
    fig_f2 = greg.plot_feat_importance(reg_sig.feature_importances_, 
                                      feat_name=feat_list, title=title_sig)
    fig_f1.savefig(pdir + crun + '_Niobrara_Feature_Importance_Mean.png')
    fig_f2.savefig(pdir + crun + '_Niobrara_Feature_Importance_Variance.png')
except:
    pass

# Plot confusion:
kbest = test_mu['kbest']
title = srun + 'Confusion'
fig_cf = greg.plot_confusion(test_mu['pred'][kbest], test_mu['targ'], 
                                    title=title, key_t=key_t, unit_t=unit_t)

# CRS colormap:    
crs_wrk = [[1,0,0,1], [1,0,0,1], [1,0,0,1], [1,0,0,1], 
           [1,1,0,1], [1,1,0,1], [0,1,0,1], [0,1,0,1]]
crs = ListedColormap(crs_wrk,len(crs_wrk))

# Plot map predictions:
title=srun + 'Prospectivity '

if version == 'b': cmap = crs
else: cmap = cm.jet

fig_pm = greg.plot_pred_maps(df_pred_r, key_t, key_x=key_x, key_y=key_y,
                             nx = nx, ny=ny, title=title, 
                             vmin=vmin, vmax=vmax, unit_t=unit_t, cmap=cmap,
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                             unit_x=unit_x, unit_y=unit_y, verbose=1)

#-------------------------------------------
# Analyse overfitting/underfitting
# TODO: Put this into a function
#-------------------------------------------

ev_cv_train =  test_mu['cv']['train_explained_variance']
ev_cv_test  = test_mu['cv']['test_explained_variance']
ev_ho_test = test_mu['ev']

r2_cv_train =  test_mu['cv']['train_r2']
r2_cv_test  = test_mu['cv']['test_r2']
r2_ho_test = test_mu['r2']

cv_fold = np.array(list(range(len(r2_cv_train))))

fig, axs = plt.subplots(1,2, figsize=(10,5))
fig.tight_layout(pad=4.0)
ax = axs[0]
ax.plot(cv_fold, r2_cv_train, 'ko-', label='CV train')
ax.plot(cv_fold, r2_cv_test , 'bo-', label='CV test')
ax.plot(cv_fold, r2_ho_test , 'ro-', label='Held-out test')
ax.set_ylim(0, 1)
ax.legend()
ax.set_xlabel('CV fold [-]')
ax.set_ylabel('R2 score [-]')
ax.set_title('R2')

ax = axs[1]
ax.plot(cv_fold, ev_cv_train, 'ko-', label='CV train')
ax.plot(cv_fold, ev_cv_test , 'bo-', label='CV test')
ax.plot(cv_fold, ev_ho_test , 'ro-', label='Held-out test')
ax.set_ylim(0, 1)
ax.legend()
ax.set_xlabel('CV fold [-]')
ax.set_ylabel('EV score [-]')
ax.set_title('Explained variance')

fig.savefig(pdir + crun + '_Niobrara_Over_Under_Fit.png')


# Overview map
try:
    fig_ow = plt.figure(figsize=(8,7))
    xtnt = [xmin,xmax, ymin,ymax]
    elev = np.array(df_feat_r['surf_elev']).reshape(ny,nx)
    im=plt.imshow(elev, origin='lower', extent=xtnt, cmap=cm.jet)
    plt.colorbar(im)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(key_x+unit_x)
    plt.ylabel(key_y+unit_y)
    plt.scatter(df_samp[key_x], df_samp[key_y], c='k', marker='.')
    plt.title('Elevation and sample locations')
    fig_ow.savefig(pdir + 'Niobrara_OverviewMap.png')
except:
    pass

plt.show(block=block)
#-------------------------------------------------------------------
#   Dump figs on png files
#-------------------------------------------------------------------

fig_ts.savefig(pdir + 'Niobrara_Pseudo_Sample_Picks.png')
fig_fm.savefig(pdir + crun + '_Niobrara_Feature_Maps.png')
fig_cr.savefig(pdir + crun + '_Niobrara_Feature_Correlation.png')
fig_sc.savefig(pdir + crun + '_Niobrara_CV_Scores.png')
fig_cf.savefig(pdir + crun + '_Niobrara_Confusion.png')
fig_pm.savefig(pdir + crun + '_Niobrara_Prospectivity_Map_Prediction.png')
