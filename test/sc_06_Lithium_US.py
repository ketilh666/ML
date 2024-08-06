# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:58:39 2022

@author: kehok
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import geopandas as gpd
from matplotlib.colors import ListedColormap
import sklearn.ensemble as ens
import os

# KetilH stuff
import khio
from gravmag.common import gridder
from gravmag.common import MapData
import curvature as crv

import ml.greg as greg
import ml.gclas as gclas

#----------------------------------------
# Read data from G-drive
#----------------------------------------

block=True

exl = '../data_Li/'
pkl = '../data_Li/'
shp = '../data_Li/shp/'

png = 'png_Li/'
if not os.path.isdir(png): os.mkdir(png)

#-----------------------------
#   Read train and test data
#-----------------------------

fname_smp  = 'Train_Test_Samples_and_Features_US.xlsx'
fname_temp = 'Train_Test_Samples_and_Features_Temp_US.xlsx'

with pd.ExcelFile(exl + fname_smp) as fid:
    df_Li_in = pd.read_excel(fid)

# with pd.ExcelFile(exl + fname_temp) as fid:
#     df_temp_in = pd.read_excel(fid)

df_smp_in = df_Li_in

#--------------------------------------------
#  Read data from pickle file
#--------------------------------------------

with open(pkl + 'Grids_10km_US.pkl', 'rb') as fid:
    data = pickle.load(fid)

# US
lon1, lon2 = -125, -65
lat1, lat2 =  25, 50

lon, lat = data.x, data.y

# Grids corresponding to feat_list_all
ml_list = [jj for jj in range(len(data.grd))]

#-----------------------
#  ML setup
#-----------------------

# Column keys and units:
key_x, unit_x = list(df_smp_in.columns)[0], ' [deg]'
key_y, unit_y = list(df_smp_in.columns)[1], ' [deg]'
key_t, unit_t = list(df_smp_in.columns)[2], ' [mg/L]'

feat_list_red = data.label_short
feat_list_all = [feat_list_red[ii] for ii in ml_list]

#unit_list_red = data.unit
unit_list_red = len(feat_list_red)*['[-]']
unit_list_all = [unit_list_red[ii] for ii in ml_list]

# Which test to run:
krun, version = 5, ''   # version = empty or a, b, c etc
print(f'### Run {krun}{version}')
split, var = 'Random',  'MLE'
if   krun ==  0: # No good
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
elif krun ==  1: #  
     ind_feat_use = [3,4,5,7,8,10,11,13]
elif krun ==  2: #  
     ind_feat_use = [3,4,5,7,8,10,11]
elif krun ==  3: #  
     ind_feat_use = [3,4,5,7,8,11]
elif krun ==  4: #  
     ind_feat_use = [3,4,5,7,8]
elif krun ==  5: #  
     ind_feat_use = [3,4,5,7]

# Out put files
crun = 'run' + str(krun) + version
out_file_model = '_US_Li_classification_model.pkl'
out_file_pred  = '_US_Li_Predicted_Maps.pkl' 
out_file_excel = '_US_Li_Predicted_Maps.xlsx' 

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [unit_list_all[jj] for jj in ind_feat_use]

# ML pars:
test_size = 0.20
n_halluc  = 0
#scaling = 'Standardization' #Standardization, Normalization or None
scaling = None

#-------------------------------------------------
#   Maps for prediction
#-------------------------------------------------

# FInd all finite
jnd_list = []
for jj in ml_list:
    jnd_list.append(np.isfinite(data.grd[jj]))
    
ind = jnd_list[0]
for jnd in jnd_list[1:]:
    ind = ind & jnd

df_grd_in = pd.DataFrame(columns=[key_x,  key_y] + feat_list_all)

# AOI
ieu = (data.gx >= -20.0) &  (data.gx <= 20.0) & (data.gy >= 30.0) &  (data.gy <= 70.0)

df_grd_in[key_x] = data.gx.ravel()
df_grd_in[key_y] = data.gy.ravel()

for jj, idd in enumerate(ml_list):
    key = feat_list_all[jj]
    print(jj, key)
    df_grd_in[key] = data.grd[idd].ravel()

# Get the features to use in prediction    
df_grd = df_grd_in[[key_x, key_y] + feat_list].copy()

#------------------------------------------------------------
# Get the features of interest:
#------------------------------------------------------------

df_smp = df_smp_in[[key_x, key_y, key_t] + feat_list].copy()
df_grd = df_grd_in[[key_x, key_y] + feat_list].copy()

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

#------------------------------------
# Classifiers
#------------------------------------

# quant = df_smp[key_t].copy()
# df_smp.loc[quant<100, key_t] = 'L'
# df_smp.loc[(quant>=100) & (quant<200), key_t]  = 'M'
# df_smp.loc[quant>=200, key_t]  = 'H'

# numb_dict = { 'L': 0, 'M': 1, 'H': 2, 'nan': np.nan}

quant = df_smp[key_t].copy()
df_smp.loc[quant<100, key_t] = 'L'
df_smp.loc[quant>=100, key_t]  = 'H'

numb_dict = { 'H': 1, 'L': 2, 'nan': np.nan}

#------------------------------------------------------------
#   Train ML models for classification:
#     o etc : model for estimating the class (extra trees classifier)
#     o random or spatial train/test split
#------------------------------------------------------------

# Init classifier objects:
n_est = 500
etc  = ens.ExtraTreesClassifier(n_estimators=n_est)

# Cross validation train/test
kplot = False
etc, test = gclas.fit_cv(etc, df_smp, key_t, 
                         key_x=key_x, key_y=key_y, split=split, 
                         verbose=1, kplot=kplot, qc_cv=True)


# Fix the title of the cluster plot
try:  
    ax = test['fig_clu'].get_axes()[0]
    ax.set_title(f'run {krun}{version}: Train&Test ')
    test['fig_clu'].savefig(pdir + crun + '-kMeans_clusters.png')            
except: 
    pass

#------------------------------------------------------------
#   Plot results: Train&Test 
#------------------------------------------------------------

# Just for plotting labels:
clas_name, srun = str(etc.__str__).split()[3], crun + ': '

# Plot correlation heatmap:
title = 'Feature correlation'
fig_cr = gclas.plot_correl(df_smp, key_t, key_x=key_x, key_y=key_y, title=title)
fig_cr.savefig(png + crun + '_Feature_Correlation.png')

# Plot cross-validation scores:
scores, title = ['accu', 'prec', 'recall','f1'], srun + clas_name
fig_sc = gclas.plot_scores(test, scores, title=title)

# PLot feature importance:
title = srun + 'CV importante features'
fig_fi = greg.plot_feat_importance(etc.feature_importances_, 
                                  feat_name=feat_list, title=title)
fig_fi.savefig(png + crun + '_Feature_Importance.png')

# Plot confusion:
kbest = test['kbest']
title = srun + 'Confusion matrix'
fig_cf = gclas.plot_confusion(test['pred'][kbest], test['targ'], 
                              title=title, key_t=key_t, unit_t=unit_t, 
                              label = numb_dict.keys())

plt.show(block=False)

#--------------------------------
#  Classification on maps
#--------------------------------

df_pred = gclas.predict_ml(etc, df_grd, key_t,
                          key_x=key_x, key_y=key_y, verbose=1)

# Put the datafram back in a regular grid:
grd_pred = MapData(data.x, data.y, data.z, 
                   [np.empty_like(data.grd[0],dtype=object)])

grd_pred.grd[0].fill('nan')
grd_pred.grd[0] = np.array(df_pred['pred']).reshape(data.ny, data.nx)

# Replace categories by numbers
for cc in numb_dict.keys():
    ind = grd_pred.grd[0] == cc
    grd_pred.grd[0][ind] = numb_dict[cc]

# OK HIT 17/4-2023

#------------------------------------------------------------
#   Write results
#------------------------------------------------------------

# ML regression models for mean and variance:
with open(pkl + crun + out_file_model,'wb') as wrt:
    pickle.dump([etc, test], wrt)

# ML prediction:
with open(pkl + crun + out_file_pred,'wb') as wrt:
    pickle.dump([grd_pred], wrt)

#--------------------------------
#  Supervised classification
#  Maybe it will work
#--------------------------------

# Plot control
plot_list = [0,12,1,2,3,4,5,7,8,9,11,10]
vmin = [-10000, 0.0,     0, -30000, -80000, -300, -200, -150, -500,     0,   0, 0]
vmax = [  8000, 0.2, 24000,  10000, -10000,  300,  200,  150,  500, 90000, 350, 2]

# # Plot control
# plot_list = [0,12,1,2,3,4,5,7,8,9,11,13]
# vmin = [-10000, 0.0,     0, -30000, -80000, -300, -200, -150, -500,     0,   0, 0.000]
# vmax = [  8000, 0.2, 24000,  10000, -10000,  300,  200,  150,  500, 90000, 350, 0.005]
        
#-----------------------------------------
# Read the World Stress Map
#-----------------------------------------

# Read shapefile with contnents
fname_continent = shp + 'continent_ln/continent_ln.shp'
fname_borders = shp + 'World_Countries/World_Countries__Generalized_.shp'
fname_states = shp + 'United_States/tl_2017_us_state.shp'
conts = gpd.read_file(fname_continent)
borders = gpd.read_file(fname_borders)
states = gpd.read_file(fname_states)

# Make a colormap with 3 colors
bgr_wrk = [[0.5,0.5,1,1], [0.5,1,0.5,1], [1,0.5,0.5,1]] 
bgr = ListedColormap(bgr_wrk,len(bgr_wrk))
kols = list('bgrk')

# Tectonic regimes
reg_list = ['NF', 'SS', 'TF']

#------------------------
#  Make some plots
#------------------------
  
fig, axs = plt.subplots(4,3, figsize=(18,14))
xtnt = [lon[0], lon[-1], lat[0], lat[-1]]

for ii, jj in enumerate(plot_list):
    
    ax = axs.ravel()[ii]
    if jj == 14:
        im = ax.imshow(data.grd[jj], origin='lower', extent=xtnt, cmap=bgr,
                        vmin=vmin[ii], vmax=vmax[ii])
        cb = ax.figure.colorbar(im, ax=ax)
        cb.set_ticks([jj for jj in range(len(reg_list))], labels=reg_list)
        
    else:
        im = ax.imshow(data.grd[jj], origin='lower', extent=xtnt,
                        vmin=vmin[ii], vmax=vmax[ii])
        cb = ax.figure.colorbar(im, ax=ax)
        
    # conts.plot(ax=ax, color='k', linewidth=0.5)
    #borders.boundary.plot(ax=ax, color='k', linewidth=0.5)
    states.boundary.plot(ax=ax, color='k', linewidth=0.5)

    ax.set_xlabel('Lon [deg]')
    ax.set_ylabel('Lat [deg]')
    ax.set_title(data.label[jj])
    
    ax.set_xlim(lon1, lon2)
    ax.set_ylim(lat1, lat2)
    
fig.tight_layout(pad=1.0)
fig.savefig(png + 'All_Maps_US.png')    

#---------------------------------
# PLot supervised classification
#---------------------------------

# Make a colormap with 3 colors
gr_wrk = [[0.5,1,0.5,1], [1,0.5,0.5,1]] 
gr = ListedColormap(gr_wrk,len(gr_wrk))

lab = clas_name

gwrk = grd_pred.grd[0].astype(float)
fig, ax =plt.subplots(1, figsize=(16, 8))
im = ax.imshow(gwrk, origin='lower', extent=xtnt, cmap=gr)

states.boundary.plot(ax=ax, color='k', linewidth=0.5)

ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')
ax.set_title(f'Supervised {lab}')
fig.tight_layout(pad=1.0)
    
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)

fig.savefig(png+ f'Supervised_US_{crun}.png')

#-----------------------------
#  PLot train and test data
#-----------------------------

dfwrk = df_smp_in[df_smp_in['Li']<=300]
dfwrk = dfwrk.sort_values(by='Li', ascending=True)

fig, ax =plt.subplots(1, figsize=(18, 8))

states.boundary.plot(ax=ax, color='k', linewidth=0.5)
im = ax.scatter(dfwrk.lon, dfwrk.lat, c=dfwrk.Li)
ax.figure.colorbar(im, ax=ax)

ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')
ax.set_title(f'Train&test')
fig.tight_layout(pad=1.0)
    
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)

fig.savefig(png+ f'Train&test_US.png')

plt.show(block=block)
print(f'krun = {krun}')
