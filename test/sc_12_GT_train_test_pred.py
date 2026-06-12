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

# KetilH stuff
import ml.greg as greg

#---------------------------------------------------
# Read input data
#---------------------------------------------------

# Data directory
cdir = '../data_GT/'
pdir = 'png_ML/'

noise = 1
in_file = 'GT_Train_and_Test_Data_noise' + str(noise) + '.xlsx'
df_smp_in = pd.read_excel(pd.ExcelFile(cdir + in_file))

# Features on grid for prediction:
in_file = cdir + 'Synt_Test_Log_0.xlsx'
df_grd_in = pd.read_excel(pd.ExcelFile(in_file))

# Make som fake x,y coordinates (for plotting only)
df_smp_in[['x', 'y']] = df_smp_in[['tvd', 'T']]
df_grd_in[['x', 'y']] = df_grd_in[['tvd', 'T']]

#----------------------------------------
#   Set job parameters
#---------------------------------------

# Features and target keys and units
feat_list_all = ['tvd', 'phi', 'log_rh', 'mag', 'dens', 'vp', 'vs', 'ps_rat']
feat_unit_all = ['[m]', '[-]', '[ohmm]', '[A/m]', '[kg/m3]', '[m/s]', '[m/s]', '[-]']

# Which test to run:
krun, version = 1, ''   # version = empty or a, b, c etc
print('### Run{}{}'.format(krun,version))
if   krun ==  0: # No good, dont run this
     ind_feat_use = [ii for ii in range(len(feat_list_all))]
     split, var = 'Random',  'MLE'
elif krun ==  1: #  
     ind_feat_use = [0,1,2,4,5,7]
     split, var = 'Random',  'MLE'
elif krun ==  3: #  
     ind_feat_use = [0,1,2,4]
     split, var = 'Random',  'MLE'

# Output files
crun = 'run' + str(krun) + version
out_file_model = '_GT_1st_Test_ML_Model.pkl'
out_file_pred  = '_GT_1st_Test_ML_Prediction.xlsx' 

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [feat_unit_all[jj] for jj in ind_feat_use]

# ML pars:
test_size = 0.20
n_halluc  = 0

# Column keys and units:
key_x, unit_x = 'x', ' [dummy]'
key_y, unit_y = 'y', ' [dummy]'
key_t, unit_t = 'T', ' [K]'

#------------------------------------------------------------
#  Thrash some sample data?
#------------------------------------------------------------

# Get the features of interest:
df_smp = df_smp_in[[key_x, key_y, key_t] + feat_list]
df_grd = df_grd_in[[key_x, key_y] + feat_list]
    
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

# Initialize regressor objects:
n_est = 500
reg_mu  = ens.RandomForestRegressor(n_estimators=n_est)
reg_sig = ens.RandomForestRegressor(n_estimators=n_est)

# Cross validation train&test
reg_mu, reg_sig, test_mu, test_sig = greg.fit_cv(reg_mu, reg_sig, 
                                     df_smp, key_t, split=split, 
                                     key_x='x', key_y='y',
                                     verbose=1, kplot=True, qc_cv=True)

# Fix the title of the cluster plot
try:  
    ax = test_mu['fig_clu'].get_axes()[0]
    ax.set_title('run{}{}: train&test data'.format(krun, version))
    ax.set_xlabel('depth [m]')
    ax.set_ylabel('T [K]')    
except: 
    pass

#------------------------------------------------------------
#   Prediction
#------------------------------------------------------------

df_pred = greg.predict_ml(reg_mu, reg_sig, df_grd, key_t, 
                          key_x=key_x, key_y=key_y, verbose=1)

#------------------------------------------------------------
#   Write results
#------------------------------------------------------------

# Prediciton of nodule abundance
# with pd.ExcelWriter(pd.ExcelWriter(cdir + crun + out_file_pred)) as wrt:
#     df_pred.to_excel(wrt, index=False)

# ML regression models for mean and variance:
# with open(cdir + crun + out_file_model,'wb') as writer:
#     pickle.dump([reg_mu, reg_sig, test_mu, test_sig], writer)

#------------------------------------------------------------
#   Plot inputs 
#------------------------------------------------------------

# Just for plotting labels:
reg_name, srun = str(reg_mu.__str__).split()[3], crun + ': '

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

# PLot well prediction
fig_T = plt.figure(figsize=(4,7))
zzz = 1e-3*df_grd_in['tvd']
TAZ = 273.0
mu_T, sig_T = df_pred['mu'] - TAZ, df_pred['sig']

plt.plot(df_grd_in['T']-TAZ, zzz,'k-', label='True')
plt.plot(mu_T, zzz,'r-', label='ML prediction')  
plt.plot(mu_T + sig_T, zzz,'r:')  
plt.plot(mu_T - sig_T, zzz,'r:')  
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel('T [oC]')
plt.ylabel('depth [km]')
plt.title('{} Temperature ML pred'.format(srun))

plt.show()

#-------------------------------------------------------------------
#   Dump figs on png files
#-------------------------------------------------------------------

test_mu['fig_clu'].savefig(pdir + crun + '_GT_1st_Test_Train_Test_Split.png')        

fig_cr.savefig(pdir + crun + '_GT_1st_Test_Feature_Correlation.png')
fig_sc.savefig(pdir + crun + '_GT_1st_Test_CV_Scores.png')
fig_f1.savefig(pdir + crun + '_GT_1st_Test_Feature_Importance_Mean.png')
fig_f2.savefig(pdir + crun + '_GT_1st_Test_Feature_Importance_Variance.png')
fig_cf.savefig(pdir + crun + '_GT_1st_Test_Confusion.png')
fig_T.savefig(pdir + crun + '_GT_1st_T_Prediction.png')
