# -*- coding: utf-8 -*-
"""
Purpose: 
Compute resources in mega tonnes from ML predictions

Created on Fri Apr 21 09:23:37 2020
Programmed: Ketil Hokstad
"""

#---------------------------------------------------------
#  CCZ resource assessment
#  Regional evaluation
#  Before running this script, run sc_02_ML_Nodules 
#  4 times for krun=5,6,7,8
#---------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy   as np
import pandas  as pd
import json
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D  # for 3D graphics

# My new module:
import resource.nodules as noddy

##---------------------------------
# Folders
#---------------------------------

# Data directory
cdir = '../data_nodules/'

pdir = 'png_nodules/'
if not os.path.isdir(pdir): os.mkdir(pdir)

block = True

#--------------------------------------------------------------------------
# Read AOI grid definition
#--------------------------------------------------------------------------

# Read AOI:
aoi = json.load(open(cdir + 'CCZ_AOI_Dense.json')) 
lon1, lon2, nlon, dlon = [aoi[key] for key in ['lon1', 'lon2', 'nlon', 'dlon']]
lat1, lat2, nlat, dlat = [aoi[key] for key in ['lat1', 'lat2', 'nlat', 'dlat']]

# Read polygons and unpack
# polygons = pd.read_pickle(cdir + 'CCZ_polygons.pkl') # old file, not compatible with pandas 2x
polygons =  pickle.load(open(cdir + 'CCZ_polygons_pandas_2x.pkl','rb')) # 
poly_isa,  poly_spc, poly_eur, poly_aoi = polygons

# Use this:
poly, poly2 = poly_isa, poly_eur

#-------------------------------------------------------------------------
#   Read Random forest output
#-------------------------------------------------------------------------

# Input root file names
root_file_model = '_CCZ_ML_regression_model.pkl'
root_file_pred  = '_CCZ_Predicted_Maps.xlsx' 

# Output files:
file_score    = 'CCZ_All_Scores.xlsx'
root_file_resource = '_CCZ_Resource.xlsx'
root_file_tonnage  = '_CCZ_Tonnages.xlsx'

run_list = ['5a', '6a', '7a', '8a'] # All runs
nrun = len(run_list)
run_list_short = ['5a']             # Best run selected for bar plots 

print('### run_list = {}'.format(run_list))
print('### run_list_short = {}'.format(run_list_short))

# Read pickle files: Stats
keys = ['reg_mu', 'reg_sig','test_mu','test_sig']
test_list = []
for run in run_list:
    in_file = cdir + 'run' + str(run) + root_file_model
    with open(in_file,'rb') as fid: 
        test_list.append(dict(zip(keys, pickle.load(fid)))) 

# Read excel files: Maps
pred_list = []
for run in run_list_short:
    in_file = cdir + 'run' + str(run) + root_file_pred 
    with pd.ExcelFile(in_file) as fid:
        pred_list.append(pd.read_excel(fid)) 

# Column keys and units:
key_x, unit_x = 'lon', ' [deg]'
key_y, unit_y = 'lat', ' [deg]'
key_t, unit_t = 'abu', ' [kg/m2]'

# Percentiles in ML predictions
perc = ['P90','P50','P10']

### OK til hit 20/11-2020

#---------------------------------------------------------------------------
# Gather the stats from all test runs:
#---------------------------------------------------------------------------

figs_imp, feat_imp = noddy.plot_feat_stats(test_list, run_list )

figs_score, score  = noddy.plot_score_stats(test_list, run_list)

# write the scores to Excel file
with (pd.ExcelWriter(cdir + file_score)) as wrt:
    for key in score:
        pd.DataFrame.from_dict(score[key]).to_excel(wrt, sheet_name=key)
        
#--------------------------------------------------------------------------
#   Grid setup
#--------------------------------------------------------------------------

# Definition of grid with surfacee elements:
glon, glat, surf_elem = noddy.grid_setup(lon1, lon2, nlon, lat1, lat2, nlat)

# Compute a mask for inside/outside of polygons:
mask, mask_ar = noddy.polygon_mask(poly, glon=glon, glat=glat, kplot=False)

#--------------------------------------------------------------------------
#   Sum up resources by area
#--------------------------------------------------------------------------

resource_list = noddy.resource_sum(pred_list, surf_elem, mask_ar, poly, 
                                   perc=perc, verbose=0)

# Write to Excel
tonnage_keys_out = ['abu', 'wet_wgt', 'Mn_wgt','Ni_wgt', 'Cu_wgt', 'Co_wgt']
jnk = noddy.resource_to_excel(resource_list, run_list_short, perc=perc, 
                              tonnage_keys=tonnage_keys_out, cdir=cdir,
                              root_file_resource=root_file_resource,
                              root_file_tonnage=root_file_tonnage)

#--------------------------------------------------------------------------
# PLot results
#--------------------------------------------------------------------------

name_t, p50 = 'Abundance', ['P50']
figs_p50 = noddy.plot_maps_masked(pred_list, run_list_short, mask, 
                                  perc=p50, vmin=0, vmax=40,
                                  poly=poly, poly2=poly2, 
                                  key_t=name_t, unit_t=unit_t)

#---------------------------
#  Bar plots
#---------------------------

tre_d, cset = False, 1

# Label with license number only
poly_wrk = poly.copy()
poly_wrk[0]['label'] = ''

figs_bar_a = noddy.plot_bars(resource_list, run_list_short,
                             perc=perc, poly=poly_wrk, tre_d=tre_d, cset=cset,
                             key_t='abu', vmin=0, vmax=40,
                             name_t='abundance', unit_t=' [kg/m2]')

figs_bar_b = noddy.plot_bars(resource_list, run_list_short,
                             perc=perc, poly=poly_wrk, tre_d=tre_d, cset=cset,
                             key_t='wet_wgt', vmin=0, vmax=2000,
                             name_t='wet weight', unit_t=' [Mt]')

figs_bar_0 = noddy.plot_bars(resource_list, run_list_short,
                             perc=perc, poly=poly2, tre_d=tre_d, cset=cset,
                             key_t='abu', vmin=0, vmax=40,
                             name_t='abundance', unit_t=' [kg/m2]')

figs_bar_1 = noddy.plot_bars(resource_list, run_list_short,
                             perc=perc, poly=poly2, tre_d=tre_d, cset=cset,
                             key_t='wet_wgt', vmin=0, vmax=2000,
                             name_t='wet weight', unit_t=' [Mt]')

figs_bar_2 = noddy.plot_bars(resource_list, run_list_short,
                             perc=perc, poly=poly2, tre_d=tre_d, cset=cset,
                             key_t='Ni_wgt', vmin=0, vmax=30,
                             name_t='Ni weight', unit_t=' [Mt]')

figs_bar_3 = noddy.plot_bars(resource_list, run_list_short,
                             perc=perc, poly=poly2, tre_d=tre_d, cset=cset,
                             key_t='Co_wgt', vmin=0, vmax=5,
                             name_t='Co weight', unit_t=' [Mt]')

#-------------------------------------------------------------------
#   Dump figs on png files
#-------------------------------------------------------------------

figs_imp[0].savefig(pdir + 'Feature_Importance_mu.png')
figs_imp[1].savefig(pdir + 'Feature_Importance_sig.png')

figs_score[0].savefig(pdir + 'Score_mu.png')
figs_score[1].savefig(pdir + 'Score_sig.png')

for jj, run in enumerate(run_list_short):
    crun = 'run' + str(run)
    figs_p50[jj].savefig(pdir + crun + 'Abundance_Map.png')
    figs_bar_a[jj].savefig(pdir + crun + 'Abundance_Bar3d_ISA.png')
    figs_bar_b[jj].savefig(pdir + crun + 'WetWeight_Bar3d_ISA.png')
    figs_bar_0[jj].savefig(pdir + crun + 'Abundance_Bar3d_Eur.png')
    figs_bar_1[jj].savefig(pdir + crun + 'WetWeight_Bar3d_Eur.png')
    figs_bar_2[jj].savefig(pdir + crun + 'Ni_Weight_Bar3d_Eur.png')
    figs_bar_3[jj].savefig(pdir + crun + 'Co_Weight_Bar3d_Eur.png')

plt.show(block=block)
