# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:02:22 2022

@author: kehok
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:04:10 2021

@author: kehok
"""

#---------------------------------------------------------------
#   Synt test based on IDDP-2 temperature profile
#   Rock physics modeling using IDDP2 MGI_GT result
#---------------------------------------------------------------

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import os

# Ketil's stuff
import rockphys.geochem as gch
import rockphys.fugacity as fug
import rockphys.gt as gt

#---------------------------------------------------------------
#   Read the IDDP2 temps
#---------------------------------------------------------------

cdir = '../data_GT/'
pdir = 'png_ML/'
if not os.path.isdir(pdir): os.mkdir(pdir)

block = False

#--------------------------------------
#  Setup: parameters
#---------------------------------------

noise = 1 # Percentage of noise to add

# Parameters
TAZ = gt.TAZ           # Absolute zero
gg = gt.gg             # Acceleration of gravity
salinity = 0.03        # salinity for resistivity computation

mu0 = gt.mu0           # Magnetic vacuum perm
H0 = 51879*1e-9/mu0    # Magnetic BG field

vcl0, cec0 = 0.05, 20   # Clay fraction and cation exchange capacity

#-------------------------------------
#  Setup: arrays
#-------------------------------------

# Need a depth array for pressure, stress and f_O2 computation
nz = 101
tvd = np.linspace(0,10000,nz)

# Porosity range (old design value: gamma=0.46e-3)
phi0, ngamma = 0.10, 6
ln_gamma = np.array([-8.25 + 0.25*jj for jj in range(ngamma)])
gamma_arr = np.exp(ln_gamma)

# Temperature range
T0, T8, nbeta = 0, 900, 11 # Temps in oC
ln_beta = np.array([-9.75 + jj*0.20 for jj in range(nbeta)]) 
beta_arr = np.exp(ln_beta)

# PLot Temperature and porosity
fig, axs = plt.subplots(1,2, figsize=(10,5))
ax = axs[0]
for beta in beta_arr:
    T = T8 + (T0-T8)*np.exp(-beta*tvd) + TAZ
    labtxt = 'beta = {:4.2f}/km'.format(1e3*beta)
    ax.plot(T-TAZ, 1e-3*tvd, '-', label=labtxt)

ax.invert_yaxis()
ax.set_xlabel('Temperature [oC]')
ax.set_ylabel('Depth [km]')
ax.legend()

ax = axs[1]
for gamma in gamma_arr:
    phi = phi0*np.exp(-gamma*tvd)
    labtxt = 'gamma = {:4.2f}/km'.format(1e3*gamma)
    ax.plot(phi, 1e-3*tvd, '-', label=labtxt)

ax.invert_yaxis()
ax.set_xlabel('Porosity [-]')
ax.set_ylabel('Depth [km]')
ax.legend()

fig.suptitle('Train&Test parameters')
fig.savefig('png_ml/Train_Test_Temp_and_Por.png')

#--------------------------------------------------------------------
#  Create train&test data
#--------------------------------------------------------------------

# Data frame to collect all train%test data
idd_list  = ['tvd']
targ_list = ['T', 'phi']
feat_list = ['log_rh', 'mag', 'dens', 'vp', 'vs', 'ps_rat']
aux_list  = ['dens_w', 'phyd', 'seff', 'f_O2', 'x_oxi',
             'Tc1', 'Tc2', 'Tc3', 'wt1', 'wt2', 'wt3','cec']
columns=idd_list + targ_list + feat_list + aux_list
df_list = []

kplot = False
for jj, gamma in enumerate(gamma_arr):
    phi = phi0*np.exp(-gamma*tvd)
    
    for ii, beta in enumerate(beta_arr):
        T = T8 + (T0-T8)*np.exp(-beta*tvd) + TAZ
            
        # Resistivity:
        vcl = vcl0*np.ones_like(T)
        cec = gt.cat_ex_cap(T, cec0=cec0, ltap=10)
        log_rh, qc = gt.resistivity(T, phi, vcl, cec, salinity) 
    
        # Density
        dens = gt.density(T, phi, vcl, rho_w=1030)
    
        # P- and S-wave velocity (and density)
        dz = tvd[1] - tvd[0]
        dens_w = gt.water_density(T, 1030)
        slit = 1e-6*np.cumsum(gg*dens*dz)        # Lithostatic stress MPa
        phyd = 1e-6*np.cumsum(gg*dens_w*dz)      # Hydrostatic stress
        seff = slit - phyd
        vp, vs, rho = gt.velocity(T, phi, vcl, seff, kplot=kplot)
    
        # Oxygen fugacity
        P0, x0_O2 = 0.1013, 0.21
        log_f_O2 = fug.fugacity_hydro(T, phyd, P0=P0, x0_O2=x0_O2, kplot=kplot)
        f_O2 = np.exp(log_f_O2)
    
        # oxydation
        u0_tm = 0.6
        x0, z8 = 1.0, 0.8 # z8 = 1-x is the end of hi-temp oxy
        E0, mu, A0  = 100.0, 135.0, 1e8
        f_O2 = phi*phyd # Assumed oxygen fugacity
        A = A0*f_O2**(2/3) 
        t = np.linspace(0,5,5001) # time in ka
        x_oxi = np.zeros_like(T)
        for jj in range(len(T)):
            x_wrk, x_wrk2 = gch.mixed_kinetic_isot(t, T[jj], A[jj], E0, mu, x0=x0, z8=z8, R=gch.Rsi)
            x_oxi[jj] = x_wrk[-1]
    
        # Spinodal decomp and Ising weights:
        Tc, wt, qc = gch.spinodal_tm(T, u0_tm, x_oxi, x0=x0, kplot=False)
    
        # Magnetization
        M0 = [40., 20., 10.0]                  # Magnetization at absolute zero
        cs = [1/(2*H0), 1/(2*H0), 1/(2*H0)]    # Induced magnetization parameter  
        mag, rem, chi = gt.magnet(T, H0, Tc, M0, cs, wt, kplot=kplot)
        
        # Add some noise:
        if noise > 0:
            rn0 = 1e-2*noise
            log_rh = (1.0 + rn0*(2*rnd.rand(nz)-1))*log_rh
            dens   = (1.0 + rn0*(2*rnd.rand(nz)-1))*dens
            vp     = (1.0 + rn0*(2*rnd.rand(nz)-1))*vp
            vs     = (1.0 + rn0*(2*rnd.rand(nz)-1))*vs
            mag    = (1.0 + rn0*(2*rnd.rand(nz)-1))*mag
            
        # Build dataframe and append
        #idd_list  = ['tvd']
        #targ_list = ['T', 'phi']
        #feat_list = ['log_rh', 'mag', 'dens', 'vp', 'vs']
        #aux_list  = ['dens_w', 'phyd', 'seff', 'f_O2', 'x_oxi','Tc1', 'Tc2', 'Tc3']

        dwrk = np.array([tvd, T, phi, log_rh, mag, dens, vp, vs, vp/vs,
                        dens_w, phyd, seff, f_O2, x_oxi, 
                        Tc[0], Tc[1], Tc[2], wt[0], wt[1], wt[2], cec])
        df_list.append(pd.DataFrame(columns=columns, data=dwrk.T))
        
# Append all dataframes into one big frame
df_all = pd.concat(df_list)

# Save to Excel file
fname = cdir + 'GT_Train_and_Test_Data_noise' + str(noise) + '.xlsx'
with pd.ExcelWriter(fname) as fid:
    df_all.to_excel(fid, index=False)

#-----------------------------------------
#   Plot train&test data
#-----------------------------------------  
    
unit_list = ['[km]', '[K]', '[-]',
             '[ohmm]', '[A/m]', '[kg/m3]', '[m/s]', '[m/s]', '[-]',
             '[kg/m3]', '[MPa]', '[MPa]', '[MPa]', '[-]', 
             '[K]', '[K]', '[K]', '[-]', '[-]', '[-]', '[meq/100g]']
            
# Open a figure for plotting
fig, axs = plt.subplots(3,6, figsize=(18,10))

for df in df_list:    
    zzz = 1e-3*df['tvd'] # km
    for ia, col in enumerate(columns[1:18+1]):
        ax = axs.ravel()[ia]
        ax.plot(df[col], zzz,'-')
        
# Labels
for ia, col in enumerate(columns[1:18+1]):
    ax = axs.ravel()[ia]
    ax.set_ylabel('depth [km]')
    ax.set_xlabel('{} {}'.format(col, unit_list[ia+1]))
    #ax.set_title(col)       
    ax.invert_yaxis()

fig.tight_layout(pad=2.0)
fig.savefig(pdir + 'Train_Test_Data_All_noise' + str(noise) + '.png')

plt.show(block=block)