# -*- coding: utf-8 -*-
"""
Module with objects and functions for simple Bayesian Regression 
of spatial (geostatistical) exploration propblems.

Screening Bayes code ported/adapted from Matlab code by KetilH from 2018.

Created on Mon Mar 30 00:25:09 2020

@author: KetilH (kehok@equinor.com)
"""

import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

#-----------------------------------------------------------------
# ScreeningBayesRegressor: 
# The old method ported from Matlab.
# Computing probability of success from known locations where
# success criteria are fulfilled (e.g. location of geothermal plants)
#-----------------------------------------------------------------

class ScreeningBayesRegressor:
    """
    ScreeningBayesRegressor object
    
    Parameters
    ----------
    n_estimator: int, optional
        Number of models to build
    kbest, bool, optional (default is False)
        Return best case if True, average if False
    verbose: int, optional
        Dump some shit if verbose>0
        
    Programmed: KetilH, 29. April 2020
    """
   
    #-----------------------------------------------
    # Initialize the object
    #-----------------------------------------------
    
    def __init__(self,**kwargs):
        
        self.n_estimator = kwargs.get('n_estimator',1) # No of estimators to build (bagging)
        self.kbest = kwargs.get('kbest',False) # Plot or not?
        verbose = kwargs.get('verbose',0) # Print shit? 

        if verbose >0:
            print('ml.screen.ScreeningBayesRegressor.__init__() v0.1 (KetilH, 29. April 2020)')
            print(' o n_estimator = %d' %self.n_estimator)
    
    #-------------------------------------------------
    #   ScreeningBayesRegressor: 
    #   Fit model on features from success locations
    #-------------------------------------------------
        
    def fit(self,X, *args, **kwargs):
        
        """
        Fit Bayesian ML network on features from locations fullfilling
        the success criteria (e.g. hydrothermal vents or geothermal plants)
        
        Parameters:
        -----------
        X : array-like of shape = [n_samples, n_features] 
            The training input samples.
    
        y : array-like, shape = [n_samples], optional (default=1)
            Evidence weights (0<yi<=1)

        Returns:
        --------
        self : object

        Programmed: KetilH, 29. April 2020
        """
        
        # Check if input data is pd.DataFrame:
        if isinstance(X,pd.DataFrame):
            feat_name = list(X.columns)
            X = X.values
        
        # Dimensions of the problem
        n_estimator = self.n_estimator
        n_samp, n_feat = X.shape

        # y (used as weights) is optional
        if len(args) == 0:
            y = np.ones((n_samp,), dtype=float)
        else:
            y = args[0]
            # Check if input data is pd.Series:
            if isinstance(y, pd.Series):
                targ_name = str(y.name)
                y = y.values
        
        # Normalize y
        y = np.clip(y,a_min=0.0, a_max=None)
        y_max = np.max(y)
        y = y/y_max
        
#        print('ml.screen.ScreeningBayesRegressor.fit(X, *args, **kwargs):')
#        print(' o n_estimator = %d' %self.n_estimator)
#        print(' o n_samp, n_feat = (%d, %d)' %(n_samp,n_feat))
#        print(' o y_max = %f' %y_max)
        
        # Loop over estimators (keep the best):
        score_list, mu_list, sig_list = [], [], []
        for kk in range(n_estimator):
            
            # Draw oob samples:
            ind = rnd.randint(0, n_samp, n_samp)
            Xb = X[ind,:]
            yb = y[ind]

            # Initialize some args:
            mu  = np.ones([n_feat], dtype=float)
            sig = np.ones([n_feat], dtype=float)

            # Compute mean and variance:            
            for jj in range(n_feat):
                mu[jj]  = np.mean(yb*Xb[:,jj])
                sig[jj] = np.std(yb*Xb[:,jj])
                
            # Test prediction
            fac, arg = 1.0, np.zeros(n_samp,dtype=float) 
            for jj in range(n_feat):
                fac = fac*(1.0/(np.sqrt(2*np.pi)*sig[jj]))
                arg = arg + ((X[:,jj] - mu[jj])/sig[jj])**2
                
            post = fac*np.exp(-0.5*arg)
            score_list.append(np.sum(post)/n_samp)
            mu_list.append(mu)
            sig_list.append(sig)
            
        # Average over oob estimators:
        mu_sum = np.zeros([n_feat], dtype=float)
        sig_sum = np.zeros([n_feat], dtype=float)        
        for kk in range(n_estimator):
            mu_sum  = mu_sum  + mu_list[kk]
            sig_sum = sig_sum + sig_list[kk]
        
        self.mu_avg = mu_sum/n_estimator
        self.sig_avg = sig_sum/n_estimator
        self.score_avg = np.mean(score_list)
        
        # Best score:
        kkb = np.argmax(score_list)
        self.mu_best = mu_list[kkb]
        self.sig_best = sig_list[kkb]
        self.score_best = score_list[kkb]
        self.score_list = score_list
                
        # Return best or average from cross validation?
        if self.kbest:
            self.mu  = self.mu_best
            self.sig = self.sig_best
        else:
            self.mu  = self.mu_avg
            self.sig = self.sig_avg

    #-----------------------------------
    #   ScreeningBayesRegressor: 
    #   Predict on feature data (maps)
    #-----------------------------------

    def predict(self,X, *args):
        """
        Predict probability of success on features from map locations
        (e.g. probability of finding SMS deposits or working geothermal systems)

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.
        y : Array-like, optional (default is 1)
            Doesn't work, don't use it
            
        Returns
        -------
        post : array of shape = [n_samp]. 
            The predicted posterior 'probability of success'.
            
        Programmed: KetilH, 29. April 2020
        """
        
        # Check if input data is pd.DataFrame:
        if isinstance(X,pd.DataFrame):
            feat_name = list(X.columns)
            X = X.values

        # Dimensions of the problem:
        n_samp = X.shape[0]
        n_feat = X.shape[1]
        
        # y (used as weights) is optional
        if len(args) == 0:
            pri = np.ones((n_samp,), dtype=float)
        else:
            pri = args[0]
            # Check if input data is pd.Series:
            if isinstance(pri, pd.Series):
                targ_name = str(pri.name)
                pri = pri.values
            
        # Prediction:
        fac, arg = 1.0, np.zeros(n_samp,dtype=float) 
        mu, sig = self.mu, self.sig
        for jj in range(n_feat):
            fac = fac*(1.0/(np.sqrt(2*np.pi)*sig[jj]))
            arg = arg + ((X[:,jj] - mu[jj])/sig[jj])**2
            
        post = fac*np.exp(-0.5*arg)*pri
        
        # Normalize:
        p_max = np.max(post)
        post  = (1.0/p_max)*post
        
        return post

#----------------------------------------------------------------
#   Screening Bayes fit top routine
#----------------------------------------------------------------

def fit_sb(reg_kh, df_smp, **kwargs):
    """ Simple screening training top function 
    
    Embarrisingly simple method based on Bayes rule.
    
    The regressor can be used withot evidence data. Only a set of 
    success locations with features is needed.
    
    Parameters
    ----------
    reg_kh : ScreeningBayesRegressor object
        Initialized object
    df_smp: pd.DataFrame
        Dataframe with feature samples to train model
        Columns with lat and lon anre not used
        
    **kwargs
    --------
    key_p: str, optional (default is 'pos')
        Header to be used for output probability of df
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    verbose: int, optional (default 0)
        Print shit?
    kplot: bool, optional (default False)
        Make som e QC plots?
        
    Returns
    -------
    reg_kh : ScreeningBayesRegressor object
        The trained model for 'probability of success'.
        
    Programmed: KetilH, 20. Ocotber 2020
    """
        
    key_p = kwargs.get('key_p','pos') # df column header for probability of success (pos) 
    key_x = kwargs.get('key_x','lon') # df column header for lon 
    key_y = kwargs.get('key_y','lat') # df column header for lat 
    
    verbose = kwargs.get('verbose',0) # Print shit? 
    kplot = kwargs.get('kplot',False) # Plot or not?

    # Drop co-ordinates, if present:
    try:
        arrX = df_smp.drop(columns=[key_x, key_y]).values
        print('ml.screen.fit_sb: dropped coordinates')
    except:
        arrX = df_smp.values

    # Feature names:
    feat_list = df_smp.columns.tolist()
    n_samp, n_feat = arrX.shape

#    if verbose >0:
#    print('ml.screen.fit_sb:')
#    print(' o n_samp, n_feat = (%d, %d)' %(n_samp,n_feat))
        
    # Fit
    reg_kh.fit(arrX)

    # PLot the distributions:
    if kplot:
        
        # Units for xlabels:
        feat_unit = kwargs.get('feat_unit', ['[-]' for ii in range(n_feat)])        
        
        # Subplots:
        if n_feat < 5:
            npcol = 2
        else:
            npcol = 3
        nprow = np.int(np.ceil(n_feat/npcol))
        
        fig, ax = plt.subplots(nprow,npcol,figsize=(npcol*4,nprow*2.4))
        fig.tight_layout(pad=4.0)

        mu, sig = reg_kh.mu, reg_kh.sig   
        mu_a, sig_a = reg_kh.mu_avg , reg_kh.sig_avg  
        mu_b, sig_b = reg_kh.mu_best, reg_kh.sig_best 

        for jj in range(n_feat):
            kj, ki = jj//npcol, jj%npcol
            
            fmin, fmax = np.min(arrX[:,jj]), np.max(arrX[:,jj])
            delf = fmax-fmin
            
            # The one used
            xp = np.linspace(fmin-delf,fmax+delf,1001)
            fac = (1.0/(np.sqrt(2*np.pi)*sig[jj]))
            arg = ((xp - mu[jj])/sig[jj])**2
            pdf = fac*np.exp(-0.5*arg)
            
            # Average:
            fac = (1.0/(np.sqrt(2*np.pi)*sig_a[jj]))
            arg = ((xp - mu_a[jj])/sig_a[jj])**2
            pdf_a = fac*np.exp(-0.5*arg)
            
            # Best:
            fac = (1.0/(np.sqrt(2*np.pi)*sig_b[jj]))
            arg = ((xp - mu_b[jj])/sig_b[jj])**2
            pdf_b = fac*np.exp(-0.5*arg)
            
            # Plot:
            nb = np.maximum(n_samp//20, 20)
            xb = np.linspace(fmin-delf, fmax+delf, nb)
            #print(xb.shape)
            ax[kj][ki].hist(arrX[:,jj], xb, density=True)
            
            ax[kj][ki].plot(xp, pdf_a,'k-')
            ax[kj][ki].plot(xp, pdf_b,'b-')
            ax[kj][ki].plot(xp, pdf,'r-')
            
            ax[kj][ki].set_xlabel(feat_list[jj] + feat_unit[jj], fontsize=14)

        # Store the figure object
        test_kh = {}
        test_kh['fig_hist'] = fig
        
    # Return regressor object
    return reg_kh, test_kh

#----------------------------------------------------------------
#   Screening Bayes predict top routine
#----------------------------------------------------------------

def predict_sb(reg_kh, df_grd, **kwargs):
    """ Simple screening training top function 
    
    Embarrisingly simple method based on Bayes rule.
    
    The regressor can be used withot evidence data. Only a set of 
    success locations with features is needed.
    
    Parameters
    ----------
    reg_kh : ScreeningBayesRegressor object
        Initialized object
    df_grd: pd.DataFrame
        Dataframe with feature maps for prediction
        Must also contain columns with lon and lat
        
    **kwargs
    --------
    key_p: str, optional (default is 'pos')
        Header to be used for output probability of df
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    verbose: int, optional (default 0)
        Print shit?
    kplot: bool, optional (default False)
        Make som e QC plots?
        
    Returns
    -------
    reg_kh : ScreeningBayesRegressor object
        The trained model for 'probability of success'.
        
    Programmed: KetilH, 20. Ocotber 2020
    """
    
    # Get the kwargs:    
    key_p = kwargs.get('key_p','pos') # df column header for probability of success (pos) 
    key_x = kwargs.get('key_x','lon') # df column header for lon 
    key_y = kwargs.get('key_y','lat') # df column header for lat 
    verbose = kwargs.get('verbose',0) # Print shit? 
    #kplot = kwargs.get('kplot',False) # Plot or not?

    # Get feature list:
    feat_list = df_grd.drop(columns=[key_x, key_y]).columns.tolist()
#    feat_unit = kwargs.get('feat_unit', ['[-]' for ii in range(n_feat)])        
    
    # Predict:    
    kh_arr = reg_kh.predict(df_grd[feat_list]) # output is np.array

    # Store in a pd.DataFrame
    kh_maps = pd.DataFrame(columns=[key_x,key_y,key_p])
    kh_maps[[key_x, key_y]] = df_grd[[key_x, key_y]]
    kh_maps[key_p] = kh_arr

    return kh_maps

def plot_post_map(df_post, key_p, **kwargs):
    """ Plot posterior "probability of success" computed by 
        screening Bayes
    
    Parameters
    ----------
    df_post: pd.DataFrame
        Posterior "probability of success"
    key_p: str
        key of target variable in dataframe
        
    **kwargs
    --------
    nx: int
        Size of grid in x (shape=(ny,nx))
    ny: int
        Size of grid in y (shape=(ny,nx))
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    unit_x: str, optional (default '[-]')
    unit_y: str, optional (default '[-]')
    xmin: float, optional (default min(df_post[key_x]) )
    xmax: float, optional (default max(df_post[key_x]) )
    ymin: float, optional (default min(df_post[key_y]) )
    ymax: float, optional (default max(df_post[key_y]) )
    sclp: float, optional (default is 1.0)
    poly: list of structs, optional (default [])
    title: str, optional 
    verbose: int, optional (default 0)
        Print shit?
        
    Returns
    -------
    fig: Figure object
       
    Programmed: KetilH 21. October 2020
    """ 

    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    sclp = kwargs.get('sclp', 1.0)
    xmin = kwargs.get('xmin', np.min(df_post[key_x]))
    xmax = kwargs.get('xmax', np.max(df_post[key_x]))
    ymin = kwargs.get('ymin', np.min(df_post[key_y]))
    ymax = kwargs.get('ymax', np.max(df_post[key_y]))
    unit_x = kwargs.get('unit_x', '[-]') # Column key for x or longitude 
    unit_y = kwargs.get('unit_y', unit_x) # Column key for y or latitude
    poly = kwargs.get('poly', []) # subtract 2 for the lon/lat
    title  = kwargs.get('title', 'Predicted target ') # Plot title
    verbose = kwargs.get('verbose', 0)
    
    # Must be given for reshape:
    nx = kwargs.get('nx', 0)
    ny = kwargs.get('ny', 0)
    
    if nx*ny == 0:
        print('ml.screen.plot_post_map: Must give nx=, ny= for reshape')
        return 0

    # Print some stuff?
    if verbose:
        print('ml.screen.plot_post_map')
        print(' o nx, ny = %d, %d' %(nx, ny))

    # Make a mask for whatever;
    mask = np.ones([ny, nx], dtype=float)
        
    xtnt = [xmin, xmax, ymin, ymax]
    xsp, ysp = 12, 5
    
    # CRS colormap:    
    crs_wrk = [[1,0,0,1], [1,1,0,1], [1,1,0,1], [1,1,0,1], 
              [0,1,0,1], [0,1,0,1], [0,1,0,1], [0,1,0,1]]
    crs = ListedColormap(crs_wrk,len(crs_wrk))
    
    fig,ax=plt.subplots(figsize=(xsp, ysp))
    grd_wrk = sclp*np.array(df_post[key_p]).reshape(ny, nx)   
    
    im=ax.imshow(grd_wrk,cmap=crs, origin='lower',extent=xtnt,interpolation='bicubic')
    
    cm.ScalarMappable.set_clim(im,vmin=0,vmax=1)
    #cbar = ax.figure.colorbar(im, ax=ax) 
    #cbar.set_label('Posterior [-]')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(key_x + unit_x, fontsize=12)
    ax.set_ylabel(key_y + unit_y, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # PLot polygons
    for kk in range(len(poly)):
        ls = poly[kk].get('ls', '-')
        col = poly[kk].get('col', 'k')
        lab = poly[kk].get('label', '')
        ax.plot(poly[kk][key_x], poly[kk][key_y], c=col, 
                     ls=ls, lw=1, label=lab)
    
    ax.legend()
    
    return fig
