# -*- coding: utf-8 -*-
"""
Module for geostatistical regression-type machine learning


Created on Mon Oct 12 14:23:16 2020
@author: kehok@equinor.com
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as rnd
from scipy.interpolate import griddata

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.cluster import k_means

#------------------------------------------------------------------
#  Fit ML regressors for a set of target and feature samples
#------------------------------------------------------------------

def fit_cv(reg_mu, reg_sig, df_smp, key_t, **kwargs):
    
    """ Fit ML regressors by cross validation 
        for a set of target and feature samples.
    
    Two estimators are computed by CV:
        1. ML model for the mean
        2. ML model for the variance
        
    Splitting of train and test can be done in two ways:
        1. Random splitting
        2. Spatial splitting (using clustering)

    Parameters
    ----------
    reg_mu: initialized regressor object
        Regressor for estimating the mean
    reg_sig: initialized regressor object
        Regressor for estimating the variance
    df_smp: pd.DataFrame
        Target and feature samples for train and test
        Columns of the dataframe are assumed to be:
        [lon, lat, target, features] (in any order)
    key_t: str
        Target key (column name in df)
        
    **kwargs
    --------
    split: str, optional (default is 'random')
        Random or spatial train/test splitting
    test_size: float, optional (default is 0.2)
        Fraction of samples to be used for test
    n_cv: int, optional (deafult is 5)
        Cross validation fold
    qc_cv: bool, optional (default False)
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
    reg_mu: regressor object
        Model for estimating the mean, best model from CV1
    reg_sig: regressor object
        Model for estimating the variance, best model from CV2
    test_mu: dict
        Regressor for estimating the mean, best from CV
    test_sig: dict
        Test prediction and score for variance
        
    Programmed: KetilH 15. October 2020
    """
    
    # Get the kwargs:
    split = kwargs.get('split', 'random')
    test_size = kwargs.get('test_size', 0.2) 
    n_cv = kwargs.get('n_cv', 5) 
    use_all = kwargs.get('use_all', True)
    qc_cv = kwargs.get('qc_cv', False)

    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    key_c = 'clu' # for internal use
    
    verbose = kwargs.get('verbose', 0)
    kplot = kwargs.get('kplot', False)

    # Add some dummies for key_x, key_y if necessary
    if not key_x in df_smp.keys(): df_smp[key_x] = df_smp.index
    if not key_y in df_smp.keys(): df_smp[key_y] = df_smp.index
    
    if verbose:
        print('ml.greg.fit_cv:')
        print(f' o key_x, key_y = {key_x}, {key_y}')
        print(f' o key_t = {key_t}')
        print(f' o split = {split}')
        print(f' o test_size = {test_size}')
        print(f' o n_cv = {n_cv}')
        print(f' o use_all = {use_all}')

    # Get the list of features
    feat_list = df_smp.columns.drop([key_x, key_y, key_t]).tolist()
    
    ### Train/test split 1: Different for spatial and random   
    if split.lower()[0] == 's':
        
        if verbose>1: print('Branch: Spatial split')
        
        # Spatial clustering using kMeans
        n_clu = n_cv + 1
        centroid, clu_id, inertia = k_means(df_smp[[key_x, key_y]], n_clu)
        
        # Keep track of cluster id by df_smp.index (for use in CV groups):
        df_clu = pd.DataFrame(index=df_smp.index, columns=[key_c], data=clu_id)
        
        # Random hold-out cluster for final testing:
        kkk = rnd.randint(0,n_clu,1)[0]  
        testX  = df_smp[df_clu[key_c] == kkk][feat_list]
        testy  = df_smp[df_clu[key_c] == kkk][key_t]
        # Clusters for training:
        trainX0 = df_smp[df_clu[key_c] != kkk][feat_list] 
        trainy0 = df_smp[df_clu[key_c] != kkk][key_t]
                
        # QC plot the clustering
        if kplot:
            color = ['r','b','k','m','c','g','r','b','k','m','c','g']
            fig_qc, ax = plt.subplots(1,1, figsize=(12,5))
            for jj in range(n_clu):
                if jj == kkk:
                    marker = 'x'
                else:
                    marker = '.'
           
                ax.scatter(df_smp[df_clu['clu'] == jj][key_x],
                           df_smp[df_clu['clu'] == jj][key_y],
                           marker=marker, c=color[jj])
            ax.set_xlabel(key_x)
            ax.set_ylabel(key_y)
            ax.set_title('kMeans clustering')
            ax.set_aspect('equal')

    else:
        
        if verbose>1: print('Branch: Random split')

        # Add a dummy cluster index, all different:
        df_clu = pd.DataFrame(index=df_smp.index, columns=[key_c], 
                              data =df_smp.index) # Dummy data = index
        
        # Random splitting:  X=feature_matrix, y=target_vector
        trainX0, testX, trainy0, testy = train_test_split(df_smp[feat_list], 
                                   df_smp[key_t], test_size=test_size, 
                                   shuffle=True, random_state=0)
        
        # QC plot for random split
        if kplot:
            fig_qc, ax = plt.subplots(1,1, figsize=(12,5))
            # Coordinates get lost in train_test_split; reference via index
            ax.scatter(df_smp.loc[trainy0.index,key_x], 
                       df_smp.loc[trainy0.index,key_y], marker='.', c='b')
            ax.scatter(df_smp.loc[testy.index,key_x], 
                       df_smp.loc[testy.index,key_y], marker='.', c='r')
            ax.set_xlabel(key_x)
            ax.set_ylabel(key_y)
            ax.set_title('Random split')
            ax.set_aspect('equal')
        
    ### Train/test split 1: Same for spatial and random   
    
    test_size2 = 0.5 # always
    trainX1, trainX2, trainy1, trainy2 = train_test_split(trainX0, trainy0,
                               test_size=test_size2, shuffle=True, random_state=0)   
        
    # Use all for mean model:
    if use_all:
        trainX1, trainy1 = trainX0, trainy0
        
    if split.lower()[0] == 's':
        # Keep track of cluster ids (for use in CV):
        grp1, grp2 = df_clu.loc[trainX1.index, key_c], df_clu.loc[trainX2.index, key_c] 
    else:
        # Groups not needed for Random split:s
        grp1, grp2 = None, None 
        
    ####  Cross-validation 1: Train mean model 
        
    # n-fold crossvalidation
    scoring = ['r2', 'explained_variance', 'neg_mean_squared_error']
    cv_mu = cross_validate(reg_mu,trainX1,trainy1,
                           cv=n_cv, groups=grp1, 
                           scoring=scoring,
                           return_train_score=True,
                           return_estimator=True)    

    # Test CV models for mean on held out data
    keys_mu = ['pred', 'r2', 'ev', 'mase']
    test_mu = {key: [np.nan for ii in range(n_cv)] for key in keys_mu}
    test_mu['targ'] = np.array(testy)      # Same for all
    for jj in range(n_cv):
        test_mu['pred'][jj] = cv_mu['estimator'][jj].predict(testX)
        test_mu['r2'][jj] = r2_score(testy, test_mu['pred'][jj])
        test_mu['ev'][jj] = explained_variance_score(testy, test_mu['pred'][jj])  
        test_mu['mase'][jj] = mase_score(testy, test_mu['pred'][jj])

    # Best model for mean:
    ind = np.array(test_mu['r2']) <= cv_mu['train_r2']   # To avoid underfitting
    kbest_mu = np.argmax(np.array(test_mu['r2'])[ind])
    #kbest_mu = np.argmax(test_mu['r2'])
    reg_mu = cv_mu['estimator'][kbest_mu]
    test_mu['kbest'] = kbest_mu
    test_mu['feat_list'] = feat_list
    test_mu['split']  = split
    
    # Return full CV dict for QC?
    if qc_cv:
        cv_mu.pop('estimator') # No need to store all
        test_mu['cv'] = cv_mu
    
    # Return fig-handle to k_means clusters?
    try:
        test_mu['fig_clu'] = fig_qc 
        test_mu['fig_qc']  = fig_qc 
    except: pass

    ####  Cross-validation 1: Train variance model 
    
    # Using the best estimator to compute mean for Step2
    trainy2_se = (trainy2 - reg_mu.predict(trainX2))**2
    cv_sig = cross_validate(reg_sig,trainX2,trainy2_se,
                            cv=n_cv, groups=grp2, 
                            scoring=scoring,
                            return_train_score=True,
                            return_estimator=True)    
   
    # Test CV models for variance on held out data
    keys_sig = ['pred', 'r2', 'ev', 'mase']
    test_sig = {key: [np.nan for ii in range(n_cv)] for key in keys_sig}
    testy_se = (testy - reg_mu.predict(testX))**2   # square error
    testy_rmse = np.sqrt(testy_se)                  # linear error
    test_sig['targ'] = testy_rmse                   # same for all
    for jj in range(n_cv):
        try:
            test_sig['pred'][jj] = np.sqrt(cv_sig['estimator'][jj].predict(testX))
            test_sig['r2'][jj] = r2_score(testy_rmse, test_sig['pred'][jj])
            test_sig['ev'][jj] = explained_variance_score(testy_rmse, test_sig['pred'][jj])    
        except:
            test_sig['pred'][jj] = testy_rmse
            test_sig['r2'][jj] = -1
            test_sig['ev'][jj] = -1          
        test_sig['mase'][jj] = mase_score(testy_rmse, test_sig['pred'][jj])
            
    # Best model for variance:
    ind = np.array(test_sig['r2']) <= cv_sig['train_r2']   # To avoid underfitting
    kbest_sig = np.argmax(np.array(test_sig['r2'])[ind])
    #kbest_sig = np.argmax(test_sig['r2'])
    reg_sig = cv_sig['estimator'][kbest_sig]
    test_sig['kbest'] = kbest_sig
    test_sig['feat_list'] = feat_list
    test_sig['split'] = split    
 
    # Return full CV dict for QC?
    if qc_cv:
        cv_sig.pop('estimator') # No need to store all
        test_sig['cv'] = cv_sig

    # Return estimators and score stats:
    return reg_mu, reg_sig, test_mu, test_sig

#------------------------------------------------------------------
#  Fit ML regressors for a set of target and feature samples
#------------------------------------------------------------------

def predict_ml(reg_mu, reg_sig, df_grd, key_t, **kwargs):
    
    """ Predict target map from a set of featiure maps.
    
    Two basic predictions are performed:
        1. Predict the target mean
        2. Predict the target  variance
    The prediction is done using two ML models, one for each step..
    
    Subsequently the mean and variance can be used to compute 
    the P10/P50/P90 estimates
    
    Parameters
    ----------
    reg_mu: initialized regressor object
        Regressor for estimating the mean
    reg_sig: initialized regressor object
        Regressor for estimating the variance
    df_grd: pd.DataFrame
        Feature maps for prediction of target
        [lon, lat, features] (in any order)
    key_t: str
        Target key (column root name in output df)
        
    **kwargs
    --------
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
    pred_maps: pd.DataFrame
        Target maps predicted by ML models
        
    Programmed: KetilH 16. October 2020
    """
    
    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    
    verbose = kwargs.get('verbose', 0)

    # Add some dummies for key_x, key_y if necessary
    if not key_x in df_grd.keys(): df_grd[key_x] = df_grd.index
    if not key_y in df_grd.keys(): df_grd[key_y] = df_grd.index

    if verbose:
        print('ml.greg.predict_ml:')
        print(' o key_t = %s' %key_t)

    # Get the list of features
    feat_list = df_grd.columns.drop([key_x, key_y]).tolist()
    perc_list = ['P90','P50','P10','mu','sig']
    
    # Predict the mean
    pred_mu   = reg_mu.predict(df_grd[feat_list])
    
    # Predict the variance squared
    pred_sig = np.sqrt(reg_sig.predict(df_grd[feat_list]))
    
    # Output pd.DatFrame:
    pred_maps = pd.DataFrame(columns=[key_x, key_y] + perc_list)
    pred_maps[[key_x, key_y]] = df_grd[[key_x, key_y]]

    # Output maps:
    sf = 1.28 # To compute P10 and P90
    pred_maps[perc_list[0]] = pred_mu - sf*pred_sig
    pred_maps[perc_list[1]] = pred_mu
    pred_maps[perc_list[2]] = pred_mu + sf*pred_sig
    pred_maps[perc_list[3]] = pred_mu
    pred_maps[perc_list[4]] = pred_sig
    
    # Clip on zero:
    for ii, key in enumerate(perc_list):
        pred_maps[key] = np.clip(pred_maps[key], a_min=0, a_max=None)
    
    return pred_maps
    
#----------------------------------------------------------
#    MASE score
#----------------------------------------------------------

def mase_score(y_true, y_pred):
    """ Compute Mean Absolute Scaled Error (MASE) score  """

    nn = y_true.shape[0]
    rnum = y_true - y_pred
    rden = y_true - np.mean(y_true)
    mase = np.sum(np.abs(rnum)/np.abs(rden))/float(nn)

    return mase

#----------------------------------------------------------
#   Halluscinate some data
#----------------------------------------------------------

def add_halluc(df_grd, df_smp, n_hal, val_hal, key_t, **kwargs):
    
    """ Add hallucinatory data to test/train data
    
    Parameters
    ----------
    df_grd: pd.DataFrame
        Feature data on spatial grid
    df_smp: pd.DataFrame
        Target and feature samples for train and test
    n_hal: int
        Number of halluscinatory data points to add
    val_hal: int or array
        Value to fill inn for halluscinatory data
    key_t: str
        Target key (column name in df)
        
    **kwargs
    --------
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    xmin: float, optional (default min(df_grd[key_x]) )
    xmax: float, optional (default max(df_grd[key_x]) )
    ymin: float, optional (default min(df_grd[key_y]) )
    ymax: float, optional (default max(df_grd[key_y]) )
    verbose: int, optional (default 0)
        Print shit?
    kplot: bool, optional (default False)
        Plot haluscinatory data locations
        
    Returns
    -------
    df_smp: pd:dataFrame
        Input samples with hallusinatory samples appended    
        
    Programmed: KetilH 12. October 2020
    """
    
    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    xmin = kwargs.get('xmin', np.min(df_grd[key_x]))
    xmax = kwargs.get('xmax', np.max(df_grd[key_x]))
    ymin = kwargs.get('ymin', np.min(df_grd[key_y]))
    ymax = kwargs.get('ymax', np.max(df_grd[key_y]))
    verbose = kwargs.get('verbose', 0)
    kplot = kwargs.get('kplot', False)
    
    # Print some shit?
    if verbose:
        print('ml.greg.add_halluc: n_hal = %d' %n_hal)
        print(' o xmin, xmax = %s' %[xmin, xmax])
        print(' o ymin, ymax = %s' %[ymin, ymax])
        print(' o key_t = %s' %key_t)
        
    # Unemployed?
    if n_hal == 0:
        return df_smp
    
    # Hallucinate some spatial locations
    xh = xmin + (xmax-xmin)*rnd.rand(n_hal)
    yh = ymin + (ymax-ymin)*rnd.rand(n_hal)    
    
    # One or more halucinatory target values given?
    if not hasattr(val_hal,'__len__'):
        vh = val_hal*np.ones_like(xh)
    elif len(val_hal) < n_hal:
        vh = val_hal[0]*np.ones_like(xh)
    else:
        vh = val_hal
    
    # Append hallucinatory positions and target:
    df_wrk = pd.DataFrame(index=range(n_hal), columns=df_smp.columns)
    df_wrk[key_x] = xh
    df_wrk[key_y] = yh
    df_wrk[key_t] = vh
    
    # Interpolate features from the maps
    feat_list = df_grd.columns.drop([key_x,key_y]).tolist()
    pts = np.array([df_grd[key_x],df_grd[key_y]], dtype=float).T
    for col in feat_list:
        fillVal = np.mean(df_grd[col]) # value to impute for nan
        df_wrk[col] = griddata(pts,df_grd[col],(xh,yh),
                               method='linear', fill_value=fillVal)

    # Append to sample data:
    df_smp = df_smp.append(df_wrk,ignore_index=True)
    
    # Plot?
    if kplot:
        plt.figure()
        plt.plot(df_smp[key_x], df_smp[key_y], 'r.', label='measured data')
        plt.plot(df_wrk[key_x], df_wrk[key_y], 'b.', label='hallucinated data')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(key_x)
        plt.ylabel(key_y)
        plt.legend()
        plt.axis('equal') 
        
    return df_smp

#----------------------------------------------------------
#    PLot functions
#------------------------------------------------------------

def plot_correl(df_smp, key_t, **kwargs):
    """ Plot some data analytics """
    
    # Get the kwargs
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude
    title = kwargs.get('title', 'Feature correlation') # Plot title
    
    # Features
    feat_list = df_smp.columns.drop([key_x,key_y, key_t]).tolist()
    
    # Pearson correlation:
    pear_list = [key_t] + feat_list
    n_pear = len(pear_list)
    rr_pear = df_smp[pear_list].corr()

    # Make a heatmap
    fig = plt.figure(figsize=(12,10))
    ax = sns.heatmap(rr_pear, annot=True,
                     xticklabels=pear_list, yticklabels=pear_list,
                     cmap=cm.Reds, vmin=-1, vmax=1, square=True)
    ax.set_yticklabels(pear_list, rotation=0)
    ax.set_title(title, fontsize=14)
    bot, top = plt.ylim()
    plt.ylim(bot+0.5,top-0.5)    
    fig.tight_layout(pad=1.0)
        
    return fig

def plot_feat_importance(feat_imp, **kwargs):
    """ Plot feature importance barchart """

    # Get kwargs
    nf = feat_imp.shape[0]
    feat_name = kwargs.get('feat_name', ['feat_'+str(ii)  for ii in range(nf)])
    title = kwargs.get('title', 'Relative importance')

    # Sort to get indices:
    ind = np.argsort(feat_imp)

    # Plot barchart
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    ax.barh(np.array(feat_name)[ind], feat_imp[ind])
    ax.set_xlabel('Relative importance [-]')
    ax.set_title(title, fontsize=14)
    fig.tight_layout(pad=1.0)

    return fig

def plot_confusion(pred, targ, *args, **kwargs):
    """ Plot confusion """

    # Get kwargs
    key_t = kwargs.get('key_t', 'target')
    unit_t = kwargs.get('unit_t', '[-]')
    title = kwargs.get('title', 'Confusion')
    
    # Get the args
    if len(args) > 0: 
        pred_sig = args[0]
    else:   
        pred_sig = np.zeros_like(pred)

    # 45 degree line
    vmin = np.min([np.min(pred-pred_sig), np.min(targ)])
    vmax = np.max([np.max(pred+pred_sig), np.max(targ)])
    a45 = np.array([vmin, vmax],dtype=float)

    # Make plot
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    fig.tight_layout(pad=4.0)
    ax.plot(a45,a45,'k-')
    ax.errorbar(targ, pred, pred_sig, c='r', fmt='.')
    ax.scatter(targ, pred, c='r',marker='o')
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal','box')
    ax.set_xlim(vmin,vmax)
    ax.set_ylim(vmin,vmax)
    ax.set_ylabel(f'Predicted {key_t}  {unit_t}')
    ax.set_xlabel(f'Observed {key_t}  {unit_t}')
    fig.tight_layout(pad=1.0)

    return fig

def plot_targ_samp(df_smp, key_t, **kwargs):
    """Plot target samples (the evidence for training the ML model) 
            Parameters
    ----------
    df_smp: pd.DataFrame
        Feature data on spatial grid
    key_t: str
        The key for the target in the pd.DataFrame
        
    **kwargs
    --------
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    unit_x: str, optional (default '[-]')
    unit_y: str, optional (default '[-]')
    unit_t: str, optional (default '[-]')
    xmin: float, optional (default min(df_smp[key_x]) )
    xmax: float, optional (default max(df_smp[key_x]) )
    ymin: float, optional (default min(df_smp[key_y]) )
    ymax: float, optional (default max(df_smp[key_y]) )
    vmin: float, optional (default min(df_smp[key_t]) )
    vmax: float, optional (default max(df_smp[key_t]) )
    cmap: colormap, optional (default cm.jet)
    poly: list of dicts, optional (default [])
    verbose: int, optional (default 0)
        Print shit?
        
    Returns
    -------
    fig: Figure object
       
    Programmed: KetilH 14. October 2020
    """
    
    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude
    xmin  = kwargs.get('xmin', np.min(df_smp[key_x]))
    xmax  = kwargs.get('xmax', np.max(df_smp[key_x]))
    ymin  = kwargs.get('ymin', np.min(df_smp[key_y]))
    ymax  = kwargs.get('ymax', np.max(df_smp[key_y]))
    vmin  = kwargs.get('vmin', np.min(df_smp[key_t]))
    vmax  = kwargs.get('vmax', np.max(df_smp[key_t]))
    cmap  = kwargs.get('cmap', cm.jet)
    unit_x = kwargs.get('unit_x', '[-]') # Column key for x or longitude 
    unit_y = kwargs.get('unit_y', unit_x) # Column key for y or latitude
    unit_t = kwargs.get('unit_t', '[-]') # Column key for target
    poly   = kwargs.get('poly', [])        
    title  = kwargs.get('title', 'Target samples: ' + key_t) # Plot title
    label  = kwargs.get('label', key_t) # Plot title

    print(xmin, xmax)
    print(ymin,ymax)

    # Fig size
    size, rat = 10, (ymax-ymin)/(xmax-xmin)
    xsp, ysp = size, rat*size
    fig,ax=plt.subplots(figsize=(xsp, ysp))
    
    # Plot sample data
    im = ax.scatter(df_smp[key_x], df_smp[key_y], s=8, c=df_smp[key_t], 
                    cmap=cmap, marker='o', label=label)
    cm.ScalarMappable.set_clim(im, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax); 
    cbar.set_label(key_t + unit_t)
    
    # Set axes labels:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(key_x + unit_x, fontsize=12)
    ax.set_ylabel(key_y + unit_y, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Plot polygons:
    for jj in range(len(poly)):
        ls = poly[jj].get('ls', '-')
        col = poly[jj].get('col', 'k')
        lab = poly[jj].get('label', '')
        ax.plot(poly[jj][key_x], poly[jj][key_y], c=col, ls=ls, lw=1, label=lab)
    
    ax.legend()
    ax.axis('equal')
    fig.tight_layout(pad=2.0)
   
    return fig
    
def plot_feat_maps(df_grd, **kwargs):
    """ Plot feature maps (input to ML)
    
        Parameters
    ----------
    df_grd: pd.DataFrame
        Feature data on spatial grid
        
    **kwargs
    --------
    nx: int (must be given)
        Size of grid in x (shape=(ny,nx))
    ny: int (must be given)
        Size of grid in y (shape=(ny,nx))
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    unit_x: str, optional (default '[-]')
    unit_y: str, optional (default '[-]')
    xmin: float, optional (default min(df_grd[key_x]) )
    xmax: float, optional (default max(df_grd[key_x]) )
    ymin: float, optional (default min(df_grd[key_y]) )
    ymax: float, optional (default max(df_grd[key_y]) )
    cmap: colormap, optional (default cm.jet)
    poly: list of dicts, optional (default [])
    aspect: float, aspect ratio, passed to imshow
    verbose: int, optional (default 0)
        Print shit?
        
    Returns
    -------
    fig: Figure object
       
    Programmed: KetilH 14. October 2020
    """ 
    
    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    xmin = kwargs.get('xmin', np.min(df_grd[key_x]))
    xmax = kwargs.get('xmax', np.max(df_grd[key_x]))
    ymin = kwargs.get('ymin', np.min(df_grd[key_y]))
    ymax = kwargs.get('ymax', np.max(df_grd[key_y]))
    unit_x = kwargs.get('unit_x', '[-]') # Column key for x or longitude 
    unit_y = kwargs.get('unit_y', unit_x) # Column key for y or latitude
    cmap = kwargs.get('cmap', cm.jet)
    poly = kwargs.get('poly', []) # subtract 2 for the lon/lat
    aspect = kwargs.get('aspect', 'equal')
    verbose = kwargs.get('verbose', 0)
    
    # UTM or lon,lat?
    if (key_x[0].lower() == 'x') | (key_x.lower() == 'y'):
        scl = 1.0e-3
        unit_x = unit_y = '[km]'
    else:
        scl = 1.0

    # Must be given for reshape:
    nx = kwargs.get('nx', 0)
    ny = kwargs.get('ny', 0)
    
    if nx*ny == 0:
        print('plot_feat_maps: Must give nx=, ny= for reshape')
        return 0

    # Get feature list from df:
    feat_list = df_grd.columns.drop([key_x, key_y]).tolist()
    n_feat = len(feat_list)

    # Print some stuff?
    if verbose:
        print('ml.greg.plot_feat_maps')
        print(' o nx, ny = %d, %d' %(nx, ny))
        print(' o n_feat = %d' %n_feat)
        print(' o nx, ny = %d, %d' %(nx, ny))
        print(' o unit_x, unit_y = %s, %s' %(unit_x, unit_y))
    
    # PLot size:
    #np_row, np_col = (n_feat+3)//3, 3
    np_row, np_col = (n_feat+2)//3, 3
    # xsize, ysize= 8.0*np_col, 3.0*np_row    
    scl_xsize, scl_ysize = 3.0*(xmax-xmin)/(ymax-ymin), 3.0
    xsize, ysize= scl_xsize*np_col, scl_ysize*np_row    
    
    #xtnt = [scl*xmin, scl*xmax, scl*ymin, scl*ymax]
    xtnt = [scl*np.min(df_grd[key_x]), scl*np.max(df_grd[key_x]), 
            scl*np.min(df_grd[key_y]), scl*np.max(df_grd[key_y])]
    
    if verbose >1:
        print('ml.greg.plot_feat_maps:')
        print(' o xsize, ysize = {}, {}'.format(xsize, ysize))

    fig, ax = plt.subplots(np_row, np_col, figsize=(xsize, ysize))
    fig.tight_layout(pad=4.0)
    
    # Loop over features
    for jf, feat in enumerate(feat_list):
        
        jj, ii = jf//np_col, jf % np_col
        
        if verbose >0:
            print('plot feat=%s: jf, jj, ii = %d, %d, %d' %(feat,jf,jj,ii))
        
        grd = np.array(df_grd[feat]).reshape(ny, nx)
        im = ax[jj][ii].imshow(grd,origin='lower', extent=xtnt, 
                               cmap=cmap, aspect=aspect)

        ax[jj][ii].set_xlim(scl*xmin, scl*xmax)
        ax[jj][ii].set_ylim(scl*ymin, scl*ymax)
        ax[jj][ii].set_xlabel(key_x + unit_x); 
        ax[jj][ii].set_ylabel(key_y + unit_y); 
        ax[jj][ii].set_title(feat, fontsize=14)

        cbar = ax[jj][ii].figure.colorbar(im, ax=ax[jj][ii])
        cm.ScalarMappable.set_clim(im,vmin=np.nanmin(grd),vmax=np.nanmax(grd))
        cbar.set_label(feat)

        plt.tight_layout()
    
        # PLot polygons
        for kk in range(len(poly)):
            ls = poly[kk].get('ls', '-')
            col = poly[kk].get('col', 'k')
            ax[jj][ii].plot(scl*poly[kk][key_x], scl*poly[kk][key_y], c=col, ls=ls, lw=1)
            
    return fig
    
def plot_pred_maps(df_pred, key_t, **kwargs):
    """ Plot target maps P10/P50/P90 (output from ML)
    
    Parameters
    ----------
    df_pred: pd.DataFrame
        Feature data on spatial grid
    key_t: str
        key of target variable in dataframe
        
    **kwargs
    --------
    pred_perc: str or list of str 
        Percentiles to plot (default ['P90', 'P50', 'P10'])
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
    unit_t: str, optional (default '[-]')
    xmin: float, optional (default min(df_pred[key_x]) )
    xmax: float, optional (default max(df_pred[key_x]) )
    ymin: float, optional (default min(df_pred[key_y]) )
    ymax: float, optional (default max(df_pred[key_y]) )
    cmap: colormap, optional (default cm.jet)
    poly: list of dicts, optional (default [])
    ticks: list, ticks on colorbar
    title: str, optional 
    aspect: float, aspect ratio, passed to imshow
    verbose: int, optional (default 0)
        Print shit?
        
    Returns
    -------
    fig: Figure object
       
    Programmed: KetilH 14. October 2020
    """ 

    # First things first:    
    pred_perc  = kwargs.get('pred_perc',['P90', 'P50', 'P10'])
    if not isinstance(pred_perc, list): pred_perc = [pred_perc]

    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    vmin = kwargs.get('vmin', np.nanmin(df_pred[pred_perc[0]]))
    vmax = kwargs.get('vmax', np.nanmax(df_pred[pred_perc[0]]))
    xmin = kwargs.get('xmin', np.min(df_pred[key_x]))
    xmax = kwargs.get('xmax', np.max(df_pred[key_x]))
    ymin = kwargs.get('ymin', np.min(df_pred[key_y]))
    ymax = kwargs.get('ymax', np.max(df_pred[key_y]))
    unit_t = kwargs.get('unit_t', '[-]') # Column key for target prediction 
    unit_x = kwargs.get('unit_x', '[-]') # Column key for x or longitude 
    unit_y = kwargs.get('unit_y', unit_x) # Column key for y or latitude
    cmap = kwargs.get('cmap', cm.jet)
    ticks = kwargs.get('ticks', None)
    poly = kwargs.get('poly', []) # subtract 2 for the lon/lat
    title  = kwargs.get('title', 'Predicted target ') # Plot title
    aspect = kwargs.get('aspect', 'equal')
    verbose = kwargs.get('verbose', 0)
    
    # UTM or lon,lat?
    if (key_x[0].lower() == 'x') | (key_x.lower() == 'y'):
        scl = 1.0e-3
        unit_x = unit_y = '[km]'
    else:
        scl = 1.0
    
    # Must be given for reshape:
    nx = kwargs.get('nx', 0)
    ny = kwargs.get('ny', 0)
    
    if nx*ny == 0:
        print('plot_feat_maps: Must give nx=, ny= for reshape')
        return 0

    # Print some stuff?
    if verbose:
        print('ml.greg.plot_pred_maps')
        print(' o nx, ny = {}, {}'.format(nx, ny))
        
    # Make a mask for whatever;
    mask = np.ones([ny, nx], dtype=float)
    
#    xsp, ysp = 10,12
    np_row, np_col = 1, len(pred_perc)    
    scl_xsize, scl_ysize = 3.0*(xmax-xmin)/(ymax-ymin), 5.0
    xsize, ysize= scl_xsize*np_col, scl_ysize*np_row    
    
    xtnt = [scl*np.min(df_pred[key_x]), scl*np.max(df_pred[key_x]), 
            scl*np.min(df_pred[key_y]), scl*np.max(df_pred[key_y])]
    
    #
    #   OK HIT 14/10-2020 kl 21:49
    #
        
    # PLot predictions
    fig, axs=plt.subplots(np_row, np_col, figsize=(xsize, ysize))
    for ii, key in enumerate(pred_perc):

        if verbose>1: print(' o ii, key = {:d}, {:s}'.format(ii,key))
        
        if isinstance(axs,np.ndarray): ax = axs[ii]
        else: ax = axs
        
        grdwrk = np.array(df_pred[key]).reshape(ny,nx)
        im=ax.imshow(mask*grdwrk, cmap=cmap, origin='lower',
                         extent=xtnt,interpolation='bicubic', aspect=aspect)
        
        cm.ScalarMappable.set_clim(im, vmin=vmin, vmax=vmax)
        cbar = ax.figure.colorbar(im, ax=ax, ticks=ticks) 
        cbar.set_label(key_t + '_' + key + unit_t)
        
        ax.set_xlim(scl*xmin, scl*xmax)
        ax.set_ylim(scl*ymin, scl*ymax)
        ax.set_title(title + pred_perc[ii])
        ax.set_xlabel(key_x + unit_x) 
        ax.set_ylabel(key_y + unit_y)
        
        # PLot polygons
        for kk in range(len(poly)):
            ls = poly[kk].get('ls', '-')
            col = poly[kk].get('col', 'k')
            lab = poly[kk].get('label', '')
            ax.plot(scl*poly[kk][key_x], scl*poly[kk][key_y], c=col, 
                        ls=ls, lw=1, label=lab)
        
        if len(poly)>0: ax.legend()
        #plt.tight_layout()

    fig.tight_layout(pad=2.0)

    return fig

def plot_scores(test_mu, test_sig, score_list, **kwargs):
    """ Plot test scores from cross validation """
    
    title  = kwargs.get('title', '') # Plot title

    nscore = len(score_list)
    fig, ax = plt.subplots(1, nscore, figsize=(3*nscore,4))
    fig.suptitle(title)
    fig.tight_layout(pad=4.0)
    
    for ii, ai in enumerate(ax):
        score = score_list[ii]
        indx = [jj for jj in range(len(test_mu[score]))]
    
        ai.scatter(indx, test_mu[score], label='CV1')
        ai.scatter(indx, test_sig[score], label='CV2')
        ai.legend()

        ai.set_title(score)
        ai.set_xlabel('CV iter [-]')
        ai.set_ylabel('Score [-]')

    fig.tight_layout(pad=1.0)
    return fig

