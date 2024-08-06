"""
Modified version of Ketil H's module for geostatistical regression-type machine learning 
to classification-type machine learning


Created on summer 2021
@author: sekr@equinor.com
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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.cluster import k_means
from sklearn.metrics import confusion_matrix


def fit_cv(clf, df_smp, key_t, **kwargs):
    
    """ Fit ML classifier by cross validation 
        for a set of target and feature samples.
    
        
    Splitting of train and test can be done in two ways:
        1. Random splitting
        2. Spatial splitting (using clustering)
    
    Parameters
    ----------
    clf: initialized classifier object
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
    n_cv: int, optional (default is 5)
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
    clf: classifier object
        Model for estimating the class, best model from CV based on accuracy
    test: dict
        Classifier for estimating the class, best from CV based on accuracy
        
    Programmed: Stine Ekrheim summer 2021 (Based on KetilH version from 15. October 2020)
    """
    
    # Get the kwargs:
    split = kwargs.get('split', 'random')
    test_size = kwargs.get('test_size', 0.2) 
    n_cv = kwargs.get('n_cv', 5) 
    qc_cv = kwargs.get('qc_cv', False)

    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    key_c = 'clu' # for internal use
    
    verbose = kwargs.get('verbose', 0)
    kplot = kwargs.get('kplot', False)

    if verbose:
        print('ml.gclas.fit_cv:')
        print(' o key_t = %s' %key_t)
        print(' o split = %s' %split)
        print(' o test_size = %.2f' %test_size)
        print(' o n_cv = %d' %n_cv)

    # Get the list of features
    feat_list = df_smp.columns.drop([key_x,key_y, key_t]).tolist()
                                                                         
 
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
            ax.set_title('Train&test: kMeans clustering')
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
            ax.set_title('Train&test: Random split')
            ax.set_aspect('equal')
                
    if split.lower()[0] == 's':
        # Keep track of cluster ids (for use in CV):
        grp = df_clu.loc[trainX0.index, key_c]
    else:
        # Groups not needed for Random split:s
        grp = None
        

    ####  Cross-validation 1: Train model 
        
    # n-fold crossvalidation
    scoring =['accuracy','precision_macro', 'recall_macro','f1_macro'] 
        
    cv = cross_validate(clf,trainX0,trainy0,
                        cv=n_cv, groups=grp, 
                        scoring=scoring,
                        return_train_score=True,
                        return_estimator=True)    

   # Test CV models for held out data
    keys = ['pred','accu', 'prec', 'recall','f1']
    test = {key: [np.nan for ii in range(n_cv)] for key in keys}
    test['targ'] = np.array(testy)      # Same for all
    for jj in range(n_cv):
        test['pred'][jj] = cv['estimator'][jj].predict(testX)
        test['accu'][jj] = accuracy_score(testy, test['pred'][jj])
        test['prec'][jj] = precision_score(testy, test['pred'][jj],average='macro')  
        test['recall'][jj] = recall_score(testy, test['pred'][jj],average='macro')
        test['f1'][jj] = f1_score(testy, test['pred'][jj],average='macro')
        

    # Best model:
    ind = np.array(test['accu']) <= cv['train_accuracy']   # To avoid underfitting
    kbest = np.argmax(np.array(test['accu'])[ind])
    clf = cv['estimator'][kbest]
    test['kbest'] = kbest
    test['feat_list'] = feat_list
    test['split']  = split
    
    # Return full CV dict for QC?
    if qc_cv:
        cv.pop('estimator') # No need to store all
        test['cv'] = cv
            
    # Return fig-handle to k_means clusters?
    try:
        test['fig_clu'] = fig_qc 
        test['fig_qc']  = fig_qc 
    except: pass

    # Return estimators and score stats:
    return clf, test

#------------------------------------------------------------------
#  Fit ML regressors for a set of target and feature samples
#------------------------------------------------------------------

def predict_ml(clf, df_grd, key_t, **kwargs):
    
    """ Predict target map from a set of feature maps.
    
    Predict the target class
    The prediction is done using classification ML
    
    Parameters
    ----------
    clf: initialized regressor object
        Regressor for estimating the mean
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
        
    Programmed: KetilH 16. October 2020 (modified by StineE to fit classifiers)
    """
    
    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude 
    
    verbose = kwargs.get('verbose', 0)

    if verbose:
        print('ml.gclas.predict_ml:')
        print(' o key_t = %s' %key_t)

    # Get the list of features
    feat_list = df_grd.columns.drop([key_x, key_y]).tolist()
 
    
    # Predict
    pred = clf.predict(df_grd[feat_list])
    

    # Output pd.DatFrame:
    pred_maps = pd.DataFrame(columns=[key_x, key_y,'pred'] )
    pred_maps[[key_x, key_y]] = df_grd[[key_x, key_y]]

    # Output maps:
    pred_maps['pred'] = pred

    
    return pred_maps

#----------------------------------------------------------
#    PLot functions
#------------------------------------------------------------

def plot_scores(test, score_list, **kwargs):
    """ Plot test scores from cross validation """
    
    title  = kwargs.get('title', '') # Plot title

    fig, ax = plt.subplots(1, len(score_list), figsize=(15,4))
    fig.suptitle(title)
    fig.tight_layout(pad=4.0)
    
    for ii, ai in enumerate(ax):
        score = score_list[ii]
      #  indx = [jj for jj in range(len(test_mu[score]))]
        indx = np.arange(0,len(test[score]))
        ai.scatter(indx, test[score], label='CV')
        ai.legend()

        ai.set_title(score)
        ai.set_xlabel('CV iter [-]')
        ai.set_ylabel('Score [-]')

    return fig

def plot_confusion(pred, targ, **kwargs):
    """ Plot confusion """

    # Get kwargs
    key_t = kwargs.get('key_t', 'target')
    unit_t = kwargs.get('unit_t', '[-]')
    title = kwargs.get('title', 'Confusion Matrix')
    label = kwargs.get('label', '')

    # Make plot
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    fig.tight_layout(pad=4.0)
    
    con_mat = confusion_matrix(targ,pred)
    im = sns.heatmap(con_mat.T,annot=True,ax=ax,cmap=cm.Reds, cbar=False,
                     xticklabels = label, yticklabels = label)
    bot, top = plt.ylim()
    plt.ylim(bot+0.5,top-0.5)    
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal','box')
    ax.set_ylabel('Observesd ' + key_t  + unit_t); 
    ax.set_xlabel('Predicted ' + key_t  + unit_t); 
    plt.tight_layout()

    return fig

def plot_pred_maps(df_pred, key_t, **kwargs):
    """ Plot target maps for the preditions from ML
    
    Parameters
    ----------
    df_pred: pd.DataFrame
        Feature data on spatial grid
    key_t: str
        key of target variable in dataframe
        
    **kwargs
    --------
    pred_perc: str or list of str 
        Percentiles to plot (default ['pred'])
    numb_dict: dictionary from name of class to a number
        (default {1:1,2:2,3:3})
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
    aspect: float, aspect ratio, passed to imshow
    title: str, optional 
    verbose: int, optional (default 0)
        Print shit?
        
    Returns
    -------
    fig: Figure object
       
    Programmed: KetilH 14. October 2020
                Modified to work for classification by Stine Ekrheim (Summer 2021)
    """ 

    # First things first:    
    pred_perc  = kwargs.get('pred_perc',['pred'])
    if not isinstance(pred_perc, list): pred_perc = [pred_perc]

    # Get the kwargs:
    numb_dict = kwargs.get('numb_dict',{0:0,1:1,2:2})
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', 2)
    xmin = kwargs.get('xmin', np.min(df_pred[key_x]))
    xmax = kwargs.get('xmax', np.max(df_pred[key_x]))
    ymin = kwargs.get('ymin', np.min(df_pred[key_y]))
    ymax = kwargs.get('ymax', np.max(df_pred[key_y]))
    unit_t = kwargs.get('unit_t', '[-]') # Column key for target prediction 
    unit_x = kwargs.get('unit_x', '[-]') # Column key for x or longitude 
    unit_y = kwargs.get('unit_y', unit_x) # Column key for y or latitude
    cmap = kwargs.get('cmap', cm.rainbow)
    ticks = kwargs.get('ticks', None)
    poly = kwargs.get('poly', []) # subtract 2 for the lon/lat
    title  = kwargs.get('title', 'Predicted target ') # Plot title
    verbose = kwargs.get('verbose', 0)
    aspect = kwargs.get('aspect', 'equal')

    xsize = kwargs.get('xsize', 7.0*(xmax-xmin)/(ymax-ymin))
    ysize = kwargs.get('ysize', 5.0)
    print('xsize, ysize = {}, {}'.format(xsize, ysize))
    
    # Override ticks
    ticks = list(numb_dict.keys())
    if 'nan' in ticks: ticks.pop(ticks.index('nan'))
    
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
        print('ml.gclas.plot_pred_maps')
        print(f' o nx, ny = {nx}, {ny}')
        
    # Make a mask for whatever;
    mask = np.ones([ny, nx], dtype=float)
    
    xtnt = [scl*np.min(df_pred[key_x]), scl*np.max(df_pred[key_x]), 
            scl*np.min(df_pred[key_y]), scl*np.max(df_pred[key_y])]

    # PLot predictions
    fig, ax=plt.subplots(figsize=(xsize, ysize))
    #fig.tight_layout(pad=4.0)

    if verbose>1: print(' o ii, key = {:d}, {:s}'.format(0,pred_perc[0]))
        
    try:
        grdwrk_numb = np.array([numb_dict[jj] for jj in df_pred['pred']])
        grdwrk = np.array(grdwrk_numb).reshape(ny,nx)
    except:
        grdwrk = np.array(df_pred['pred']).reshape(ny,nx)

   
    im=ax.imshow(mask*grdwrk, cmap=cmap, origin='lower',
                 extent=xtnt, interpolation='bicubic', aspect=aspect)
        
    cm.ScalarMappable.set_clim(im, vmin=vmin, vmax=vmax)
    numb = np.arange(0,len(ticks))
    cbar = ax.figure.colorbar(im,ticks = numb, ax=ax) 
    cbar.set_label(key_t + '_' + pred_perc[0] + unit_t)
    cbar.ax.set_yticklabels(ticks)  
    ax.set_xlim(scl*xmin, scl*xmax)
    ax.set_ylim(scl*ymin, scl*ymax)
    ax.set_title(title)
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
    numb_dict: dictionary from name of class to a number
        (default {0:0,1:1,2:2})
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
    numb_dict = kwargs.get('numb_dict',{0:0,1:1,2:2,3:3})
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude
    xmin  = kwargs.get('xmin', np.min(df_smp[key_x]))
    xmax  = kwargs.get('xmax', np.max(df_smp[key_x]))
    ymin  = kwargs.get('ymin', np.min(df_smp[key_y]))
    ymax  = kwargs.get('ymax', np.max(df_smp[key_y]))
    vmin  = kwargs.get('vmin', np.min(df_smp[key_t]))
    vmax  = kwargs.get('vmax', np.max(df_smp[key_t]))
    cmap  = kwargs.get('cmap', cm.viridis)
    unit_x = kwargs.get('unit_x', '[-]') # Column key for x or longitude 
    unit_y = kwargs.get('unit_y', unit_x) # Column key for y or latitude
    unit_t = kwargs.get('unit_t', '[-]') # Column key for target
    poly   = kwargs.get('poly', [])        
    title  = kwargs.get('title', 'Target samples: ' + key_t) # Plot title
    label  = kwargs.get('label', key_t) # Plot title

    # print(xmin, xmax)
    # print(ymin,ymax)

    # Fig size
    size, rat = 8, (ymax-ymin)/(xmax-xmin)
    xsp, ysp = size, rat*size
    fig,ax=plt.subplots(figsize=(xsp, ysp))
    fig.tight_layout(pad=4.0)
    
    colors = np.array([numb_dict[jj] for jj in df_smp[key_t]])
    # Plot sample data
    im = ax.scatter(df_smp[key_x], df_smp[key_y], s=8, c=colors, 
                    cmap=cmap, marker='o')
                    #, label=label)
    cm.ScalarMappable.set_clim(im, vmin=vmin, vmax=vmax)
    numb = np.arange(0,len(numb_dict.keys()))
    print('label, numb = {}, {}'.format(label, numb))
    cbar = ax.figure.colorbar(im, ax=ax, ticks = numb)
    cbar.ax.set_yticklabels(numb_dict.keys())
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
    
    if len(poly) > 0: ax.legend()
    ax.axis('equal')
    
    return fig


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
    # To aviod that .corr() ingnor the correlation to the result
    df_smp_numb = df_smp.copy()
    df_smp_numb[key_t] = df_smp_numb[key_t].astype('category').cat.codes
    rr_pear = df_smp_numb[pear_list].corr()

    # Make a heatmap
    fig = plt.figure(figsize=(12,10))
    ax = sns.heatmap(rr_pear, annot=True,
                     xticklabels=pear_list, yticklabels=pear_list,
                     cmap=cm.Reds, vmin=-1, vmax=1, square=True)
    ax.set_yticklabels(pear_list, rotation=0)
    ax.set_title(title, fontsize=14)
    bot, top = plt.ylim()
    plt.ylim(bot+0.5,top-0.5)    
        
    return fig