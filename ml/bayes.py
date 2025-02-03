# -*- coding: utf-8 -*-
"""
Module with objects and functions for simple Bayesian Regression 
of spatial (geostatistical) exploration propblems.

Histogram Bayes code developed in Python by KetilH, May 2020.

Created on Mon Mar 30 00:25:09 2020

@author: KetilH (kehok@equinor.com)
"""

import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.cluster import k_means

#-----------------------------------------------------------------
# HistBayesRegressor: 
# New method using histograms to obtain conditional probability
# distributions for use in Bayes theorem.
# Computing posterior mean and variance of the target variable
# Object methods: self.fit(X,y), self.precict(X)
#-----------------------------------------------------------------

class HistBayesRegressor:
    """
    HistBayesRegressor object
    
    Parameters
    ----------
    nbin_t: int, optional (default is 10)
        Number of histogram bins for target variable
    nbin_f: int, optional (default is 10)
        Number of histogram bins for feature variables (same for all)
    n_estimator: int, optional
        Number of models to build
    estimator: string, optional (default is MAP)
        Estimator may be set to MEAN or MAP (max aposteriori)
        
    resfac: float, optional (default = 1)
        Interpolation resampling factor for histograms
    bal: float, optional (default 0.0)
        Normalize histigram per target bin if bal>0
    boost: int, optional (default 1)
        Apply boosting step (once or twice if boost = 1 or 2)
    boost_wgt: list of 3 elements, optional (default [1,1,1]) 
    
    verbose: int, optional
        Dump some shit if verbose>0
                
    Programmed: KetilH, 5. May 2020
    """
        
    def __init__(self,**kwargs):
        
        self.n_estimator = kwargs.get('n_estimator',1) # No of estimators to build (bagging)
        self.estimator = kwargs.get('estimator','Mean') # Mean or MAP

        self.nbin_t = kwargs.get('nbin_t',8) # Histogram binning of target
        self.nbin_f = kwargs.get('nbin_f',10) # Histogram binning of features

        self.resfac = kwargs.get('resfac',1) # Resampling of pdf for target variable
        self.bal = kwargs.get('bal',0)       # Balancing histogramd per target bin
        self.boost = kwargs.get('boost',1)   # Boosting
        self.boost_wgt = kwargs.get('boost_wgt',[1.0,1.0,1.0])   # Boosting weights

        self.verbose = kwargs.get('verbose',0) # Print shit? 
        
        if self.verbose > 0:
            print('HistBayesRegressor.__init__() v0.1 (KetilH, 5. May 2020)')
            print(' o n_estimator = %d' %self.n_estimator)
            print(' o estimator = %s' %self.estimator)
            print(' o nbin_t = %d' %self.nbin_t)
            print(' o nbin_f = %d' %self.nbin_f)
            print(' o resfac = %f' %self.resfac)
            print(' o bal = %f' %self.bal)
            print(' o boost = %d' %self.boost)

    #---------------------------------------------------------
    #   HistBayesRegressor: 
    #   Fit model on histogram distributions from evidence
    #---------------------------------------------------------
        
    def fit(self, X, y, **kwargs):
        
        """
        Fit Bayesian ML network on features from locations fullfilling
        the success criteria (e.g. hydrothermal vents or geothermal plants)
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.
        y : array-like of shape = [n_samples] 
            Target, evidence
            
        estimator: string, optional (default is MAP)
            Estimator may be set to MEAN or MAP (max aposteriori)
            Overrides value set in __init__
        For QC plotting
        ---------------    
        kplot: bool, optional (default False)
            Make som e QC plots?
        krl: bool, optional (default True)
            Plot red line on histograms?    
        kkk: Int, optinal. Over-rides self.kkk
            Plot mlh and posterior distribution for sample kkk

        Returns
        -------
        self: object

        Programmed: KetilH, 11. May 2020
        """
        
        kplot = kwargs.get('kplot', True)
        krl = kwargs.get('krl', True)
        kkk = kwargs.get('kkk', 0)
        
        # Mean or MAP?
        estimator = kwargs.get('estimator',self.estimator)

        # Dimensions of the problem
        n_estimator = self.n_estimator
        nbin_t, nbin_f = self.nbin_t, self.nbin_f
        n_samp, n_feat = X.shape[0], X.shape[1]

        # Parameters set by __init__:
        resfac = self.resfac     # Resampling factor for pdf integration
        bal = self.bal           # Histogram balancing (use bal=0)
        boost = self.boost       # Boosting if boost>0
        
        # Used for histogram ploting:
        feat_list = kwargs.get('feat_list',['feat_'+str(jj) for jj in range(n_feat)])
        key_t     = kwargs.get('key_t',  'targ')

        # Print some shit:
        if self.verbose > 0:
            print('HistBayesRegressor.fit(X, y, **kwargs):')
            print(' o n_estimator = %d' %n_estimator)
            print(' o estimator = %s' %estimator)
            print(' o n_samp, n_feat = (%d, %d)' %(n_samp,n_feat))
            print(' o nbin_t, nbin_f = (%d, %d)' %(nbin_t, nbin_f))
            print(' o resfac = %f' %resfac)
            print(' o bal = %f' %bal)
            print(' o boost = %d' %boost)
           
        # Make consitent bin edges for all histograms:
        bins_t = np.histogram_bin_edges(y, bins=nbin_t)
        bins_f = [0 for jj in range(n_feat)]
        for jj in range(n_feat):
            bins_f[jj] = np.histogram_bin_edges(X[:,jj], bins=nbin_f)

        rmse = [0 for kk in range(n_estimator)]
        r2_score = [0 for kk in range(n_estimator)]
        rmse_best, r2_best = np.inf, -np.inf
        
        # Loop over estimators (keep the best):
        for kk in range(n_estimator):
                        
            # Draw oob samples:
            ind = rnd.randint(0, n_samp, n_samp)
            Xoob, yoob = X[ind,:], y[ind]
            
            # Initial fitting: Conditional histogram distributions p(feat|targ):
            hist_list, hist_targ = hist_dist(Xoob, yoob, bins_f, bins_t, bal=bal,
                                             feat_list=feat_list, 
                                             key_t=key_t)

            # Predict on train data:
            #print('First: Xoob.shape[0]=%d' %Xoob.shape[0])
            post0 = stat_pred(Xoob, hist_list, hist_targ, estimator, 
                              kplot=kplot, kkk=kkk, krl=krl)
            
            # Boosting?
            if boost >= 1:
                
                # Samples for the boosting step:
                err = np.abs(yoob-post0['mu'])
                #print(np.mean(err), np.median(err))
                jnd = err > np.median(err)
                X1, y1, = Xoob[jnd,:], yoob[jnd]
                
                # Boosting histograms:
                hist_list1, hist_targ1 = hist_dist(X1, y1, bins_f, bins_t, bal=bal)
                post1 = stat_pred(X1, hist_list1, hist_targ1, estimator, kplot=False)

                # Averaging:
                w0, w1, w2 = self.boost_wgt
                for jj in range(len(hist_list)):
                    h0 = hist_list[jj]['hist']
                    h1 = hist_list1[jj]['hist']
                    hist_list[jj]['hist'] = (w0*h0 + w1*h1)/(w0+w1)
                    #hist_list[jj]['hist'][0,:] = 1

            # 2nd boosting?
            if boost == 2:
                pass
                # TODO: Finish this part

            # Run boosted prediction on all samples:
            #print('Second: Xoob.shape[0]=%d' %Xoob.shape[0])
            post_boost = stat_pred(Xoob, hist_list, hist_targ, estimator, 
                                   kplot=kplot, kkk=kkk, krl=krl)

            # Don't make a Hell of a lot of plots:
            if kplot:
                qc_figs = [] # Store figure handles for qc
                qc_figs.append([post0['fig_hist'], post0['fig_samp']])
                qc_figs.append([post_boost['fig_hist'], post_boost['fig_samp']])
                kplot = False
            
            # Best so far?
            rmse[kk] = np.sqrt( np.sum( (yoob-post_boost['mu'])**2)/n_samp )
            sst = np.sum( (yoob - np.mean(yoob))**2 )
            ssr = np.sum( (yoob - post_boost['mu'])**2 )
            r2_score[kk] =  1.0 - ssr/sst
#            if r2_score[kk] > r2_best:
#                r2_best = r2_score[kk]
            if rmse[kk] < rmse_best:
                rmse_best = rmse[kk]
                kest_best = kk
                hist_list_best = hist_list
                hist_targ_best = hist_targ
                
        # Keep the best:
        self.kest_best = kest_best
        self.hist_list = hist_list_best
        self.hist_targ = hist_targ_best
        self.rmse_score = rmse
        self.r2_score = r2_score
        self.qc_figs = qc_figs

    #----------------------------------------
    #  END: fit()
    #----------------------------------------
    
    def predict(self, X, **kwargs):
        
        """
        Fit Bayesian ML network on features from locations fullfilling
        the success criteria (e.g. hydrothermal vents or geothermal plants)
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Features for prediction of target
        estimator: string, optional (default is MAP)
            Estimator may be set to MEAN or MAP (max aposteriori)
            Overrides value set in __init__
            
        For QC plotting
        ---------------    
        kplot: bool, optional (default False)
            Make som e QC plots?
        krl: bool, optional (default True)
            Plot red line on histograms?    
        kkk: Int, optinal. Over-rides self.kkk
            Plot mlh and posterior distribution for sample kkk

        Returns
        -------
        post: Dictionary of arrays
            Dict with elements: mu, sig, qc

        Programmed: KetilH, 11. May 2020
        """
        
        # Mean or MAP?
        estimator = kwargs.get('estimator', self.estimator)
        kplot = kwargs.get('kplot', False)
        krl = kwargs.get('krl', False)
        kkk = kwargs.get('kkk', 0)

        # Print some shit:
        if self.verbose > 0:
            print('HistBayesRegressor.predict(X, **kwargs):')
            print(' o estimator = %s' %estimator)

        # Predict on data:
        post = stat_pred(X, self.hist_list, self.hist_targ, 
                         estimator=estimator, krl=krl, kkk=kkk) 

        return post
        
    #----------------------------------------
    #  END: predict()
    #----------------------------------------
    
####################   START NEW   ###########################

#----------------------------------------------------------------
#   fit top routine
#----------------------------------------------------------------

def fit_nb(reg_nb, df_smp, key_t, **kwargs):
    
    """ Fit Naive Bayes regressor on a set of target and feature samples.
            
    Splitting of train and test can be done in two ways:
        1. Random splitting
        2. Spatial splitting (using clustering)
    
    Parameters
    ----------
    reg_nb: initialized regressor object
        Regressor for estimating the mean and variance
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
    estimator: string, optional (default is MAP)
        Estimator may be set to MEAN or MAP (max aposteriori)
        Overrides value set in __init__
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    verbose: int, optional (default 0)
        Print shit?
     For QC plotting
    ---------------    
    kplot: bool, optional (default False)
        Make som e QC plots?
    kkk: int, optional (default 0)
        Plot mlh and posterior distribution for sample kkk
    krl: bool, optional (default True)
        Plot red line on histograms?    
        
    Returns
    -------
    reg_nb: regressor object
        Model for estimating the mean, best model from CV1
    test_nb: dict
        Regressor for estimating the mean, best from CV
        
    Programmed: KetilH 22. October 2020
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
    
    # For QC plotting:
    kplot = kwargs.get('kplot' ,False)
    kkk = kwargs.get('kkk' ,0)
    krl = kwargs.get('krl' ,False)

    if verbose:
        print('ml.bayes.fit_nb:')
        print(' o split = %s' %split)
        print(' o test_size = %.2f' %test_size)
        print(' o n_cv = %d' %n_cv)
        print(' o use_all = %r' %use_all)

    # Get the list of features
    feat_list = df_smp.columns.drop([key_x,key_y, key_t]).tolist()
    
    ### Train/test split 1: Different for spatial and random   

    if split.lower()[0] == 's':
        
        print('Branch: Spatial split')
        
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
        trainX = df_smp[df_clu[key_c] != kkk][feat_list] 
        trainy = df_smp[df_clu[key_c] != kkk][key_t]
                
        # QC plot the clustering
        if kplot:
            color = ['r','b','k','m','c','g','r','b','k','m','c','g']
            fig_clu = plt.figure()
            for jj in range(n_clu):
                if jj == kkk:
                    marker = 'x'
                else:
                    marker = '.'
           
                plt.scatter(df_smp[df_clu['clu'] == jj][key_x],
                            df_smp[df_clu['clu'] == jj][key_y],
                            marker=marker, c=color[jj])
            plt.xlabel(key_x)
            plt.ylabel(key_y)
            plt.title('kMeans clustering')
            
    else:
        
        print('Branch: Random split')

        # Add a dummy cluster index, all different:
        df_clu = pd.DataFrame(index=df_smp.index, columns=[key_c], 
                              data =df_smp.index) # Dummy data = index
        
        # Random splitting:  X=feature_matrix, y=target_vector
        trainX, testX, trainy, testy = train_test_split(df_smp[feat_list], 
                                   df_smp[key_t], test_size=test_size, 
                                   shuffle=True, random_state=0)
  
    # Train model. For the time being np.arrays must be input
    reg_nb.fit(trainX.values, trainy.values, 
               kplot=kplot, kkk=kkk, krl=krl)
           
    # Predict on held out test data
    Xwrk, ywrk = testX.values, testy.values
    post_test = reg_nb.predict(Xwrk)

    # Test output for QC
    test_nb = {}
    test_nb['targ'] = ywrk
    test_nb['mu']  = post_test['mu']
    test_nb['sig'] = post_test['sig']
    test_nb['r2'] = reg_nb.r2_score
    test_nb['rmse'] = reg_nb.rmse_score
    test_nb['feat_list'] = feat_list

    test_nb['qc_figs'] = reg_nb.qc_figs 
    try: test_nb['fig_clu'] = fig_clu
    except: pass

    # Return estimators and score stats:
    return reg_nb, test_nb

#----------------------------------------------------------------
#   predict top routine
#----------------------------------------------------------------

def predict_nb(reg_nb, df_grd, key_t, **kwargs):
    
    """ Predict target map from a set of featiure maps.
    
    Two basic predictions are performed:
        1. Predict the target mean
        2. Predict the target  variance
    The prediction is done with a single Bayesian ml model.
        
    Subsequently the mean and can be used to compute P10/P50/P90 estimates
    
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
    estimator: string, optional (default is MAP)
        Estimator may be set to MEAN or MAP (max aposteriori)
        Overrides value set in reg_nb.__init__
        
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
        
    Programmed: KetilH 23. October 2020 (Happy Birthday)
    """
    
    # Get the kwargs:
    estimator = kwargs.get('estimator', reg_nb.estimator) # Column key for x or longitude 
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    
    verbose = kwargs.get('verbose', 0)

    if verbose:
        print('ml.bayes.predict_nb:')

    # Get the list of features
    feat_list = df_grd.columns.drop([key_x, key_y]).tolist()
    pred_list = ['P10','P50','P90','mu','sig']
    
    # Predict the mean and variance:
    pred_arr = reg_nb.predict(df_grd[feat_list].values, estimator=estimator)
    
    # Output pd.DatFrame:
    pred_maps = pd.DataFrame(columns=[key_x, key_y] + pred_list)
    pred_maps[[key_x, key_y]] = df_grd[[key_x, key_y]]

    # Output maps:
    sf = 1.28 # To compute P10 and P90
    pred_maps[pred_list[0]] = pred_arr['mu'] + sf*pred_arr['sig']
    pred_maps[pred_list[1]] = pred_arr['mu'] 
    pred_maps[pred_list[2]] = pred_arr['mu'] - sf*pred_arr['sig']
    pred_maps[pred_list[3]] = pred_arr['mu'] 
    pred_maps[pred_list[4]] = pred_arr['sig']
    
    # Clip on zero:
    for ii, key in enumerate(pred_list):
        pred_maps[key] = np.clip(pred_maps[key], a_min=0, a_max=None)
    
    return pred_maps

####################   END   NEW   ###########################
    
#--------------------------------------------------------------
#   Helper fuctions on module level, independent of classes
#   Functions can be called by ml.bayes.func_name()
#--------------------------------------------------------------
    
def hist_dist(X, y, bins_f, bins_t, **kwargs):
    """
    Helper function for bayesml.fit().
    Bin and compute histograms distributions of feature and target data.
    One histogram for each feature and the target.
    
    Parameters
    ----------
    X: numpy-array, shape = (n_samp, n_feat)
        Array of features (as in sklearn)
    y: numpy-array, shape = (n_samp)
        Array of targets (as in sklearn)
    bins_f: list of numpy-arrays of float
        Bin edges for binning of features
    bins_t: numpy-array of float
        Bin edges for binning of target
    bal: float, optional (float, dedault=0)
        Balance(normalize) histograms for each target bin, 
        bal=0 for no normalization
    feat_list: list of strings (optional, default is '')
        Names of features. Used only for labeling plots
    key_t: string (optional, default is '')
        Name of target. Used only for labeling plots
    
    Returns
    -------
    hist_list : length n_feat list of dicts {hist, targ, feat, rbsi, marg, inti}
    hist_targ : dict with target marginal {marg, targ, inti}

    Programmed: KetilH, 7. May 2020
    """
    
    # Sizes:
    n_samp, n_feat = X.shape
    nbin_f, nbin_t = bins_f[0].shape[0]-1, bins_t.shape[0]-1

    # Some optional stuff:
    bal = kwargs.get('bal',0)
    feat_list = kwargs.get('feat_list','')
    key_t = kwargs.get('key_t','')
    
    if len(feat_list) == 0:
        feat_list = ['feat_'+str(kk) for kk in range(n_feat)]
    
    if len(key_t) == 0:
        key_t = 'target'
        
    # Marginal for the target:
    marg, tedg = np.histogram(y, nbin_t)
    targ = 0.5*(tedg[0:-1] + tedg[1:]) # Bin centers  
    hist_targ = {'marg': marg, 'targ': targ, 'name': key_t, 'unit': ' [?]'}

    # Compute histogram distributions for all features
    hist_list = [0 for jj in range(n_feat)] # List of p(f|t)
    
    for jj in range(n_feat):
        
        # Histogram distribution:
        hist, tedg, fedg = np.histogram2d(y, X[:,jj], [bins_t, bins_f[jj]])
        feat = 0.5*(fedg[0:-1] + fedg[1:]) # Bin centers
        targ = 0.5*(tedg[0:-1] + tedg[1:]) # Bin centers
        
        # Avg no of target samples per bin:
        ns_avg = np.sum(hist)/nbin_t 
        
        # Balance histograms
        for ii in range(nbin_t):
            rnorm = ns_avg/np.maximum( np.sum(hist[ii,:]), 1 )
            hist[ii,:] = hist[ii,:]*(rnorm**bal)
        
        # Marginal:
        marg = np.sum(hist,axis=0)

        # Gather stuff in a dict
        hist_list[jj] = {'hist': hist, 'targ': targ, 'feat': feat, 'marg': marg,
                         'name': feat_list[jj], 'unit': ' [?]'}

    return hist_list, hist_targ
        
def stat_pred(X, hist_list, hist_targ, estimator, **kwargs):
    """
    Helper function for bayesml.fit() and bayesml.predict().
    Compute posterior mean and variance, given histogram distributions 
    
    Parameters
    ----------
    X: numpy-array, shape = (n_samp, n_feat)
        Array of features (as in sklearn)
    hist_list : length n_feat list of dicts {hist, targ, feat, rbsi, marg, inti}
    hist_targ : dict with target marginal {marg, targ, inti}
    estimator : str
        Estimator for the mean; estimater = MAP or Mean
        
    kwargs
    ------
    resfac: float, optional (default 4)
        Resampling factor for histogram interpolation
    kplot: bool, optional (default False)
    kkk: int, optional (default 0)
        Plot mlh and posterior distribution for sample kkk
    krl: bool, optional (default True)
        Plot red line on hostograms?
               
    Returns
    -------
    post: Dictionary of arrays
        Dict with elements: mu, sig, qc
    
    Programmed: KetilH, 7. May 2020
    """
    # Get kwargs:
    kplot = kwargs.get('kplot', False)
    kkk = kwargs.get('kkk',0)
    krl = kwargs.get('krl',True)

    # Resampling factor (optional input):
    resfac = kwargs.get('resfac', 4)

    # Get number of samples and features:
    n_samp, n_feat = X.shape

    # Get number of bins
    nbin_t, nbin_f = hist_list[0]['hist'].shape

    # Define a dense target range for integration:
    targ_min, targ_max =  np.min(hist_list[0]['targ']), np.max(hist_list[0]['targ'])
    #dtarg = (targ_max-targ_min)/(nbin_t-1)
    ntax = np.int(np.round( resfac*(nbin_t - 1) )) + 1
    tax  = np.linspace(targ_min, targ_max, ntax) # pdf integration variable
    dtax = tax[1] - tax[0] 
    
    # Interpolation order: Linear makes sure that pdf>0:
    kind = 'linear' # polynomial order for interp.interp1d
    kx, ky = 1, 1   # spline order for interp.RectBivariateSpline

    # Interpolator object for the target histogram:
    targ, marg = hist_targ['targ'], hist_targ['marg']
    hist_targ['inti'] = interp.interp1d(targ ,marg, kind=kind) #1d interpolator object for the marginal

    marg_targ = hist_targ['inti'](tax)
    marg_targ = marg_targ/(dtax*np.sum(marg_targ))
    
    # Interpolater object for the pdf. Function call: hd2 = rbsi(td,tf)
    for jj in range(n_feat):
        feat, targ = hist_list[jj]['feat'], hist_list[jj]['targ']
        hist, marg = hist_list[jj]['hist'], hist_list[jj]['marg']
        hist_list[jj]['rbsi'] = interp.RectBivariateSpline(targ, feat, hist, kx=kx, ky=ky) 
        hist_list[jj]['inti'] = interp.interp1d(feat,marg) #1d interpolator object for the marginal

    # Loop over samples
    mu_post = np.zeros(n_samp, dtype=float)
    sig_post = np.zeros(n_samp, dtype=float)
    for kk in range(n_samp):
        
        # Build posterior:
        pdf_post = np.ones(ntax, dtype=float)
        mlh = [0 for ii in range(n_feat)]
        
        # Loop over features for current sample:
        for jj in range(n_feat):
            
            # Get the MLH for current feature:
            mlh[jj] = hist_list[jj]['rbsi'](tax, X[kk,jj]).flatten() # 2d numpy array; flatten !!!

            # Avoid all-zeros:
            if np.sum(mlh[jj]<1.0):
                mlh[jj][0] = 1.0

            # Normalize MLH:
            mlh[jj] = mlh[jj]/(dtax*np.sum(mlh[jj]))

            # Upfate posterior:
            pdf_post = pdf_post*mlh[jj]
        
        if (np.min(pdf_post)<0):
            print('Impossible pdf<0: kk = %d' %kk)
        
        # Multiply with the prior:
        # pdf_post = pdf_post/marg_targ
        
        # Normalize posterior:
        pdf_post = pdf_post/(dtax*np.sum(pdf_post))
        
        if estimator.lower() == 'map':
            jmax = np.argmax(pdf_post)
            mu_post[kk] = tax[jmax]
            sig_post[kk] = np.sqrt( dtax*np.sum(((mu_post[kk]-tax)**2)*pdf_post) )/1.0
        else:
            mu_post[kk] = dtax*np.sum(tax*pdf_post)
            sig_post[kk] = np.sqrt( dtax*np.sum(((mu_post[kk]-tax)**2)*pdf_post) )/1.0 
            
        # Store selected for plotting:   
        if kk == kkk:
            qc = {'post': pdf_post, 'mlh': mlh, 'tax': tax, 'kkk': kk, 
                  'mu_post': mu_post[kk], 'sig_post': sig_post[kk]}
                      
    # QC plotting:    
    if kplot:
                
        #nhs = np.int(np.sum(hist_list[0]['hist']))
        nhs = n_samp
        
        if n_feat > 4:
            npcol = 3   
        else:
            npcol = 2
        
        nprow = np.maximum(np.int(np.ceil(n_feat/npcol)),2)
        fig, ax = plt.subplots(nprow,npcol,figsize=(4.0*npcol,3.0*nprow))
        fig.tight_layout(pad=4.0)
        fig.suptitle('Histograms for '+str(nhs)+' labeled samples', fontsize=14)
        for jj in range(n_feat):
            kj, ki = jj//npcol, jj%npcol
            feat_min = np.min(hist_list[jj]['feat'])
            feat_max = np.max(hist_list[jj]['feat'])
            dfeat = hist_list[jj]['feat'][1] - hist_list[jj]['feat'][0]
            xtnt = [feat_min-dfeat/2, feat_max+dfeat/2, targ_min-dtax/2, targ_max+dtax/2]
            im = ax[kj,ki].imshow(hist_list[jj]['hist'], origin='lower', extent=xtnt, aspect='auto')
            xred = X[kkk,jj]*np.array([1,1],dtype=float) 
            yred = np.array([targ_min-dtax/2, targ_max+dtax/2],dtype=float)
            if krl:
                ax[kj][ki].plot(xred, yred, 'r-')
            ax[kj][ki].set_xlim(feat_min-dfeat/2, feat_max+dfeat/2)
            ax[kj][ki].set_ylim(targ_min-dtax/2, targ_max+dtax/2)
            ax[kj][ki].set_xlabel(hist_list[jj]['name'] + hist_list[jj]['unit']); 
            ax[kj][ki].set_ylabel(hist_targ['name'] + hist_targ['unit']); 
            #ax[kj][ki].set_title(targName, fontsize=14)
            cbar = ax[kj][ki].figure.colorbar(im, ax=ax[kj][ki])
            #cbar.set_label(targName+targUnit)
            
        if krl:
            fig2 = plt.figure()
            for jj in range(n_feat):
                plt.plot(qc['tax'], qc['mlh'][jj], label='MLH ' + hist_list[jj]['name'])
            plt.plot(qc['tax'], qc['post'], linewidth=3, label='Posterior')   
            plt.legend()
            plt.xlabel(hist_targ['name'] + hist_targ['unit'] )
            plt.ylabel('pdf [-]')
            plt.suptitle('Posterior and likelihood for sample ' + str(qc['kkk']))
    
#    post = {'mu': mu_post, 'sig': sig_post, 'qc': qc}        
    post = {'mu': mu_post, 'sig': sig_post}        
    
    # Return figure objects:
    try: post['fig_hist'] = fig
    except: post['fig_hist'] = None
    
    try: post['fig_samp'] = fig2
    except: post['fig_samp'] = None
        
    return post    

