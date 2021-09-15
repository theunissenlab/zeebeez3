"""
    PARD = Performance and Relevance Determination
"""

import numpy as np
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA


class PARD(object):

    def __init__(self):
        self.is_good_model = None

    def fit(self, X, y, cv_fit_function, plot=False, zscore=False, individual_feature_weights=False, verbose=False):

        if zscore:
            # zscore the data
            X = X - X.mean(axis=0)
            X = X / X.std(axis=0, ddof=1)
            y = y - y.mean(axis=0)
            y = y / y.std(ddof=1)

        nsamps,nfeatures = X.shape
        assert len(y) == nsamps, "# of data points in X and y do not match!"

        if plot:
            fig = plt.figure()
            gs = plt.GridSpec(1, 2)

            ax = plt.subplot(gs[0, 0])
            p99 = np.percentile(np.abs(X), 99)
            plt.imshow(X, interpolation='nearest', aspect='auto', vmin=-p99, vmax=p99, cmap=plt.cm.seismic)
            plt.colorbar()
            plt.title('X')

            C = np.corrcoef(X.T)
            ax = plt.subplot(gs[0, 1])
            plt.imshow(C, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot)
            plt.colorbar()
            plt.title('Cov(X)')

        # use cross validation to fit the projected data
        full_model_dict = cv_fit_function(X, y)
        if verbose:
            print('Full model performance: %0.2f' % full_model_dict['r2'])

        # perform statistical test for relevance of full model
        self.is_good_model,full_perf_stat = self.relevance(full_model_dict['lk_full'], full_model_dict['lk_null'])

        if not self.is_good_model:
            return None

        # if requested, do regression on each individual feature to get a weight and performance value
        Wind = np.zeros([nfeatures])
        perf_final_ind = np.zeros([nfeatures])

        if individual_feature_weights:
            for k in range(nfeatures):
                Xone = X[:, k].reshape([nsamps, 1])
                # compute the leave-one-out model
                one_model_dict = cv_fit_function(Xone, y)
                
                # perform statistical test for relevance
                is_rel,pstat = self.relevance(one_model_dict['lk_full'], one_model_dict['lk_null'])
                if verbose:
                    print('\tSingle Feature %d: relevant=%d, perf=%0.6f' % (k, int(is_rel), one_model_dict['r2']))
                if is_rel:
                    Wind[k] = one_model_dict['W'][0]
                    perf_final_ind[k] = one_model_dict['r2']

        full_model_dict['W_independent'] = Wind
        full_model_dict['r2_independent'] = perf_final_ind

        return full_model_dict

    def relevance(self, lk_full, lk_subset, pval_thresh=0.01):
        tstat,pval = ttest_rel(lk_full, lk_subset)
        perf_stat = lk_subset.mean() / lk_full.mean()
        return pval < pval_thresh, perf_stat


class IdentityTransformer(object):

    def fit(self, X):
        return

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
