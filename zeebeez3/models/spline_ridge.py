from copy import deepcopy
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge

from lasp.basis import cubic_spline_basis


class StagewiseSplineRidgeRegression(object):
    """
    This predictor does regression using a nonlinear spline basis, and adds features in a stagewise way only if
    they improve cross validation performance.
    """

    def __init__(self):
        pass

    def fit(self, X, y, baseline_features=list(), cv_indices=None, num_hyperparams=50, verbose=True, feature_names=None):
        nsamps,nfeatures = X.shape
        assert len(y) == nsamps

        if cv_indices is None:
            cv_indices = list(KFold(nsamps, n_folds=10))

        if feature_names is None:
            feature_names = ['%d' % k for k in range(nfeatures)]

        best_features = deepcopy(baseline_features)
        all_features = list(range(nfeatures))
        features_left = list(np.setdiff1d(all_features, best_features))

        # run the baseline model
        if len(best_features) == 0:
            Xsub = np.zeros(nsamps)
        else:
            Xsub = X[:, best_features]
        Xsub = self.spline_basis(Xsub)
        m_baseline = self.cv_fit(Xsub, y, cv_indices, num_hyperparams)
        best_r2 = max(m_baseline['r2'], 0)
        if verbose:
            print('Baseline model performance with %d features: R2=%0.2f' % (len(best_features), m_baseline['r2']))

        iter = 0
        feature_improvements = [0.]*len(best_features)
        while len(features_left) > 0:
            if verbose:
                print('Round %d of stagewise testing. Good features: %s' % (iter, ','.join([feature_names[k] for k in best_features])))
                print('\tFeatures left: %s' % ','.join([feature_names[k] for k in features_left]))
            best_feature = None
            best_feature_r2 = best_r2

            bad_features = list() # features that do not lead to an improvement in model performance
            for k in features_left:
                fi = deepcopy(best_features)
                fi.append(k)
                Xsub = X[:, fi]
                Xsp = self.spline_basis(Xsub)
                m_feature = self.cv_fit(Xsp, y, cv_indices, num_hyperparams)
                if m_feature['r2'] <= best_feature_r2:
                    if verbose:
                        print('\tFeature %s is a bad feature. best_feature_r2=%0.2f, incremental_r2=%0.2f' % (feature_names[k], best_feature_r2, m_feature['r2']))
                    bad_features.append(k)
                else:
                    if verbose:
                        print('\tFeature %s is a good feature. best_feature_r2=%0.2f, incremental_r2=%0.2f' % (feature_names[k], best_feature_r2, m_feature['r2']))
                    best_feature = k
                    best_feature_r2 = m_feature['r2']
                    features_left.remove(k)

            # remove the bad features so they're not tried again
            for bf in bad_features:
                features_left.remove(bf)

            if best_feature is not None:
                feature_improvements.append(best_feature_r2 - best_r2)
                best_features.append(best_feature)
                best_r2 = best_feature_r2

            iter += 1

        # train a final model using the best features
        Xsub = X[:, best_features]
        Xsp = self.spline_basis(Xsub)
        m_final = self.cv_fit(Xsp, y, cv_indices, num_hyperparams)
        if verbose:
            print('Final model features: R2=%0.2f, features=%s' % (m_final['r2'], ','.join([feature_names[k] for k in best_features])))
        m_final['features'] = best_features
        m_final['feature_improvements'] = np.array(feature_improvements)

        return m_final

    def spline_basis(self, X):

        nfeatures = X.shape[1]
        dof = 6
        Xsp = np.zeros([X.shape[0], nfeatures*dof])

        for k in range(nfeatures):
            si = k*dof
            ei = si + dof
            Xsp[:, si:ei] = cubic_spline_basis(X[:, k], num_knots=3)

        return Xsp

    def cv_fit(self, X, y, cv_indices, num_hyperparams):

        hparams = list(np.logspace(-2, 6, num_hyperparams))
        hparams.insert(0, 0)

        fold_perfs = list()
        for alpha in hparams:

            model_perfs = list()
            for train_i, test_i in cv_indices:
                
                Xtrain = X[train_i, :]
                ytrain = y[train_i]
                
                Xtest = X[test_i, :]
                ytest = y[test_i]

                rr = Ridge(alpha=alpha)
                rr.fit(Xtrain, ytrain)

                ypred = rr.predict(Xtest)

                sst = np.sum((ytest - ytrain.mean())**2)
                sse = np.sum((ytest - ypred)**2)
                r2 = 1. - (sse / sst)

                model_perfs.append({'r2':r2, 'W':rr.coef_, 'b':rr.intercept_})

            mean_r2 = np.mean([d['r2'] for d in model_perfs])
            mean_W = np.mean([d['W'] for d in model_perfs], axis=0)
            mean_b = np.mean([d['b'] for d in model_perfs])

            fold_perfs.append({'r2':mean_r2, 'W':mean_W, 'b':mean_b})

        fold_perfs.sort(key=operator.itemgetter('r2'), reverse=True)
        best_model = fold_perfs[0]

        return best_model

    def predict(self, X):
        pass

