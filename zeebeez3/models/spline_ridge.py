from copy import deepcopy
import operator

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge


class StagewiseSplineRidgeRegression(object):
    """
    This predictor does regression using a nonlinear spline basis, and adds features in a stagewise way only if
    they improve cross validation performance.
    """

    def __init__(self):
        pass

    def fit(self, X, y, baseline_features=list(), cv_indices=None, num_hyperparams=50, verbose=True, feature_names=None):
        # Stepwise forward regression using all baseline_features and then adding additional features one at the time
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
            best_r2 = 0.0
        else:
            Xsub = X[:, best_features]
            Xsub = self.spline_basis(Xsub)
            m_baseline = self.cv_fit(Xsub, y, cv_indices, num_hyperparams, verbose=verbose)
            best_r2 = max(m_baseline['r2'], 0)
        
        if verbose:
            print('Baseline model performance with %d features: R2=%0.2f' % (len(best_features), best_r2))

        iter = 0
        feature_improvements = [0.]*len(best_features)
        while len(features_left) > 0:
            if verbose:
                print('Round %d of stagewise testing. Good features: %s' % (iter, ','.join([feature_names[k].decode('UTF-8') for k in best_features])))
                print('\tFeatures left: %s' % ','.join([feature_names[k].decode('UTF-8') for k in features_left]))
            best_feature = None
            best_feature_r2 = best_r2

            bad_features = list() # features that do not lead to an improvement in model performance
            for k in features_left:
                fi = deepcopy(best_features)
                fi.append(k)
                Xsub = X[:, fi]
                Xsp = self.spline_basis(Xsub)
                m_feature = self.cv_fit(Xsp, y, cv_indices, num_hyperparams, verbose=verbose)
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
            print('Final model features: R2=%0.2f+-%.3f, features=%s' % (m_final['r2'], m_final['r2_std'], ','.join([feature_names[k].decode('UTF-8') for k in best_features])))
        m_final['features'] = best_features
        m_final['feature_improvements'] = np.array(feature_improvements)

        return m_final
    
    def fit_added_value(self, X, y, baseline_features=list(), cv_indices=None, num_hyperparams=50, verbose=True, feature_names=None):
        # Fits baseline features and then other features on residual to get added value plots
        nsamps,nfeatures = X.shape
        assert len(y) == nsamps

        if cv_indices is None:
            cv_indices = list(KFold(nsamps, n_folds=10))

        if feature_names is None:
            feature_names = ['%d' % k for k in range(nfeatures)]

        all_features = list(range(nfeatures))
        features_left = list(np.setdiff1d(all_features, baseline_features))
        
        feature_names_baseline = [feature_names[k] for k in baseline_features]
        feature_names_added = [feature_names[k] for k in features_left]
        

        # run the baseline model
        if len(baseline_features) == 0:
            m_baseline = {'r2': 0, 'r2_std':0, 'W': 0, 'b': 0, 'alpha': 0}
            yhat = np.zeros(y.shape)
        else:
            Xsub = X[:, baseline_features]
            Xsub = self.spline_basis(Xsub)
            m_baseline = self.cv_fit(Xsub, y, cv_indices, num_hyperparams, verbose=verbose)
            yhat = np.dot(Xsub, m_baseline['W']) + m_baseline['b']
            
        m_baseline['name'] = 'Baseline'
        m_baseline['features'] = feature_names_baseline
        m_baseline['predict'] = yhat
        
        if verbose:
            print('Baseline model performance with %d features: R2=%0.2f +- %0.3f' % (len(baseline_features), m_baseline['r2'], m_baseline['r2_std']))
 
        # Keep residual if r2 significant
        if ( m_baseline['r2'] > 2.0*m_baseline['r2_std'] ):
            yres = y - yhat
        else:
            yres = y
        
        # Test the additional performance one feature at the time.
        
        # First predict features from all the other features
        m_Xs = list()
        Xres = np.zeros((nsamps, len(features_left)))       
        for ik, k in enumerate(features_left):

            Xadd = X[:, k]
          
            # Find residual ffor X
            if len(baseline_features) == 0:
                m_X = {'r2': 0, 'r2_std':0, 'W': 0, 'b': 0, 'alpha': 0}
                Xhat = np.zeros(Xadd.shape)
            else:
                m_X = self.cv_fit(Xsub, Xadd, cv_indices, num_hyperparams, verbose=verbose)
                Xhat = np.dot(Xsub, m_X['W']) + m_X['b']
            
            if verbose:
                print('Baseline predicting %d: R2=%0.2f +- %0.3f' % (k, m_X['r2'], m_X['r2_std']))

            if (m_X['r2'] > 2.0*m_X['r2_std']) :
                Xres[:, ik] = Xadd - Xhat
            else:
                Xres[:, ik] = Xadd
            m_X['name'] = 'Feature %d' % k
            m_X['features'] = feature_names_baseline
            m_X['predict'] = Xhat
            m_Xs.append(m_X)
        
        # Now predict featuers left.             
        Xsp = self.spline_basis(Xres) 
        m_feature = self.cv_fit(Xsp, yres, cv_indices, num_hyperparams, verbose=verbose)
        yhat = np.dot(Xsp, m_feature['W']) + m_feature['b']
            
        m_feature['name'] = 'Added'
        m_feature['features'] = feature_names_added
        m_feature['predict'] = yhat
        
        if verbose:
            if ( m_feature['r2'] > 2.0*m_feature['r2_std'] ):
                print('Significant Added Value for %s: R2 = %.2f +- %.3f' % (features_left[0], m_feature['r2'], m_feature['r2_std']) )
            else:
                print('Not Significant Added Value for %s: R2 = %.2f +- %.3f' % (features_left[0], m_feature['r2'], m_feature['r2_std']) )
                        
        # Return model outputs
        return m_baseline, m_Xs, m_feature

    def fit_nested(self, X, y, del_features=list(), cv_indices=None, num_hyperparams=50, verbose=True, feature_names=None):
        # Fits baseline features and then single features on residual to get added value plots
        nsamps,nfeatures = X.shape
        assert len(y) == nsamps

        if cv_indices is None:
            cv_indices = list(KFold(nsamps, n_folds=10))

        if feature_names is None:
            feature_names = ['%d' % k for k in range(nfeatures)]
            
        feature_names_left = []

        all_features = list(range(nfeatures))
        features_left = list(np.setdiff1d(all_features, del_features))
        feature_names_left = [feature_names[k] for k in features_left]

        # run the full model
        if len(all_features) == 0:
            m_full = {  'name': 'Full',   'r2': 0, 'r2_std': 0, 'W': 0, 'b': 0, 'alpha': 0, 'features': feature_names, 'predict': np.zeros(y.shape)}
            m_nested = {'name': 'Nested', 'r2': 0, 'r2_std': 0, 'W': 0, 'b': 0, 'alpha': 0, 'features': feature_names, 'predict': np.zeros(y.shape)}
        else:
            Xsub = self.spline_basis(X)
            m_full = self.cv_fit(Xsub, y, cv_indices, num_hyperparams, verbose=verbose)
            yhat = np.dot(Xsub, m_full['W']) + m_full['b']
            m_full['name'] = 'Full'
            m_full['features'] = feature_names
            m_full['predict'] = yhat
            
            if len(features_left) == 0:
                m_nested = {'name': 'Nested', 'r2': 0, 'r2_std': 0, 'W': 0, 'b': 0, 'alpha': 0, 'features':feature_names_left, 'predict': np.zeros(y.shape)}
            else:
                Xsub = X[:, features_left]
                Xsub = self.spline_basis(Xsub)
                m_nested = self.cv_fit(Xsub, y, cv_indices, num_hyperparams, verbose=verbose)
                yhat = np.dot(Xsub, m_nested['W']) + m_nested['b']
                m_nested['name'] = 'Nested'
                m_nested['features'] = feature_names_left
                m_nested['predict'] = yhat
                            
        # Return fit results from both models
        if verbose:
            print('Full model:', feature_names)
            print('\tR2 = %.2f +- %.3f' % (m_full['r2'], m_full['r2_std']))
            print('Nested model:', feature_names_left)
            print('\tR2 = %.2f +- %.3f' % (m_nested['r2'], m_nested['r2_std']))
            stdcomp = np.sqrt((m_full['r2_std']**2 + m_nested['r2_std']**2)/2)
            if ((m_full['r2'] - m_nested['r2']) > 2.0*stdcomp):
                print('Significant Diff %.3f > %.3f ' % ((m_full['r2'] - m_nested['r2']),2.0*stdcomp ))
            else:
                print('NOT Significant Diff %.3f < %.3f' % ((m_full['r2'] - m_nested['r2']),2.0*stdcomp ))
            
        return m_full, m_nested
                


    def spline_basis(self, X):

        if (len(X.shape) == 1):
            nfeatures = 1
            X = X.reshape((X.shape[0],1))
        else:
            nfeatures = X.shape[1]
            
        dof = 6
        Xsp = np.zeros([X.shape[0], nfeatures*dof])

        for k in range(nfeatures):
            si = k*dof
            ei = si + dof
            Xsp[:, si:ei] = cubic_spline_basis(X[:, k], num_knots=3)

        return Xsp

    def cv_fit(self, X, y, cv_indices, num_hyperparams, verbose=False):

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

# FET: I use the normalization in sklearn for numerical reasons but keeps X values with regular units
                rr = Ridge(alpha=alpha, fit_intercept=True, normalize=True, solver='svd')
                rr.fit(Xtrain, ytrain)

                ypred = rr.predict(Xtest)

                sst = np.sum((ytest - ytrain.mean())**2)
                sse = np.sum((ytest - ypred)**2)
                r2 = 1. - (sse / sst)

                model_perfs.append({'r2':r2, 'W':rr.coef_, 'b':rr.intercept_})

            mean_r2 = np.mean([d['r2'] for d in model_perfs])
            std_r2 = np.std([d['r2'] for d in model_perfs], ddof=1)
            mean_W = np.mean([d['W'] for d in model_perfs], axis=0)
            mean_b = np.mean([d['b'] for d in model_perfs])

            fold_perfs.append({'r2':mean_r2, 'r2_std': std_r2, 'W':mean_W, 'b':mean_b, 'alpha':alpha})

        fold_perfs.sort(key=operator.itemgetter('r2'), reverse=True)
        best_model = fold_perfs[0]
        if verbose:
            print('cv_fit: Best model found for alpha %f' % best_model['alpha'])

        return best_model

    def predict(self, X):
        pass


def cubic_spline_basis(x, num_knots=3, return_knots=False, knots=None):

    if knots is None:
        p = 100. / (num_knots + 1)
        knots = np.array([np.percentile(x, int((k + 1) * p)) for k in range(num_knots)])
        assert knots.min() >= x.min()
        assert knots.max() <= x.max()
        if len(np.unique(knots)) != len(knots):
            # print '[cubic_spline_basis] number of unique kernels is less than the degrees of freedom, trying wider knot spacing (q10, q50, q90)'
            knots = [np.percentile(x, 10), np.percentile(x, 50), np.percentile(x, 90)]
        assert len(np.unique(knots)) == len(knots), '# of unique kernels is less than the degrees of freedom!'

    num_knots = len(knots)
    df = num_knots+3
    B = np.zeros([len(x), df])
    for k in range(3):
        B[:, k] = x**(k+1)

    for k in range(num_knots):
        i = x > knots[k]
        B[i, k+3] = (x[i]-knots[k])**3

    if return_knots:
        return B,knots
    return B


def natural_spline_basis(x, num_knots=3):
    p = 100. / (num_knots + 1)
    knots = np.array([np.percentile(x, int((k + 1) * p)) for k in range(num_knots)])
    assert knots.min() >= x.min()
    assert knots.max() <= x.max()
    assert len(np.unique(knots)) == len(knots), '# of unique kernels is less than the degrees of freedom!'

    df = num_knots
    B = np.zeros([len(x), df])
    B[:, 0] = x

    def _dk(_k):
        _i1 = x > knots[_k]
        _i2 = x > knots[-1]
        _x1 = np.zeros([len(x)])
        _x2 = np.zeros([len(x)])

        _x1[_i1] = x - knots[_k]
        _x2[_i2] = x - knots[-1]

        _num = (_x1**3 - _x2**3)
        _denom = (knots[-1] - knots[_k])
        assert abs(_denom) > 0, "denom=0, _k=%d, _i1.sum()=%d, _i2.sum()=%d" % (_k, _i1.sum(), _i2.sum())

        return _num / _denom

    for k in range(num_knots-2):
        B[:, k + 1] = _dk(k) - _dk(num_knots-2)

    return B