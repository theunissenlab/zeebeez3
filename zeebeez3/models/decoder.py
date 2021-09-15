import sys
import operator

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR

from keras.layers import Dense, Activation, Dropout, K
from keras.models import Sequential
from keras.regularizers import l2, Regularizer


class Decoder(object):

    def __init__(self, rseed=1234567):
        self.X = None
        self.Y = None
        self.S = None
        self.y = None

        self.l2_lambda = None
        self.Xmean = None
        self.Xstd = None

        self.nfolds = None

        self.integer2type = None
        self.integer2bird = None
        self.integer2prop = None

        self.index2electrode = None
        self.cell_index2electrode = None

        self.preproc_file = None

        self.rseed = rseed

        self.model_type = 'linear'

        self.freqs = None
        self.lags = None

    def read_preproc_file(self, preproc_file, verbose=False):
        # read the preproc file
        if verbose:
            print('Reading preproc file: %s' % preproc_file)
        hf = h5py.File(preproc_file, 'r')
        self.X = np.array(hf['X'])
        self.Y = np.array(hf['Y'])
        self.S = np.array(hf['S'])
        self.integer2type = list(hf.attrs['integer2type'])
        self.integer2bird = list(hf.attrs['integer2bird'])
        self.integer2prop = list(hf.attrs['integer2prop'])

        self.index2electrode = hf.attrs['index2electrode']
        self.cell_index2electrode = hf.attrs['cell_index2electrode']

        self.freqs = hf.attrs['freqs']
        self.lags = hf.attrs['lags']

        hf.close()
        self.preproc_file = preproc_file

        if self.X.shape[1] == 0:
            self.X = self.X.reshape([len(self.X), 1])
        if verbose:
            print('X.shape=',self.X.shape)
            print('Y.shape=', self.Y.shape)
            print('S.shape=', self.S.shape)

    def bootstrap(self, nsamps, max_attemps=2000, verbose=False, subset_index=None):
        """ Create nfolds training and test sets by holding out entire birds """

        if subset_index is None:
            stim_types = self.Y[:, 0]
            stim_emitters = self.Y[:, 3]
        else:
            stim_types = self.Y[subset_index, 0]
            stim_emitters = self.Y[subset_index, 3]

        num_attempts = 0

        # get call types per emitter
        calls_per_emitter = dict()
        for e in np.unique(stim_emitters):
            calls_per_emitter[e] = dict()
            for st in np.unique(stim_types):
                i = (stim_types == st) & (stim_emitters == e)
                calls_per_emitter[e][st] = i.sum()

        # identify juveniles vs adults
        juveniles = list()
        adults = list()
        for e,cpe in list(calls_per_emitter.items()):
            # skip "unknown" birds
            if e == self.integer2bird.index(b'unknown'):
                continue

            lt_i = self.integer2type.index(b'LT')  # only juveniles emit long tonal calls
            beg_i = self.integer2type.index(b'Be')  # only juveniles emit begging calls
            jcall_sum = cpe[lt_i] + cpe[beg_i]
            if jcall_sum > 0:
                juveniles.append(e)
            else:
                adults.append(e)

        # keep track of holdout set members so there aren't duplicates
        holdout_sets_used = dict()
        bad_sets = dict()

        training_sets = list()
        holdout_sets = list()

        while len(training_sets) < nsamps and num_attempts < max_attemps:

            num_attempts += 1

            # choose two juveniles at random for holdout set
            np.random.shuffle(juveniles)
            j1,j2 = juveniles[:2]

            # choose two adults at random for holdout set
            np.random.shuffle(adults)
            a1,a2 = adults[:2]

            # make a key to uniquely identify the combination of juveniles and adults that comprise the holdout set
            holdout_key = [self.integer2bird[j1], self.integer2bird[j2], self.integer2bird[a1], self.integer2bird[a2]]
            holdout_key.sort()
            holdout_key = tuple(holdout_key)

            if holdout_key in holdout_sets_used:
                if verbose:
                    print('Holdout set already used, trying again! %s' % str(holdout_key))
                continue

            if holdout_key in bad_sets:
                # print 'Bad holdout set(1), trying again! %s' % str(holdout_key)
                continue

            # determine if holdout set contains at least one example of each type, excluding song
            for st in np.unique(stim_types):
                if st == self.integer2type.index(b'song'):
                    continue

                tc = 0
                tc += calls_per_emitter[j1][st] + calls_per_emitter[j2][st]
                tc += calls_per_emitter[a1][st] + calls_per_emitter[a2][st]

                if tc == 0:
                    bad_sets[holdout_key] = True

            if holdout_key in bad_sets:
                if verbose:
                    print('Bad holdout set(2), trying again!')
                continue

            holdout_emitters = [j1, j2, a1, a2]
            holdout_set = list()
            training_set = list()
            # add the stim indices for the holdout birds
            for e in holdout_emitters:
                assert self.integer2bird[e] != b'unknown', "Something went wrong, unknown birds in holdout set!"
                i = stim_emitters == e
                ii = np.where(i)[0]
                holdout_set.extend(ii)

            # add stim indices for non holdout birds
            training_emitters = np.setdiff1d(np.unique(stim_emitters), holdout_emitters)
            for e in training_emitters:
                assert e not in holdout_emitters, "Something went wrong, holdout birds in training set!"
                if self.integer2bird[e] == b'unknown':
                    continue

                i = stim_emitters == e
                ii = np.where(i)[0]
                training_set.extend(ii)

            # now randomly hold out 25% of songs from "unknown" emitter
            i = stim_emitters == self.integer2bird.index(b'unknown')
            ii = np.where(i)[0]
            np.random.shuffle(ii)
            ntest = int(0.25*len(ii))
            holdout_set.extend(ii[:ntest])
            training_set.extend(ii[ntest:])

            # now check to make sure the training and holdout sets contain examples from each class
            training_classes = [stim_types[k] for k in training_set]
            holdout_classes = [stim_types[k] for k in holdout_set]
            if len(np.unique(training_classes)) != len(np.unique(stim_types)):
                if verbose:
                    print('Training set is missing a stim type, trying again!')
                bad_sets[holdout_key] = True
                continue

            if len(np.unique(holdout_classes)) != len(np.unique(stim_types)):
                if verbose:
                    print('Holdout set is missing a stim type, trying again!')
                bad_sets[holdout_key] = True
                continue

            # we're good to use this holdout set, make an entry in the dictionary that keeps track of them
            holdout_sets_used[holdout_key] = True

            # now we're done with this bootstrapped sample
            training_sets.append(training_set)
            holdout_sets.append(holdout_set)

        return training_sets, holdout_sets

    def check_data_matrix(self, X, border=None):
        num_samps,num_features = X.shape

        # compute the correlation coefficient between all pairs of data points
        Csamp = np.zeros([num_samps, num_samps])
        print('')
        for k in range(num_samps):
            Csamp[k, k] = 1.0
            for j in range(k):
                a = X[k, :]
                b = X[j, :]

                cov = np.dot(a-a.mean(), b-b.mean()) / num_features
                cc = cov / (a.std(ddof=1)*b.std(ddof=1))

                Csamp[k, j] = cc
                Csamp[j, k] = cc

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(X, interpolation='nearest', aspect='auto', cmap=plt.cm.jet, origin='upper')
        plt.colorbar()
        plt.title('Data Matrix')

        plt.subplot(1, 2, 2)
        plt.imshow(Csamp, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='lower', vmin=-1, vmax=1)
        plt.colorbar()
        if border is not None:
            plt.axhline(border-0.5, c='w', linewidth=3.0)
            plt.axvline(border-0.5, c='w', linewidth=3.0)
        plt.show()

    def print_stim_table(self):
        # pick out the categorical dependent variable of choice
        stim_types = self.Y[:, 0]
        stim_emitters = self.Y[:, 3]
        # print out some information about the call types and emitters
        print('Stimulus count by type/emitter:')
        print('')
        utypes = [self.integer2type[k] for k in np.unique(stim_types)]
        uemitters = [self.integer2bird[k] for k in np.unique(stim_emitters)]
        sys.stdout.write('          \t')
        for ut in utypes:
            if ut == 'song':
                ut = 'So'
            sys.stdout.write('%s\t' % ut)
        sys.stdout.write('Total')
        sys.stdout.write('\n')
        for e in uemitters:
            if e == 'unknown':
                sys.stdout.write('%s   \t' % e)
            else:
                sys.stdout.write('%s\t' % e)
            for ut in utypes:
                i = (stim_types == self.integer2type.index(ut)) & (stim_emitters == self.integer2bird.index(e))
                sys.stdout.write('%d\t' % i.sum())
            i = stim_emitters == self.integer2bird.index(e)
            sys.stdout.write('%d' % i.sum())
            sys.stdout.write('\n')

        sys.stdout.write('Total   \t')
        for ut in utypes:
            i = stim_types == self.integer2type.index(ut)
            sys.stdout.write('%d\t' % i.sum())
        sys.stdout.write('%d' % len(stim_types))
        sys.stdout.write('\n')

        print('')
        print('--------------------------')

    def compute_perfs(self, y, ypred, ymean):
        # compute correlation coefficient of prediction on test set
        cc = np.corrcoef(y, ypred)[0, 1]

        # compute r2 (likelihood ratio) of prediction on test set
        ss_res = np.sum((y - ypred) ** 2)
        ss_tot = np.sum((y - ymean) ** 2)
        r2 = 1. - (ss_res / ss_tot)

        # compute the RMSE
        rmse = np.sqrt(ss_res.mean())

        return {'cc': cc, 'r2': r2, 'rmse': rmse, 'likelihood': ss_res, 'likelihood_null': ss_tot}

    def bootstrap_fit(self, X, y, nfolds=25, verbose=False):

        # set random seed to consistent value across runs, prior to forming training and validation sets
        np.random.seed(self.rseed)

        # construct training and validation sets using the bootstrap
        training_sets_by_fold, test_sets_by_fold = self.bootstrap(nfolds)

        # keep track of the model performance for each set of model parameters
        model_perfs = list()

        ntrain = list()
        ntest = list()
        for train_i, test_i in zip(training_sets_by_fold, test_sets_by_fold):
            ntrain.append(len(train_i))
            ntest.append(len(test_i))
        ntrain = np.array(ntrain)
        ntest = np.array(ntest)
        # print 'Average # of training samples: %f +/- %f' % (ntrain.mean(), ntrain.std(ddof=1))
        # print 'Average # of test samples: %f +/- %f' % (ntest.mean(), ntest.std(ddof=1))

        nfolds = len(training_sets_by_fold)

        # initialize a model class
        model_types = {'linear': LinearModel, 'nn': FeedforwardNetModel, 'rf':RandomForestModel, 'svr':SVRModel}
        assert self.model_type in model_types, "Invalid model type: %s" % self.model_type
        model_class = model_types[self.model_type]

        hyperparams = model_class.get_hyperparams()

        # fit a distribution of models using bootstrap for each model parameter set in model_param_sets
        for hyperparam in hyperparams:

            if verbose:
                print('Hyperparams:', hyperparam)
            models = list()
            fold_data = {'ntrain': list(), 'ntest': list(), 'fold': list(),
                         'cc': list(), 'r2': list(), 'rmse': list(),
                         'likelihood': list(), 'likelihood_null': list()}

            # fit a model to each bootstrapped training set
            for k in range(nfolds):

                # grab the indices of training and validation sets
                train_indices = training_sets_by_fold[k]
                test_indices = test_sets_by_fold[k]

                assert len(np.intersect1d(train_indices, test_indices)) == 0, "Training and test sets overlap!"

                # get the training and validation matrices
                Xtrain = X[train_indices, :]
                ytrain = y[train_indices]

                Xtest = X[test_indices, :]
                ytest = y[test_indices]

                ntrain = len(ytrain)
                ntest = len(ytest)

                # construct a model and fit the data
                model = model_class(hyperparam)
                model.fit(Xtrain, ytrain)

                # make a prediction on the test set
                ypred = model.predict(Xtest)

                fold_data['ntrain'].append(ntrain)
                fold_data['ntest'].append(ntest)
                fold_data['fold'].append(k)

                d = self.compute_perfs(ytest, ypred, ytrain.mean())
                for key, val in list(d.items()):
                    fold_data[key].append(val)

                models.append(model)

                if verbose:
                    print('\tFold %d: ntrain=%d, ntest=%d, cc=%0.2f, r2=%0.3f, rmse=%0.3f, likelihood=%0.3f' % \
                          (k, ntrain, ntest, d['cc'], d['r2'], d['rmse'], d['likelihood']))

            # compute the average performances
            fold_df = pd.DataFrame(fold_data)

            mean_r2 = fold_df['r2'].mean()
            stderr_r2 = fold_df['r2'].std(ddof=1) / np.sqrt(nfolds)

            # save the model info
            model_perfs.append({'hyperparam': hyperparam, 'models': models, 'fold_df': fold_df,
                                'mean_r2': mean_r2, 'stderr_r2': stderr_r2})

        # identify the best model parameter set, i.e. the one with the highest R2
        model_perfs.sort(key=operator.itemgetter('mean_r2'), reverse=True)

        best_model_dict = model_perfs[0]
        best_fold_df = best_model_dict['fold_df']
        best_hyperparam = best_model_dict['hyperparam']

        if verbose:
            print('Best Model Params:', best_model_dict['hyperparam'])
            for pname in ['cc', 'r2', 'rmse', 'likelihood']:
                print('\t%s=%0.3f +/- %0.3f' % (pname, best_fold_df[pname].mean(), best_fold_df[pname].std(ddof=1)))

        lk_full = best_fold_df.likelihood.values
        lk_null = best_fold_df.likelihood_null.values
        cv_r2 = best_fold_df.r2.mean()
        Wcv = np.array([m.get_weights() for m in best_model_dict['models']])
        if self.model_type == 'linear':
            W = Wcv.mean(axis=0)
        else:
            W = Wcv

        rdict = {'W': W, 'lk_full': lk_full, 'lk_null': lk_null, 'hyperparam': best_hyperparam, 'r2': cv_r2}

        return rdict

    def zscore_neural_data(self):
        # deal with infs and NaNs
        self.S[np.isinf(self.S)] = 0.
        self.X[np.isinf(self.X)] = 0.
        self.S[np.isnan(self.S)] = 0.
        self.X[np.isnan(self.X)] = 0.

        # zscore the neural data
        self.X -= self.X.mean(axis=0)
        self.X /= self.X.std(axis=0, ddof=1)

        # deal with infs and nans (again)
        self.X[np.isinf(self.X)] = 0.
        self.X[np.isnan(self.X)] = 0.

    def preprocess_acoustic_features(self):

        def _normalize_feature(_k):
            _aprop = self.integer2prop[_k]
            if _aprop in ['minfund', 'maxfund', 'fund', 'fund2']:
                # normalize fundamental features, for values where fundamental can't be estimated, replace with zero
                _nz = self.S[:, _k] > 0.
                self.S[_nz, _k] -= self.S[_nz, _k].mean()
                self.S[_nz, _k] /= self.S[_nz, _k].max()
                self.S[~_nz] = 0.
            else:
                # zscore non-fundamental features
                self.S[:, _k] -= self.S[:, _k].mean()
                self.S[:, _k] /= self.S[:, _k].std(ddof=1)

        # preprocess the acoustic features by normalizing them
        for k, aprop in enumerate(self.integer2prop):
            _normalize_feature(k)


class LinearModel(object):

    def __init__(self, lambda2):
        self.lambda2 = lambda2
        self.rr = Ridge(alpha=self.lambda2)
        # self.rr = LinearRegression()

    @classmethod
    def get_hyperparams(cls):
        hparams = list(np.logspace(-2, 6, 50))
        hparams.insert(0, 0)
        return hparams

    def fit(self, X, y):
        self.rr.fit(X, y)

    def predict(self, X):
        return self.rr.predict(X)

    def get_weights(self):
        return self.rr.coef_


class CovarianceActivityRegularizer(Regularizer):

    def __init__(self, C=1.0):
        Regularizer.__init__(self)
        self.C = K.cast_to_floatx(C)
        self.uses_learning_phase = True

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        output = self.layer.output

        regularized_loss = loss + self.l1 * K.sum(K.mean(K.abs(output), axis=0))
        regularized_loss += self.l2 * K.sum(K.mean(K.square(output), axis=0))

        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c,
                'l2': self.l2}


class FeedforwardNetModel(object):

    def __init__(self, params):
        self.lambda2 = params['lambda2']
        self.num_hidden1 = params['nhidden']
        self.activation = 'sigmoid'
        self.dropout = False
        self.nn = None

    @classmethod
    def get_hyperparams(cls):
        l2 = [0.]
        nhidden = [10, 100]
        params = list()
        for nh1 in nhidden:
            for lval in l2:
                params.append({'lambda2':lval, 'nhidden':nh1})
        return params

    def fit(self, X, y):
        num_samps,num_inputs = X.shape
        assert len(y) == num_samps

        batch_size = num_samps

        self.nn = Sequential()

        # add hidden layers
        nhidden = [self.num_hidden1]
        for k,nh in enumerate(nhidden):
            idim = None
            if k == 0:
                idim = num_inputs
            self.nn.add(Dense(nh, input_dim=idim, init='glorot_uniform', W_regularizer=l2(self.lambda2)))
            self.nn.add(Activation(self.activation))
            if self.dropout:
                self.nn.add(Dropout(0.5))

        # output layer
        self.nn.add(Dense(1, init='uniform', W_regularizer=l2(self.lambda2)))
        self.nn.compile(loss='mean_squared_error', optimizer='adam')
        self.nn.fit(X, y, nb_epoch=50, batch_size=batch_size, verbose=False)

    def predict(self, X):
        batch_size = X.shape[0]
        ypred = self.nn.predict(X, batch_size=batch_size)
        return ypred.squeeze()

    def get_weights(self):
        return np.array([0., 0])


class RandomForestModel(object):

    def __init__(self, params):
        self.ntrees = params['ntrees']
        self.depth = params['depth']
        self.rf = None

    @classmethod
    def get_hyperparams(cls):
        ntrees = [5, 10, 25, 50, 100]
        depth = [2, 5, 10]

        hparams = list()
        for nt in ntrees:
            for d in depth:
                hparams.append({'ntrees':nt, 'depth':d})
        return hparams

    def fit(self, X, y):
        num_samps, num_inputs = X.shape
        assert len(y) == num_samps

        self.rf = RandomForestRegressor(n_estimators=self.ntrees, max_depth=self.depth)
        self.rf.fit(X, y)

    def predict(self, X):
        return self.rf.predict(X)

    def get_weights(self):
        return np.array([0, 0])


class SVRModel(object):

    def __init__(self, params):
        self.C = 1.
        self.eps = params['eps']
        self.gamma = params['gamma']
        self.svr = None

    @classmethod
    def get_hyperparams(cls):
        eps = [0.1, 1.0, 10.]
        gamma = list(np.logspace(-3, 1, 10))
        gamma.insert(0, "auto")
        hparams = list()
        for e in eps:
            for g in gamma:
                hparams.append({'eps':e, 'gamma':g})
        return hparams

    def fit(self, X, y):
        num_samps, num_inputs = X.shape
        assert len(y) == num_samps

        self.svr = SVR(C=self.C, epsilon=self.eps, cache_size=1000, gamma=self.gamma)
        self.svr.fit(X, y)

    def predict(self, X):
        return self.svr.predict(X)

    def get_weights(self):
        return np.array([0, 0])

