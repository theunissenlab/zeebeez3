import os
from copy import deepcopy

import h5py
import numpy as np
import matplotlib.pyplot as plt

from soundsig.basis import cubic_spline_basis
from soundsig.plots import compute_mean_from_scatter

from zeebeez3.core.utils import ACOUSTIC_FUND_PROPS, USED_ACOUSTIC_PROPS
from zeebeez3.models.decoder import Decoder
from zeebeez3.models.pard import PARD
from zeebeez3.models.spline_ridge import StagewiseSplineRidgeRegression


class AcousticEncoderDecoder(Decoder):

    def __init__(self):
        Decoder.__init__(self)

        self.encoder_perfs = None
        self.good_encoders = None
        self.encoder_weights = None

        self.encoder_feature_improvements = None
        self.encoder_features = None

        self.decoder_perfs_ind = None
        self.decoder_weights_ind = None

        self.decoder_perfs = None
        self.good_decoders = None
        self.decoder_weights = None

    def fit(self, preproc_file, model_type='linear',
            encoder=True, decoder=True,
            individual_decoder_weights=False,
            encoder_acoustic_props=None,
            zscore_response=False):

        self.read_preproc_file(preproc_file)
        self.model_type = model_type

        nsamps, nfeatures_neural = self.X.shape
        nfeatures_stim = self.S.shape[1]

        assert self.Y.shape[0] == nsamps
        
        # This method does not exist?
        self.zscore_neural_data()

        self.good_encoders = list()
        self.good_decoders = list()

        cv_indices = list(zip(*self.bootstrap(25)))

        if encoder:
            base_features = [self.integer2prop.index(b'maxAmp'), self.integer2prop.index(b'meanspect'),
                             self.integer2prop.index(b'sal')]

            # Z-score the acoustic features
            S = self.preprocess_acoustic_features(acoustic_props=encoder_acoustic_props)
            
            # run an encoder for each neural feature, which could be the spike rate of a neuron or the LFP power
            # at a given frequency
            for k in range(nfeatures_neural):

                y = deepcopy(self.X[:, k])

                if zscore_response:
                    y -= y.mean()
                    y /= y.std(ddof=1)
                    y[np.isnan(y)] = 0.
                    y[np.isinf(y)] = 0.

                # print 'y: # of nans=%d, # of infs=%d, min=%f, max=%f' % (np.sum(np.isnan(y)), np.sum(np.isinf(y)), y.min(), y.max())
                self.model_type = 'linear'
                sr = StagewiseSplineRidgeRegression()
                edict = sr.fit(S, y, baseline_features=base_features, cv_indices=cv_indices, verbose=False,
                               feature_names=self.integer2prop)

                if edict is None:
                    print('\tFeature %d is not predictable!' % k)
                else:
                    bf = [self.integer2prop[f].decode('UTF-8') for f in edict['features']]
                    print('\tFeature %d: best_props=%s, R2=%0.2f' % (k, ','.join(bf), edict['r2']))
                    self.good_encoders.append((k, edict))

        if decoder:
            # run a PARD decoder, the power across electrodes and frequency bands
            # is used to predict acoustic features

            S = self.preprocess_acoustic_features()
            for k in range(nfeatures_stim):

                y = S[:, k]

                print('Training neural->acoustic decoder on acoustic feature %s' % self.integer2prop[k])

                self.model_type = 'linear'
                pard = PARD()
                ddict = pard.fit(self.X, y, self.bootstrap_fit, plot=False, zscore=False,
                                 individual_feature_weights=individual_decoder_weights)

                if ddict is None:
                    print('\tPair is not predictable!')
                else:
                    print('\tFinal encoder performance: %0.2f' % ddict['r2'])
                    self.good_decoders.append((k, ddict))

    def preprocess_acoustic_features(self, acoustic_props=None):
        if acoustic_props is None:
            acoustic_props = self.integer2prop

        def _normalize_feature(_k, _S):
            _aprop = acoustic_props[_k]
            if _aprop in ACOUSTIC_FUND_PROPS:
                # normalize fundamental features, for values where fundamental can't be estimated, replace with zero
                _nz = _S[:, _k] > 0.
                _S[_nz, _k] -= _S[_nz, _k].mean()
                _S[_nz, _k] /= _S[_nz, _k].max()
                _S[~_nz] = 0.
            else:
                # zscore non-fundamental features
                _S[:, _k] -= _S[:, _k].mean()
                _S[:, _k] /= _S[:, _k].std(ddof=1)

        # preprocess the acoustic features by normalizing them
        S = np.zeros([self.S.shape[0], len(acoustic_props)])
        for k, aprop in enumerate(acoustic_props):
            i = self.integer2prop.index(aprop)
            S[:, k] = self.S[:, i]
            _normalize_feature(k, S)

        return S

    def compute_tuning_curves(self):
        """
            Should be called after a file is loaded using self.load(...).
            """

        # read preproc file
        self.read_preproc_file(self.preproc_file)

        num_bins = 20
        # num_bins = self.S.shape[0]
        nprops = len(self.integer2prop)

        tuning_curves_x = np.zeros([len(self.good_encoders), nprops, num_bins])
        tuning_curves = np.zeros([len(self.good_encoders), nprops, num_bins])
        # print 'tuning_curves_x.shape=',tuning_curves_x.shape

        for k, neural_feature_index in enumerate(self.good_encoders):
            # get encoder properties
            r2 = self.encoder_perfs[k]
            acoustic_feature_indices = self.encoder_features[k]
            acoustic_feature_names = [self.integer2prop[j] for j in acoustic_feature_indices]
            w = self.encoder_weights[k]

            # get neural response and zscore it
            y = self.X[:, neural_feature_index]
            y -= y.mean()
            y /= y.std(ddof=1)

            # compute the tuning curve for each neuron
            # z-score acoustic features
            S = self.preprocess_acoustic_features(acoustic_props=acoustic_feature_names)
            assert S.shape[1] == len(acoustic_feature_indices)
            for j, acoustic_feature_index in enumerate(acoustic_feature_indices):
                si = j * 6
                ei = si + 6

                # get the acoustic feature values
                x = S[:, j]

                # project the acoustic feature into a cubic spline basis, saving the knots
                B, knots = cubic_spline_basis(x, return_knots=True)

                # construct a regular interval on x that excludes extrema
                q5 = np.percentile(x, 5)
                q95 = np.percentile(x, 95)
                xrs = np.linspace(q5, q95, num_bins)

                # compute the cubic spline basis on the regular interval
                B = cubic_spline_basis(xrs, knots=knots)

                # get the encoder weights for this acoustic feature
                wj = w[si:ei]

                # get the prediction of the neural response for the regular interval
                yhat = np.dot(B, wj)

                # un-zscore the acoustic feature
                xraw = (xrs * self.S[:, acoustic_feature_index].std(ddof=1)) + self.S[:, acoustic_feature_index].mean()

                tuning_curves_x[k, acoustic_feature_index, :] = xraw
                tuning_curves[k, acoustic_feature_index, :] = yhat

        return tuning_curves_x, tuning_curves

    def plot_tuning_curves(self, acoustic_prop=None):
        """
        Should be called after a file is loaded using self.load(...).
        """

        # read preproc file
        self.read_preproc_file(self.preproc_file)

        for k, neural_feature_index in enumerate(self.good_encoders):
            # get encoder properties
            r2 = self.encoder_perfs[k]
            acoustic_feature_indices = self.encoder_features[k]
            acoustic_feature_names = [self.integer2prop[j] for j in acoustic_feature_indices]
            w = self.encoder_weights[k]

            # get neural response and zscore it
            y = self.X[:, neural_feature_index]
            y -= y.mean()
            y /= y.std(ddof=1)

            # compute the tuning curve for each neuron
            S = self.preprocess_acoustic_features(acoustic_props=acoustic_feature_names)
            for j, acoustic_feature_index in enumerate(acoustic_feature_indices):
                si = j * 6
                ei = si + 6

                # get the acoustic feature values
                x = S[:, j]

                # project the acoustic feature into a cubic spline basis
                B = cubic_spline_basis(x)

                # get the encoder weights for this acoustic feature
                wj = w[si:ei]

                # get the prediction of the neural response based on the regression
                yhat = np.dot(B, wj)

                # un-zscore the acoustic feature
                xraw = (x * self.S[:, acoustic_feature_index].std(ddof=1)) + self.S[:, acoustic_feature_index].mean()

                # get the mean data, which looks like a tuning curve
                _xcenter, _ymean, _yerr, _ymean_cs = compute_mean_from_scatter(xraw, yhat)

                if acoustic_prop is None:
                    plt.figure()
                    plt.plot(xraw, y, 'ko', alpha=0.6, markersize=10)

                if acoustic_prop is not None and self.integer2prop[acoustic_feature_index] != acoustic_prop:
                    continue

                plt.plot(_xcenter, _ymean, 'g-', alpha=0.7, linewidth=4.0)
                plt.xlabel(self.integer2prop[acoustic_feature_index])
                plt.ylabel('Neural Response')
                plt.axis('tight')

                if acoustic_prop is None:
                    plt.title(
                        'Neural feature %d (%s)' % (neural_feature_index, self.integer2prop[acoustic_feature_index]))
                    plt.show()

    def save(self, output_file):
        hf = h5py.File(output_file, 'w')

        hf.attrs['preproc_file'] = self.preproc_file
        hf.attrs['integer2prop'] = self.integer2prop

        egrp = hf.create_group('encoders')
        for neural_feature_index, edict in self.good_encoders:
            fgrp = egrp.create_group('_%d' % neural_feature_index)
            fgrp.attrs['neural_feature_index'] = neural_feature_index
            fgrp.attrs['features'] = edict['features']
            fgrp.attrs['feature_improvements'] = edict['feature_improvements']
            fgrp.attrs['r2'] = edict['r2']
            fgrp['W'] = edict['W']
            fgrp['b'] = edict['b']

        hf['good_decoders'] = np.array([x[0] for x in self.good_decoders])
        hf['decoder_weights'] = np.array([x[1]['W'] for x in self.good_decoders])
        hf['decoder_perfs'] = np.array([x[1]['r2'] for x in self.good_decoders])

        hf['decoder_weights_ind'] = np.array([x[1]['W_independent'] for x in self.good_decoders])
        hf['decoder_perfs_ind'] = np.array([x[1]['r2_independent'] for x in self.good_decoders])

        hf.close()

    @classmethod
    def load(clz, output_file):

        ped = AcousticEncoderDecoder()

        hf = h5py.File(output_file, 'r')

        ped.preproc_file = hf.attrs['preproc_file']
        if 'integer2prop' in list(hf.attrs.keys()):
            ped.integer2prop = hf.attrs['integer2prop']
        else:
            # TODO this is an awful hack!!!
            ped.integer2prop = USED_ACOUSTIC_PROPS

        ped.good_encoders = list()
        ped.encoder_perfs = list()
        ped.encoder_weights = list()
        ped.encoder_features = list()
        ped.encoder_feature_improvements = list()
        for fgrp in list(hf['encoders'].values()):
            ped.good_encoders.append(fgrp.attrs['neural_feature_index'])
            ped.encoder_perfs.append(fgrp.attrs['r2'])
            ped.encoder_weights.append(np.array(fgrp['W']))
            ped.encoder_feature_improvements.append(fgrp.attrs['feature_improvements'])
            ped.encoder_features.append(fgrp.attrs['features'])

        ped.good_decoders = list(hf['good_decoders'])
        ped.decoder_perfs = np.array(hf['decoder_perfs'])
        ped.decoder_weights = np.array(hf['decoder_weights'])

        ped.decoder_perfs_ind = np.array(hf['decoder_perfs_ind'])
        ped.decoder_weights_ind = np.array(hf['decoder_weights_ind'])

        hf.close()

        return ped


if __name__ == '__main__':
    exp_name = 'GreBlu9508M'
    agg_dir = '/auto/tdrive/mschachter/data/aggregate'
    preproc_dir = '/auto/tdrive/mschachter/data/%s/preprocess' % exp_name
    decoder_dir = '/auto/tdrive/mschachter/data/%s/decoders' % exp_name

    seg_uname = 'Site4_Call1_L'
    decomp = 'spike_rate'
    # decomp = 'full_psds'
    preproc_file = os.path.join(preproc_dir, 'preproc_%s_%s.h5' % (seg_uname, decomp))
    output_file = os.path.join(decoder_dir, '_acoustic_encoder_decoder_%s_%s.h5' % (decomp, seg_uname))

    aed = AcousticEncoderDecoder()
    aed.fit(preproc_file, model_type='linear', encoder=True, decoder=True, zscore_response=True)
    aed.save(output_file)
    # aed = AcousticEncoderDecoder.load(output_file)
    # aed.plot_tuning_curves(acoustic_prop='stdtime')
    # plt.axis('tight')
    # plt.show()
