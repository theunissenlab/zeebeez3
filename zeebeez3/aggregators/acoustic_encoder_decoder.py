import os

import h5py
import numpy as np
import pandas as pd

from soundsig.signal import find_extrema
from zeebeez3.aggregators.biosound import AggregateBiosounds
from zeebeez3.models.acoustic_encoder_decoder import AcousticEncoderDecoder
from zeebeez3.core.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, clean_region, \
    decode_if_bytes


class AcousticEncoderDecoderAggregator(object):

    def __init__(self):
        self.df = None
        self.data = None

        self.encoder_weights = None
        self.encoder_perfs = None
        self.encoder_features = None
        self.encoder_perf_improvements = None
        self.tuning_curves = None
        self.tuning_curves_x = None

        self.decoder_weights_ind = None
        self.decoder_perfs_ind = None

        self.decoder_weights = None
        self.decoder_perfs = None

        self.cell_index2electrode = None
        self.index2electrode = None
        self.index2cell = None
        self.acoustic_props = None

    def read(self, data_dir='/auto/tdrive/mschachter/data'):

        self.data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'decomp':list(),
                     'iindex':list(), 'wkey':list()}

        self.index2electrode = dict()
        self.index2cell = dict()
        self.cell_index2electrode = dict()
        self.acoustic_props = list()

        protocol_file = os.path.join(data_dir, 'aggregate', 'protocol_data.csv')
        pdata = pd.read_csv(protocol_file)

        g = pdata.groupby(['bird', 'block', 'segment'])

        """
        all_decomps = [('trial_avg_psds',), ('mean_sub_psds',), ('full_psds',),
                       ('trial_avg_psds', 'trial_avg_cfs',),
                       ('mean_sub_psds', 'mean_sub_cfs',),
                       ('full_psds', 'full_cfs',),
                       ('spike_rate',), ('spike_sync',),
                       ('spike_rate', 'spike_sync'),
                       ]
        """

        all_decomps = [('full_psds',),
                       ('spike_rate',),
                       ('spike_rate', 'spike_sync'),
                       ]

        self.encoder_weights = dict()
        self.encoder_perfs = dict()
        self.encoder_features = dict()
        self.encoder_perf_improvements = dict()
        self.tuning_curves = dict()
        self.tuning_curves_x = dict()

        self.decoder_weights = dict()
        self.decoder_perfs = dict()

        self.decoder_weights_ind = dict()
        self.decoder_perfs_ind = dict()

        bs_agg = AggregateBiosounds.load(os.path.join(data_dir, 'aggregate', 'biosound.h5'))

        for (bird,block,segment),gdf in g:

            if bird == 'BlaBro09xxF':
                continue

            for hemi in ['L', 'R']:
                for decomps in all_decomps:
                    decomp = '+'.join(decomps)

                    seg_uname = '%s_%s_%s' % (block, segment, hemi)

                    preproc_dir = os.path.join(data_dir, bird, 'preprocess')
                    decoder_dir = os.path.join(data_dir, bird, 'models')

                    # read the LFP preproc and pard files
                    fname = '%s_%s' % (seg_uname, decomp)

                    preproc_file = os.path.join(preproc_dir, 'preproc_%s.h5' % fname)
                    pard_file = os.path.join(decoder_dir, 'acoustic_encoder_decoder_%s.h5' % fname)

                    self.read_encdec_file(preproc_file, pard_file, bird, block, segment, hemi, decomp, bs_agg)

        self.df = pd.DataFrame(self.data)

    def read_encdec_file(self, preproc_file, aed_file, bird, block, segment, hemi, decomp, bs_agg):

        if not os.path.exists(preproc_file):
            print('MISSING PREPROC FILE: %s' % preproc_file)
            return

        if not os.path.exists(aed_file):
            print('Missing acoustic encoder decoder file: %s' % aed_file)
            return

        wkey = '%s_%s_%s_%s_%s' % (bird, block, segment, hemi, decomp)

        print('Reading acoustic encoder decoder file: %s' % aed_file)

        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
        if hemi == 'R':
            electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

        hf = h5py.File(preproc_file, 'r')
        cell_index2electrode = list(hf.attrs['cell_index2electrode'])
        index2electrode = list(hf.attrs['index2electrode'])
        index2prop = list(hf.attrs['integer2prop'])
        freqs = list(hf.attrs['freqs'])
        lags = list(hf.attrs['lags'])
        hf.close()

        aed = AcousticEncoderDecoder.load(aed_file)
        aed_tuning_curves_x,aed_tuning_curves = aed.compute_tuning_curves()
        num_tuning_curve_bins = aed_tuning_curves_x.shape[-1]
        # print 'num_tuning_curve_bins=%d' % num_tuning_curve_bins

        nelectrodes = len(index2electrode)
        ncells = len(cell_index2electrode)
        nfreqs = len(freqs)
        nprops = len(index2prop)
        nlags = len(lags)

        # determine re-ordered list of cells
        cell_order = list()
        ci2e = np.array(cell_index2electrode)
        for e in electrode_order:
            ii = np.where(ci2e == e)[0]
            cell_order.extend(ii)

        # determine list of indices for cross terms (lfp)
        index2cross = list()
        for n1,e1 in enumerate(index2electrode):
            for n2 in range(n1):
                e2 = index2electrode[n2]
                for lk,l in enumerate(lags):
                    index2cross.append( (e1, e2, lk) )

        # determine list of indices for cross terms (spikes)
        index2cross_sync = list()
        for ci1, e1 in enumerate(cell_index2electrode):
            for ci2 in range(ci1):
                index2cross_sync.append( (ci1, ci2) )

        dweights_ind = None
        dperfs_ind = None

        spline_dof = 6
        if decomp.endswith('psds'):

            nelectrodes = len(index2electrode)

            # read the encoder weights and performances
            eweights = np.zeros([nelectrodes, nfreqs, nprops*spline_dof])
            eperfs = np.zeros([nelectrodes, nfreqs])
            good_acoustic_features = np.zeros([nelectrodes, nfreqs, nprops], dtype='bool')
            feature_improvements = np.zeros([nelectrodes, nfreqs, nprops])

            tuning_curves_x = np.zeros([nelectrodes, nfreqs, nprops, num_tuning_curve_bins])
            tuning_curves = np.zeros([nelectrodes, nfreqs, nprops, num_tuning_curve_bins])

            for i,k in enumerate(aed.good_encoders):

                fi = k % nfreqs
                ei = k / nfreqs
                e = index2electrode[ei]
                ei_ordered = electrode_order.index(e)

                # print 'f=%d, e=%d, ei=%d, k=%d, i=%d' % (f, e, ei, k)

                eperfs[ei_ordered, fi] = aed.encoder_perfs[i]
                # there are only regression weights for features that were included in the stagewise
                # regression. so here we extract those weights and put them in a matrix that has space
                # for all feature weights. weights for features not regressed on are left equal to zero
                for j,jj in enumerate(aed.encoder_features[i]):
                    si = j*spline_dof
                    ei = si + spline_dof

                    sii = jj*spline_dof
                    eii = sii + spline_dof

                    eweights[ei_ordered, fi, sii:eii] = aed.encoder_weights[i][si:ei]
                    feature_improvements[ei_ordered, fi, jj] = aed.encoder_feature_improvements[i][j]
                    good_acoustic_features[ei_ordered, fi, jj] = True
                    tuning_curves_x[ei_ordered, fi, jj, :] = aed_tuning_curves_x[i, jj, :]
                    tuning_curves[ei_ordered, fi, jj, :] = aed_tuning_curves[i, jj, :]

            # read the decoder weights and performances
            dweights = np.zeros([nelectrodes, nfreqs, nprops])
            dperfs = np.zeros([nprops])

            dperfs_ind = np.zeros([nprops, nelectrodes, nfreqs])
            dweights_ind = np.zeros([nprops, nelectrodes, nfreqs])

            for i,k in enumerate(aed.good_decoders):
                dperfs[k] = aed.decoder_perfs[i]

                W = aed.decoder_weights[i].reshape([nelectrodes, nfreqs])
                # reorder the weight matrix
                Wrs = np.zeros_like(W)

                for n,e in enumerate(index2electrode):
                    nn = electrode_order.index(e)
                    Wrs[nn, :] = W[n, :]

                dweights[:, :, k] = Wrs
                dperfs_ind[k, :, :] = aed.decoder_perfs_ind[i, :].reshape([nelectrodes, nfreqs])
                dweights_ind[k, :, :] = aed.decoder_weights_ind[i, :].reshape([nelectrodes, nfreqs])

        elif decomp == 'spike_rate':

            # read encoder weights
            spline_dof = 6
            eweights = np.zeros([ncells, nprops*spline_dof])
            eperfs = np.zeros([ncells])
            good_acoustic_features = np.zeros([ncells, nprops], dtype='bool')
            feature_improvements = np.zeros([ncells, nprops])

            tuning_curves_x = np.zeros([ncells, nprops, num_tuning_curve_bins])
            tuning_curves = np.zeros([ncells, nprops, num_tuning_curve_bins])

            for i,k in enumerate(aed.good_encoders):
                ci = cell_order.index(k)

                eperfs[ci] = aed.encoder_perfs[i]

                # print 'encoder_weights[%d].shape=%s' % (i, str(eweights[i].shape))
                for j,jj in enumerate(aed.encoder_features[i]):
                    si = j*spline_dof
                    ei = si + spline_dof

                    sii = jj * spline_dof
                    eii = sii + spline_dof

                    # print 'si=%d, ei=%d, sii=%d, eii=%d' % (si, ei, sii, eii)

                    eweights[ci, sii:eii] = aed.encoder_weights[i][si:ei]
                    good_acoustic_features[ci, jj] = True
                    feature_improvements[ci, jj] = aed.encoder_feature_improvements[i][j]

                    tuning_curves_x[ci, jj, :] = aed_tuning_curves_x[i, jj, :]
                    tuning_curves[ci, jj, :] = aed_tuning_curves[i, jj, :]

            # read decoder weights
            dweights = np.zeros([ncells, nprops])
            dperfs = np.zeros([nprops])

            dperfs_ind = np.zeros([nprops, ncells])
            dweights_ind = np.zeros([nprops, ncells])
            for i,k in enumerate(aed.good_decoders):
                dweights[:, k] = aed.decoder_weights[i]
                dperfs[k] = aed.decoder_perfs[i]

                dperfs_ind[k, :] = aed.decoder_perfs_ind[i, :]
                dweights_ind[k, :] = aed.decoder_weights_ind[i, :]

        elif decomp.endswith('cfs'):

            # read encoder weights
            eweights = np.zeros([nelectrodes, nelectrodes, nlags, nprops])
            eperfs = np.zeros([nelectrodes, nelectrodes, nlags])
            good_acoustic_features = np.zeros([nelectrodes, nelectrodes, nlags, nprops], dtype='bool')
            feature_improvements = np.zeros([nelectrodes, nelectrodes, nlags, nprops])
            tuning_curves = np.zeros([nelectrodes, nelectrodes, nlags, nprops, num_tuning_curve_bins])
            tuning_curves_x = np.zeros([nelectrodes, nelectrodes, nlags, nprops, num_tuning_curve_bins])

            """
            for i,k in enumerate(aed.good_encoders):
                if k < nfreqs*nelectrodes:
                    # skip the weights for the power spectra (diagonal of matrix)
                    continue

                # determine electrode1, electrode2, and lag from index
                e1,e2,li = index2cross[k-nfreqs*nelectrodes]

                i1 = electrode_order.index(e1)
                i2 = electrode_order.index(e2)
                eperfs[i1, i2, li] = aed.encoder_perfs[i]
                eperfs[i2, i1, li] = aed.encoder_perfs[i]

                eweights[i1, i2, li, :] = aed.encoder_weights[i]
                eweights[i2, i1, li, :] = aed.encoder_weights[i]
            """

            # read decoder weights
            dweights = np.zeros([nelectrodes, nelectrodes, nlags, nprops])
            dperfs = np.zeros([nprops])
            for i,k in enumerate(aed.good_decoders):
                w = aed.decoder_weights[i]
                W = np.zeros([nelectrodes, nelectrodes, nlags])
                for ii,(e1,e2,li) in enumerate(index2cross):
                    i1 = electrode_order.index(e1)
                    i2 = electrode_order.index(e2)
                    wv = w[ii + nelectrodes*nfreqs]
                    W[i1, i2, li] = wv
                    W[i2, i1, li] = wv
                dweights[:, :, :, k] = W

                dperfs[k] = aed.decoder_perfs[i]

        elif decomp.endswith('sync'):

            offset = 0
            if 'rate' in decomp:
                # offset things by the # of cells, so we don't bother with the encoder weights for spike rate
                offset = ncells

            eweights = np.zeros([ncells, ncells, nprops])
            eperfs = np.zeros([ncells, ncells])
            good_acoustic_features = np.zeros([ncells, ncells, nprops], dtype='bool')
            feature_improvements = np.zeros([ncells, ncells, nprops])
            tuning_curves = np.zeros([ncells, ncells, nprops, num_tuning_curve_bins])
            tuning_curves_x = np.zeros([ncells, ncells, nprops, num_tuning_curve_bins])

            """
            for i, k in enumerate(aed.good_encoders):
                if k < offset:
                    # don't bother with encoder weights for spike rate
                    continue

                ci1,ci2 = index2cross_sync[k-offset]

                i1 = cell_order.index(ci1)
                i2 = cell_order.index(ci2)

                ew = aed.encoder_weights[i]
                eweights[i1, i2] = ew
                eweights[i2, i1] = ew

                eperfs[i1, i2] = aed.encoder_perfs[i]
            """

            # read decoder weights
            dweights = np.zeros([ncells, ncells, nprops])
            dperfs = np.zeros([nprops])
            for i, k in enumerate(aed.good_decoders):

                for j,w in enumerate(aed.decoder_weights[i]):
                    if j < offset:
                        # skip decoder weights for spike rates
                        continue

                    ci1,ci2 = index2cross_sync[j-offset]
                    i1 = cell_order.index(ci1)
                    i2 = cell_order.index(ci2)

                    dweights[i1, i2, k] = w
                    dweights[i2, i1, k] = w

                dperfs[k] = aed.decoder_perfs[i]

        # save the weights

        # print "decomp=%s, eweights.shape=%s" % (decomp, str(eweights.shape))
        assert wkey not in self.encoder_weights
        self.encoder_weights[wkey] = eweights
        self.encoder_perfs[wkey] = eperfs
        self.encoder_features[wkey] = good_acoustic_features
        self.encoder_perf_improvements[wkey] = feature_improvements
        self.tuning_curves[wkey] = tuning_curves
        self.tuning_curves_x[wkey] = tuning_curves_x

        self.decoder_weights[wkey] = dweights
        self.decoder_perfs[wkey] = dperfs

        if dweights_ind is not None:
            self.decoder_weights_ind[wkey] = dweights_ind
        if dperfs_ind is not None:
            self.decoder_perfs_ind[wkey] = dperfs_ind

        # save the indices
        self.index2electrode[wkey] = index2electrode
        self.cell_index2electrode[wkey] = cell_index2electrode
        self.index2cell[wkey] = cell_order

        iindex = len(self.acoustic_props)
        self.acoustic_props.append(index2prop)

        # save the data
        self.data['bird'].append(bird)
        self.data['block'].append(block)
        self.data['segment'].append(segment)
        self.data['hemi'].append(hemi)
        self.data['decomp'].append(decomp)
        self.data['iindex'].append(iindex)
        self.data['wkey'].append(wkey)

    def get_tuning_curves(self, decomps=(('spike_rate', -1), ('full_psds', 0), ('full_psds', 1), ('full_psds', 2))):
        """ Get tuning curves for all sites and acoustic props, by decomposition. """

        all_curves = list()
        all_curves_x = list()
        data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(), 'cell_index':list(),
                'decomp':list(), 'band':list(), 'xindex':list(), 'r2':list(), 'aprop':list(),
                'dist_l2a':list(), 'dist_midline':list(), 'region':list(), 'amp_slope':list(), 'center_freq':list()}

        edata = pd.read_csv(os.path.join('/auto/tdrive/mschachter/data', 'aggregate', 'electrode_data+dist.csv'))
        i = edata.bird != 'BlaBro09xxF'
        edata = edata[i]

        perf_thresh = 0.05
        current_index = 0
        for k, aprop in enumerate(self.acoustic_props[0]):
            for j, (decomp,band_index) in enumerate(decomps):
                i = self.df.decomp == decomp
                assert i.sum() > 0, 'decomp=%s' % decomp

                # aggregate tuning curves across sites
                for (bird, block, segment, hemi), gdf in self.df[i].groupby(['bird', 'block', 'segment', 'hemi']):
                    wkey = '%s_%s_%s_%s_%s' % (bird, block, segment, hemi, decomp)
                    eperfs = self.encoder_perfs[wkey]
                    tuning_curves = self.tuning_curves[wkey]
                    tuning_curves_x = self.tuning_curves_x[wkey]
                    acoustic_feature_index = list(self.acoustic_props[0]).index(aprop)
                    index2electrode = self.index2electrode[wkey]
                    cell_index2electrode = self.cell_index2electrode[wkey]

                    if decomp == 'spike_rate':
                        ncells, nprops, nbins = tuning_curves.shape
                        aprop_tc = tuning_curves[:, acoustic_feature_index, :]
                        aprop_tc_x = tuning_curves_x[:, acoustic_feature_index, :]

                        for k in range(ncells):
                            e = cell_index2electrode[k]
                            ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (
                            edata.electrode == e)
                            assert ei.sum() == 1
                            reg = clean_region(edata.region[ei].values[0])
                            dist_l2a = edata.dist_l2a[ei].values[0]
                            dist_midline = edata.dist_midline[ei].values[0]
                            if bird == 'GreBlu9508M':
                                dist_l2a *= 4

                            data['bird'].append(bird)
                            data['block'].append(block)
                            data['segment'].append(segment)
                            data['hemi'].append(hemi)
                            data['decomp'].append(decomp)
                            data['band'].append(band_index)
                            data['r2'].append(eperfs[k])
                            data['xindex'].append(current_index)
                            data['aprop'].append(aprop)
                            data['electrode'].append(e)
                            data['cell_index'].append(k)
                            data['dist_l2a'].append(dist_l2a)
                            data['dist_midline'].append(dist_midline)
                            data['region'].append(reg)

                            slope = 0.
                            if aprop in ['maxAmp', 'meanspect', 'sal', 'skewtime']:
                                # compute slope of tuning curve
                                _tc = aprop_tc[k, :]
                                _tc_x = aprop_tc_x[k, :]
                                if ~np.any(np.isnan(_tc)) & ~np.any(np.isinf(_tc)) & (np.sum(np.abs(_tc)) != 0):
                                    try:
                                        slope, b = np.polyfit(_tc_x, _tc, deg=1)
                                    except ValueError:
                                        print('Problem estimating slope for aprop=%s, decomp=spike_rate, cell+index=%d, bird=%s, block=%s, segment=%s, hemi=%s' % \
                                              (aprop, k, bird, block, segment, hemi))
                                        slope = 0.

                            cfreq = 0
                            if aprop == 'meanspect':
                                # compute center frequency
                                _tc = aprop_tc[k, :]
                                _tc_x = aprop_tc_x[k, :]
                                mini, maxi = find_extrema(_tc)
                                if len(maxi) > 0:
                                    cfreq = _tc_x[maxi[0]]

                            data['amp_slope'].append(slope)
                            data['center_freq'].append(cfreq)

                            current_index += 1

                    elif decomp == 'full_psds':

                        nelectrodes, nfreqs, nprops, nbins = tuning_curves.shape
                        aprop_tc = tuning_curves[:, band_index, acoustic_feature_index, :]
                        aprop_tc_x = tuning_curves_x[:, band_index, acoustic_feature_index, :]

                        for k,e in enumerate(index2electrode):
                            ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (
                                edata.electrode == e)
                            assert ei.sum() == 1
                            reg = clean_region(edata.region[ei].values[0])
                            dist_l2a = edata.dist_l2a[ei].values[0]
                            dist_midline = edata.dist_midline[ei].values[0]
                            if bird == 'GreBlu9508M':
                                dist_l2a *= 4

                            data['bird'].append(bird)
                            data['block'].append(block)
                            data['segment'].append(segment)
                            data['hemi'].append(hemi)
                            data['decomp'].append(decomp)
                            data['band'].append(band_index)
                            data['r2'].append(eperfs[k, band_index])
                            data['xindex'].append(current_index)
                            data['aprop'].append(aprop)
                            data['electrode'].append(e)
                            data['cell_index'].append(-1)
                            data['dist_l2a'].append(dist_l2a)
                            data['dist_midline'].append(dist_midline)
                            data['region'].append(reg)

                            slope = 0.
                            if aprop in ['maxAmp', 'meanspect', 'sal', 'skewtime']:
                                # compute slope of tuning curve
                                _tc = aprop_tc[k, :]
                                _tc_x = aprop_tc_x[k, :]
                                if ~np.any(np.isnan(_tc)) & ~np.any(np.isinf(_tc)) & (np.sum(np.abs(_tc)) != 0):
                                    try:
                                        slope, b = np.polyfit(_tc_x, _tc, deg=1)
                                    except ValueError:
                                        print('Problem estimating slope for aprop=%s, decomp=lfp_psd, band_index=%d, cell+index=%d, bird=%s, block=%s, segment=%s, hemi=%s' % \
                                              (aprop, band_index, k, bird, block, segment, hemi))
                                        print(_tc)
                                        slope = 0.

                            cfreq = 0
                            if aprop == 'meanspect':
                                # compute center frequency
                                _tc = aprop_tc[k, :]
                                _tc_x = aprop_tc_x[k, :]
                                mini, maxi = find_extrema(_tc)
                                if len(maxi) > 0:
                                    cfreq = _tc_x[maxi[0]]

                            data['amp_slope'].append(slope)
                            data['center_freq'].append(cfreq)

                            current_index += 1
                    else:
                        continue

                    all_curves.extend(aprop_tc)
                    all_curves_x.extend(aprop_tc_x)

        all_curves = np.array(all_curves)
        all_curves_x = np.array(all_curves_x)
        df = pd.DataFrame(data)

        """
        # get rid of bad tuning curves
        tc_sum = np.abs(all_curves).sum(axis=1)
        # good_i = (tc_sum > 0) & (all_perfs > perf_thresh)
        tc_diff = np.diff(all_curves, axis=1)
        tc_diff_max = tc_diff.max(axis=1)
        tc_diff_min = tc_diff.min(axis=1)
        tc_max = all_curves.max(axis=1)
        tc_min = all_curves.min(axis=1)
        good_i = (all_perfs > perf_thresh) & (tc_diff_max < 1.5) & (tc_diff_min > -1.5) & (tc_max < 2) & (tc_min > -2) & (tc_max > -0.5) & (tc_sum > 0)
        """

        return all_curves_x,all_curves,df

    def export_decoder_csv(self, output_file):

        decoder_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
                        'decomp':list(), 'r2':list(), 'aprop':list()}

        g = self.df.groupby(['bird', 'block', 'segment', 'hemi', 'decomp'])

        for (bird,block,segment,hemi,decomp),gdf in g:

            assert len(gdf) == 1
            wkey = gdf.wkey.values[0]
            iindex = gdf.iindex.values[0]

            aprops = self.acoustic_props[iindex]
            dperfs = self.decoder_perfs[wkey]

            for k,aprop in enumerate(aprops):
                decoder_data['bird'].append(bird)
                decoder_data['block'].append(block)
                decoder_data['segment'].append(segment)
                decoder_data['hemi'].append(hemi)
                decoder_data['decomp'].append(decomp)
                decoder_data['aprop'].append(aprop)
                decoder_data['r2'].append(dperfs[k])

        df = pd.DataFrame(decoder_data)
        df.to_csv(output_file, header=True, index=False)

    def save(self, agg_file):

        hf = h5py.File(agg_file, 'w')
        hf.attrs['wkeys'] = list(self.encoder_weights.keys())
        hf.attrs['col_names'] = list(self.data.keys())
        for k,v in list(self.data.items()):
            hf[k] = v

        hf['acoustic_props'] = np.array(self.acoustic_props)

        for wkey in list(self.encoder_weights.keys()):
            grp = hf.create_group(wkey)
            grp['encoder_weights'] = self.encoder_weights[wkey]
            grp['encoder_perfs'] = self.encoder_perfs[wkey]
            grp['encoder_features'] = self.encoder_features[wkey]
            grp['encoder_perf_improvements'] = self.encoder_perf_improvements[wkey]

            grp['tuning_curves'] = self.tuning_curves[wkey]
            grp['tuning_curves_x'] = self.tuning_curves_x[wkey]

            grp['decoder_weights'] = self.decoder_weights[wkey]
            grp['decoder_perfs'] = self.decoder_perfs[wkey]

            grp['index2electrode'] = self.index2electrode[wkey]
            grp['index2cell'] = self.index2cell[wkey]
            grp['cell_index2electrode'] = self.cell_index2electrode[wkey]

            if wkey in self.decoder_perfs_ind:
                grp['encoder_perfs_ind'] = self.decoder_perfs_ind[wkey]
            if wkey in self.decoder_weights_ind:
                grp['encoder_weights_ind'] = self.decoder_weights_ind[wkey]

        hf.close()

    @classmethod
    def load(cls, agg_file):

        agg = AcousticEncoderDecoderAggregator()

        hf = h5py.File(agg_file, 'r')

        agg.data = dict()
        for cname in hf.attrs['col_names']:
            agg.data[decode_if_bytes(cname)] = np.array([decode_if_bytes(s) for s in hf[cname]])
        agg.df = pd.DataFrame(agg.data)

        aprops = list()
        for row in hf['acoustic_props']:
            aprops.append([decode_if_bytes(s) for s in row])
        agg.acoustic_props = np.array(aprops)

        agg.encoder_weights = dict()
        agg.encoder_perfs = dict()
        agg.encoder_features = dict()
        agg.encoder_perf_improvements = dict()
        agg.tuning_curves = dict()
        agg.tuning_curves_x = dict()

        agg.decoder_weights_ind = dict()
        agg.decoder_perfs_ind = dict()
        agg.decoder_weights = dict()
        agg.decoder_perfs = dict()

        agg.index2electrode = dict()
        agg.cell_index2electrode = dict()
        agg.index2cell = dict()

        for wkey in hf.attrs['wkeys']:
            grp = hf[wkey]

            wkey = decode_if_bytes(wkey)
            agg.encoder_perfs[wkey] = np.array(grp['encoder_perfs'])
            agg.encoder_weights[wkey] = np.array(grp['encoder_weights'])
            agg.encoder_features[wkey] = list(grp['encoder_features'])
            agg.encoder_perf_improvements[wkey] = np.array(grp['encoder_perf_improvements'])
            agg.tuning_curves[wkey] = np.array(grp['tuning_curves'])
            agg.tuning_curves_x[wkey] = np.array(grp['tuning_curves_x'])

            agg.decoder_perfs[wkey] = np.array(grp['decoder_perfs'])
            agg.decoder_weights[wkey] = np.array(grp['decoder_weights'])

            if 'decoder_perfs_ind' in grp:
                agg.decoder_perfs_ind[wkey] = np.array(grp['decoder_perfs_ind'])
            if 'decoder_weights_ind' in grp:
                agg.decoder_weights_ind[wkey] = np.array(grp['decoder_weights_ind'])

            agg.index2electrode[wkey] = np.array(grp['index2electrode'])
            agg.cell_index2electrode[wkey] = np.array(grp['cell_index2electrode'])
            agg.index2cell[wkey] = np.array(grp['index2cell'])

        hf.close()

        return agg


if __name__ == '__main__':

    agg = AcousticEncoderDecoderAggregator()
    agg.read()
    agg.save('/auto/tdrive/mschachter/data/aggregate/acoustic_encoder_decoder.h5')

    # agg = PARDAggregator.load('/auto/tdrive/mschachter/data/aggregate/acoustic_encoder_decoder.h5')
