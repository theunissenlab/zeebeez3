import os
import h5py
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from soundsig.plots import multi_plot
from soundsig.signal import get_envelope_end, lowpass_filter, highpass_filter, bandpass_filter, break_envelope_into_events
from soundsig.sound import plot_spectrogram
from soundsig.spikes import plot_raster, compute_psth
from soundsig.timefreq import wavelet_scalogram

from zeebeez3.core.experiment import segment_to_unique_name
from zeebeez3.core.utils import decode_if_bytes, decode_column_if_bytes


class StimEventTransform(object):

    def __init__(self):
        self.experiment_file = None
        self.stimulus_file = None
        self.bird = None
        self.block_name = None
        self.segment_name = None
        self.seg_uname = None
        self.rcg_names = None

        self.lfp_sample_rate = None

        self.pre_stim_time = None
        self.post_stim_time = None

        self.lfp_rep_params = None
        self.lfp_reps_by_stim = None

        self.ncells = None
        self.sort_code = None
        self.spikes_by_stim = None

        self.spec_freq = None
        self.spec_by_stim = None
        self.spec_rms_by_stim = None

        self.trial_data = None
        self.trial_df = None

        self.electrode_data = None
        self.electrode_df = None
        self.index2electrode = None

        self.segment_data = None
        self.segment_df = None

        self.cell_data = None
        self.cell_df = None

    def set_segment(self, experiment, segment, rcg_names, pre_stim_time=1.0, post_stim_time=1.0):

        self.experiment_file = experiment.file_name
        self.stimulus_file = experiment.stimulus_file_name
        self.bird = experiment.bird_name
        self.block_name = segment.block.name
        self.segment_name = segment.name
        self.seg_uname = segment_to_unique_name(segment)
        segment_duration = segment.annotations['duration']

        self.rcg_names = rcg_names

        self.pre_stim_time = pre_stim_time
        self.post_stim_time = post_stim_time

        self.lfp_sample_rate = experiment.lfp_sample_rate
        self.lfp_reps_by_stim = dict()
        self.lfp_rep_params = dict()

        # initialize electrode information
        rcgs = [rcg for rcg in segment.block.recordingchannelgroups if rcg.name in rcg_names]
        self.electrode_data = {'electrode':list(), 'hemisphere':list(), 'region':list(), 'row':list(), 'col':list()}

        for rcg in rcgs:
            for e in rcg.channel_indexes:
                rcg2,rc = experiment.get_channel(segment.block, e)
                row,col = (rc.annotations['row'], rc.annotations['col'])
                region = rc.annotations['region']

                self.electrode_data['electrode'].append(e)
                self.electrode_data['hemisphere'].append(rcg.name)
                self.electrode_data['region'].append(region)
                self.electrode_data['row'].append(row[0])
                self.electrode_data['col'].append(col[0])
        self.electrode_df = pd.DataFrame(self.electrode_data)

        # map electrodes to rows in a matrix
        self.index2electrode = sorted(self.electrode_data['electrode'])

        # initialize stim data for segment
        etable = experiment.get_epoch_table(segment)
        stim_ids = etable['id'].unique()
        self.trial_data = {'stim_id':list(), 'stim_type':list(), 'start_time':list(), 'end_time':list(), 'stim_duration':list(),
                           'stim_start_time':list(), 'trial':list()}

        # get spectrograms for all the stimuli in the experiment
        self.spec_by_stim = dict()
        self.spec_rms_by_stim = dict()
        experiment.init_segment_spectrograms(sample_rate=self.lfp_sample_rate, window_length=0.015)
        seg_specs = experiment.segment_specs[self.seg_uname]
        seg_specs.load_all()
        seg_specs.log_transform()

        self.spec_freq = seg_specs.spec_freq
        for stim_id in stim_ids:
            t, freq, timefreq, sound = seg_specs.specs[stim_id]
            self.spec_by_stim[stim_id] = timefreq

        # determine the true end time of each stimulus
        actual_stim_durations = dict()
        for stim_id,spec in list(self.spec_by_stim.items()):
            spec = self.spec_by_stim[stim_id]
            rms = spec.std(ddof=1, axis=0)
            self.spec_rms_by_stim[stim_id] = rms
            tend = get_envelope_end(rms)
            actual_stim_durations[stim_id] = tend / self.lfp_sample_rate

        # construt a DataFrame to hold stimulus presentation information.
        for stim_id in stim_ids:

            # get stimulus start and end times
            index = etable['id'] == stim_id
            assert index.sum() > 0, "Stim %d has zero trials!" % stim_id

            stimes = etable[index]['start_time'].values.astype('float')
            etimes = etable[index]['end_time'].values.astype('float')
            ntrials = index.sum()
            assert ntrials > 0, "Zero trials for stim %d: %d" % (stim_id, ntrials)

            # get the stim type
            si = experiment.stim_table['id'] == stim_id
            assert si.sum() == 1, "More than one stimulus defined for id=%d" % stim_id
            stim_type = experiment.stim_table['type'][si].values[0]
            if stim_type == 'call':
                stim_type = experiment.stim_table['callid'][si].values[0]

            # sort stimulus presentations by start time, in case they not already sorted
            lst = list(zip(stimes, etimes))
            lst.sort(key=operator.itemgetter(0))

            durations = etimes - stimes
            stim_dur = durations.max()
            actual_stim_dur = actual_stim_durations[stim_id]
            assert stim_dur >= actual_stim_dur, ".wav file specified duration is %0.6fs, actual duration is %0.6fs" % \
                                                (stim_dur, actual_stim_dur)

            for k,(stime,etime) in enumerate(lst):
                # specify a slice of time that includes pre- and post-stimulus presentation activity
                sstime = max(0, stime - pre_stim_time)
                eetime = min(stime+actual_stim_dur+post_stim_time, segment_duration)

                # add to the dataset
                self.trial_data['stim_id'].append(stim_id)
                self.trial_data['stim_type'].append(stim_type)
                self.trial_data['stim_start_time'].append(stime)
                self.trial_data['start_time'].append(sstime)
                self.trial_data['end_time'].append(eetime)
                self.trial_data['stim_duration'].append(actual_stim_dur)
                self.trial_data['trial'].append(k)

        self.trial_df = pd.DataFrame(self.trial_data)

    def attach_raw(self, experiment, segment, sort_code='single'):

        assert self.experiment_file == experiment.file_name, "Mismatched experiment files: %s vs %s" % \
                                                             (self.experiment_file, experiment.file_name)
        assert self.segment_name == segment.name, "Mismatched segment names: %s vs %s" % (self.segment_name, segment.name)

        self.sort_code = sort_code

        # get all spike times
        spike_rasters = experiment.get_spike_slice(segment, 0.0, segment.annotations['duration'],
                                                   rcg_names=self.rcg_names, as_matrix=False,
                                                   sort_code=sort_code, bin_size=0.001)

        # reorganize spike raster data by merging cells from different recording channel groups
        cellindex2electrode = list()
        for rcg_name,(spike_trains,spike_group) in list(spike_rasters.items()):
            for electrode,cindex_list in list(spike_group.items()):
                for ci in cindex_list:
                    cellindex2electrode.append( (ci, electrode, rcg_name) )
        cellindex2electrode.sort(key=operator.itemgetter(1))
        self.ncells = len(cellindex2electrode)
        print('# of cells: %d' % self.ncells)
        print('cellindex2electrode: ', cellindex2electrode)

        all_spike_trains = list()
        for ci,electrode,rcg_name in cellindex2electrode:
            spike_trains,spike_groups = spike_rasters[rcg_name]
            all_spike_trains.append(spike_trains[ci])

        # set up cell data
        self.cell_data = {'index':list(), 'sort_code':list(), 'electrode':list(), 'hemisphere':list()}
        for k,(ci,electrode, hemi) in enumerate(cellindex2electrode):
            self.cell_data['index'].append(k)
            self.cell_data['sort_code'].append(sort_code)
            self.cell_data['electrode'].append(electrode)
            self.cell_data['hemisphere'].append(hemi)
        self.cell_df = pd.DataFrame(self.cell_data)

        # create a ragged array of spike data per stimulus, in the shape (num_trials, num_cells, num_spikes)
        stim_ids = self.trial_df['stim_id'].unique()
        self.spikes_by_stim = dict()
        for stim_id in stim_ids:
            i = self.trial_df['stim_id'] == stim_id
            ntrials = i.sum()
            spike_mat = list()
            for k in range(ntrials):
                ii = (self.trial_df['stim_id'] == stim_id) & (self.trial_df['trial'] == k)
                assert ii.sum() > 0, "No stim/trial combination found for stim_id=%d and trial %d" % (stim_id, k)
                assert ii.sum() == 1, "More than one stim/trial combination found for stim_id=%d and trial %d" % \
                                      (stim_id, k)

                trial_start_time = self.trial_df['start_time'][ii].values[0]
                trial_end_time = self.trial_df['end_time'][ii].values[0]
                stim_start_time = self.trial_df['stim_start_time'][ii].values[0]
                assert stim_start_time-trial_start_time == self.pre_stim_time, "stim_start_time=%0.6fs, trial_start_time=%0.6fs" % \
                                                                               (stim_start_time, trial_start_time)

                spike_trains_per_cell = list()
                lst = list(zip(self.cell_df['index'].values, self.cell_df['electrode'].values))
                lst.sort(key=operator.itemgetter(1))
                for ci,electrode in lst:
                    st = all_spike_trains[ci]
                    i = (st >= trial_start_time) & (st <= trial_end_time)
                    if i.sum() == 0:
                        spike_trains_per_cell.append(np.array([]))
                    else:
                        spike_trains_per_cell.append(st[i] - stim_start_time)

                spike_mat.append(spike_trains_per_cell)
            self.spikes_by_stim[stim_id] = spike_mat

        # aggregate raw LFPs by stimulus
        nelectrodes = len(self.electrode_df['electrode'].unique())
        self.lfp_reps_by_stim['raw'] = dict()
        for stim_id in stim_ids:
            # get stimulus start and end times
            index = self.trial_df['stim_id'] == stim_id
            assert index.sum() > 0, "Stim %d has zero trials!" % stim_id

            stimes = self.trial_df[index]['start_time'].values
            etimes = self.trial_df[index]['end_time'].values
            ntrials = index.sum()

            duration = (etimes - stimes).max()
            idur = int(self.lfp_sample_rate*duration)

            # build matrix to hold LFP responses
            raw_lfp = np.zeros([ntrials, nelectrodes, idur])
            for k,(stime,etime) in enumerate(zip(stimes, etimes)):
                for ei,electrode in enumerate(self.index2electrode):
                    # get the raw LFP for this electrode
                    rlfp = experiment.get_single_lfp_slice(segment, electrode, stime, stime+duration)
                    d = min(len(rlfp), idur)
                    raw_lfp[k, ei, :d] = rlfp[:d]
            self.lfp_reps_by_stim['raw'][stim_id] = raw_lfp

    def attach_wavelet(self, frequencies=None, num_cycles_per_window=10):

        assert 'raw' in self.lfp_reps_by_stim

        if frequencies is None:
            b = 3
            x1 = np.log(4.) / np.log(b)
            x2 = np.log(190.) / np.log(b)
            frequencies = np.logspace(x1, x2, 20, base=b)

        increment = (1.0 / self.lfp_sample_rate)*2

        self.lfp_rep_params['wavelet'] = {'frequencies':frequencies, 'num_cycles_per_window':num_cycles_per_window,
                                          'increment':increment}

        self.lfp_reps_by_stim['wavelet'] = dict()
        stim_ids = self.trial_df['stim_id'].unique()
        for stim_id in stim_ids:

            print('Computing wavelet transform for stim %d...' % stim_id)

            X = self.lfp_reps_by_stim['raw'][stim_id]
            ntrials,nelectrodes,nt = X.shape
            nt /= 2
            nt += 1

            Ws = np.zeros([ntrials, len(frequencies), nelectrodes, nt], dtype='complex')

            for k in range(ntrials):
                for n in range(nelectrodes):
                    # compute the wavelet transform
                    wt,wfreq,wtf = wavelet_scalogram(X[k, n, :], self.lfp_sample_rate, increment,
                                                     frequencies=frequencies,
                                                     num_cycles_per_window=num_cycles_per_window)
                    nnt = min(nt, wtf.shape[-1])
                    Ws[k, :, n, :nnt] = wtf[:, :nnt]

            self.lfp_reps_by_stim['wavelet'][stim_id] = Ws

    def attach_memd(self, memd_transform_file):

        from zeebeez3.transforms.memd import MEMDTransform

        # open up the MEMD file and read the raw IMFs
        mt = MEMDTransform.load(memd_transform_file, load_complex=False)
        X = mt.get_bands()
        nbands,nelectrodes,nt_full = X.shape

        self.lfp_rep_params['memd'] = {'nbands':nbands, 'file_name':mt.file_name}

        self.lfp_reps_by_stim['memd'] = dict()
        stim_ids = self.trial_df['stim_id'].unique()
        for stim_id in stim_ids:

            i = self.trial_data['stim_id'] == stim_id
            ntrials = i.sum()

            stim_dur = self.trial_df['stim_duration'][i].values[0]
            slice_dur = self.pre_stim_time + stim_dur + self.post_stim_time
            idur = int(self.lfp_sample_rate*slice_dur)

            imfs = np.zeros([ntrials, nbands, nelectrodes, idur])
            for k in range(ntrials):
                ii = (self.trial_df['stim_id'] == stim_id) & (self.trial_df['trial'] == k)
                assert ii.sum() == 1, "More than one trial for stim %d, trial %d" % (stim_id, k)
                stime = self.trial_df['start_time'][ii].values[0]
                si = int(self.lfp_sample_rate*stime)
                ei = si + idur
                for b in range(nbands):
                    imfs[k, b, :, :] = X[b, :, si:ei]
            self.lfp_reps_by_stim['memd'][stim_id] = imfs

    def attach_bandpass(self, bands=((80.0, np.inf), (30.0, 80.0), (15.0, 30.0), (5.0, 15.0), (2.0, 5.0), (-np.inf, 2.0))):

        assert 'raw' in self.lfp_reps_by_stim, "The raw LFP representation must be loaded before bandpass can be computed!"

        self.lfp_rep_params['bandpass'] = {'bands':bands}
        self.lfp_reps_by_stim['bandpass'] = dict()
        nbands = len(bands)

        # compute the mean and std of the raw LFP
        stim_ids = self.trial_df['stim_id'].unique()
        all_lfps = list()
        for stim_id in stim_ids:
            lfp = self.lfp_reps_by_stim['raw'][stim_id]
            all_lfps.extend(lfp.ravel())
        lfp_mean = np.mean(all_lfps)
        lfp_std = np.std(all_lfps, ddof=1)
        del all_lfps

        for stim_id in stim_ids:
            print('Bandpass filtering stim %d...' % stim_id)
            # get the raw LFP representation
            lfp = self.lfp_reps_by_stim['raw'][stim_id]
            ntrials,nelectrodes,nt = lfp.shape

            # zscore the LFP across time and electrodes
            lfp -= lfp_mean
            lfp /= lfp_std

            # bandpass the LFP by trial/electrode
            lfp_bp = np.zeros([ntrials, nbands, nelectrodes, nt])
            for k in range(ntrials):
                for n in range(nelectrodes):
                    for b,(low_freq, high_freq) in enumerate(bands):
                        if np.isinf(low_freq):
                            lfp_bp[k, b, n, :] = lowpass_filter(lfp[k, n, :], self.lfp_sample_rate, high_freq)
                        elif np.isinf(high_freq):
                            lfp_bp[k, b, n, :] = highpass_filter(lfp[k, n, :], self.lfp_sample_rate, low_freq)
                        else:
                            lfp_bp[k, b, n, :] = bandpass_filter(lfp[k, n, :], self.lfp_sample_rate, low_freq, high_freq)

            self.lfp_reps_by_stim['bandpass'][stim_id] = lfp_bp

    def attach_ica(self, ica_file):

        from zeebeez3.transforms.ica import ICALFPTransform

        # open up ICA file
        icat = ICALFPTransform.load(ica_file)
        X = icat.X.T
        ncomps,nt_full = X.shape

        self.lfp_rep_params['ica'] = {'ncomps':ncomps, 'file_name':ica_file}

        self.lfp_reps_by_stim['ica'] = dict()
        stim_ids = self.trial_df['stim_id'].unique()
        for stim_id in stim_ids:

            i = self.trial_data['stim_id'] == stim_id
            ntrials = i.sum()

            stim_dur = self.trial_df['stim_duration'][i].values[0]
            slice_dur = self.pre_stim_time + stim_dur + self.post_stim_time
            idur = int(self.lfp_sample_rate*slice_dur)

            ic_proj = np.zeros([ntrials, ncomps, idur])
            for k in range(ntrials):
                ii = (self.trial_df['stim_id'] == stim_id) & (self.trial_df['trial'] == k)
                assert ii.sum() == 1, "More than one trial for stim %d, trial %d" % (stim_id, k)
                stime = self.trial_df['start_time'][ii].values[0]
                si = int(self.lfp_sample_rate*stime)
                ei = si + idur

                ic_proj[k, :, :] = X[:, si:ei]

            self.lfp_reps_by_stim['ica'][stim_id] = ic_proj

    def shuffle_trials(self, rep_type='raw'):
        """ Shuffle trials so they no longer match up between electrodes. """

        assert rep_type in self.lfp_reps_by_stim, "No such rep type %s" % rep_type

        stim_ids = list(self.lfp_reps_by_stim[rep_type].keys())
        for stim_id in stim_ids:
            X = self.lfp_reps_by_stim[rep_type][stim_id]
            if len(X.shape) == 3:
                ntrials,nelectrodes,nt = X.shape
                t = list(range(ntrials))
                for n in range(nelectrodes):
                    np.random.shuffle(t)
                    X[:, n, :] = X[t, n, :]

            elif len(X.shape) == 4:
                ntrials,nbands,nelectrodes,nt = X.shape
                t = list(range(ntrials))
                for n in range(nelectrodes):
                    np.random.shuffle(t)
                    X[:, :, n, :] = X[t, :, n, :]

    def mean_subtract(self, rep_type='raw'):
        """ Subtract the trial averaged LFP from each trial. """

        assert rep_type in self.lfp_reps_by_stim, "No such rep type %s" % rep_type

        stim_ids = list(self.lfp_reps_by_stim[rep_type].keys())
        for stim_id in stim_ids:
            X = self.lfp_reps_by_stim[rep_type][stim_id]
            if len(X.shape) == 3:
                ntrials,nelectrodes,nt = X.shape
                for k in range(ntrials):
                    # take the mean of all trials except this one
                    i = np.ones(ntrials, dtype='bool')
                    i[k] = False
                    lfp_mean = X[i, :, :].mean(axis=0)
                    X[k, :, :] -= lfp_mean

            elif len(X.shape) == 4:
                ntrials,nbands,nelectrodes,nt = X.shape
                for k in range(ntrials):
                    # take the mean of all trials except this one
                    i = np.ones(ntrials, dtype='bool')
                    i[k] = False
                    lfp_mean = X[i, :, :, :].mean(axis=0)
                    X[k, :, :, :] -= lfp_mean

    def mean_only(self, rep_type='raw'):
        """ Replace the per-trial LFP with the trial-averaged LFP. """

        assert rep_type in self.lfp_reps_by_stim, "No such rep type %s" % rep_type

        stim_ids = list(self.lfp_reps_by_stim[rep_type].keys())
        for stim_id in stim_ids:
            X = self.lfp_reps_by_stim[rep_type][stim_id]
            if len(X.shape) == 3:
                ntrials,nelectrodes,nt = X.shape
                for k in range(ntrials):
                    # take the mean of all trials except this one
                    i = np.ones(ntrials, dtype='bool')
                    i[k] = False
                    lfp_mean = X[i, :, :].mean(axis=0)
                    X[k, :, :] = lfp_mean

            elif len(X.shape) == 4:
                ntrials,nbands,nelectrodes,nt = X.shape
                for k in range(ntrials):
                    # take the mean of all trials except this one
                    i = np.ones(ntrials, dtype='bool')
                    i[k] = False
                    lfp_mean = X[i, :, :, :].mean(axis=0)
                    X[k, :, :, :] = lfp_mean

    def zscore(self, rep_type='raw'):
        """ Z-scores each electrode individually per stimulus presentation. """

        assert rep_type in self.lfp_reps_by_stim, "No such rep type %s" % rep_type

        # first compute the mean and std per electrode across all stim ids and trials
        mean_and_std = dict()
        stim_ids = list(self.lfp_reps_by_stim[rep_type].keys())
        for n,e in enumerate(self.index2electrode):
            x = list()
            for stim_id in stim_ids:
                X = self.lfp_reps_by_stim[rep_type][stim_id]
                assert len(X.shape) == 3, "z-scoring only works with rep_type=raw or rep_type=ica right now..."
                ntrials,nelectrodes,nt = X.shape
                for k in range(ntrials):
                    x.extend(X[k, n, :])
            x = np.array(x)
            mean_and_std[n] = (x.mean(), x.std(ddof=1))
            del x

        # subtract the mean and standard deviation from each trace
        for stim_id in stim_ids:
            X = self.lfp_reps_by_stim[rep_type][stim_id]
            assert len(X.shape) == 3, "z-scoring only works with rep_type=raw or rep_type=ica right now..."
            ntrials,nelectrodes,nt = X.shape
            for k in range(ntrials):
                for n in range(nelectrodes):
                    xmean,xstd = mean_and_std[n]
                    X[k, n, :] -= xmean
                    X[k, n, :] /= xstd

    def segment_stims_from_biosound(self, bs_file):
        from zeebeez3.transforms.biosound import BiosoundTransform

        bst = BiosoundTransform.load(bs_file)
        self.segment_data = bst.stim_data
        self.segment_df = bst.stim_df

    def segment_stims(self, plot=False):
        """ Segment stimulus spectrograms by syllable. """

        stim_ids = self.trial_df['stim_id'].unique()
        stim_classes_to_segment = ['DC', 'Te', 'Ne', 'LT', 'Th']
        segment_data = {'stim_id':list(), 'stim_type':list(), 'start_time':list(), 'end_time':list(), 'order':list()}

        for stim_id in stim_ids:
            i = self.trial_df['stim_id'] == stim_id
            stim_type = self.trial_df['stim_type'][i].values[0]
            stim_dur = self.trial_df['stim_duration'][i].values[0]

            if stim_type not in stim_classes_to_segment:
                segment_data['stim_id'].append(stim_id)
                segment_data['stim_type'].append(stim_type)
                segment_data['start_time'].append(0)
                segment_data['end_time'].append(stim_dur)
                segment_data['order'].append(0)
                continue

            # get stim RMS amplitude
            spec_rms = self.spec_rms_by_stim[stim_id]

            # segment the amplitude
            minimum_isi = int(0.020*self.lfp_sample_rate)
            thresh = np.percentile(spec_rms, 2)
            syllable_times = break_envelope_into_events(spec_rms, threshold=thresh, merge_thresh=minimum_isi)

            # store the syllable start/end times
            for k,(si,ei,max_amp) in enumerate(syllable_times):
                stime = si / self.lfp_sample_rate
                etime = ei / self.lfp_sample_rate

                segment_data['stim_id'].append(stim_id)
                segment_data['stim_type'].append(stim_type)
                segment_data['start_time'].append(stime)
                segment_data['end_time'].append(etime)
                segment_data['order'].append(k)

        self.segment_data = segment_data
        self.segment_df = pd.DataFrame(segment_data)

        if not plot:
            return

        # compute the inter-segment interval times
        all_isi = list()
        for stim_id in self.segment_df['stim_id'].unique():
            i = self.segment_df['stim_id'] == stim_id
            if i.sum() == 1:
                continue

            lst = list(zip(self.segment_df['start_time'][i].values,
                      self.segment_df['end_time'][i].values,
                      self.segment_df['order'][i].values))
            lst.sort(key=operator.itemgetter(2))

            lst = np.array(lst)
            isi = lst[1:, 0] - lst[:-1, 1]
            all_isi.extend(isi)
        all_isi = np.array(all_isi)

        # plot a histogram of segment times and inter-segment intervals
        seg_durs = self.segment_df['end_time'] - self.segment_df['start_time']
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        plt.hist(seg_durs, bins=30, color='m')
        ng = np.sum(seg_durs < 0.090) / float(len(seg_durs))
        plt.title('Syllable Durations (min=%0.3f, max=%0.3f, mean=%0.3f +/- %0.3f, median=%0.3f, p10=%0.3f, %%<90ms=%0.2f)' %
                  (seg_durs.min(), seg_durs.max(), seg_durs.mean(), seg_durs.std(ddof=1), np.median(seg_durs), np.percentile(seg_durs, 10), ng))
        plt.axis('tight')

        ax = plt.subplot(2, 1, 2)
        plt.hist(all_isi, bins=30, color='g')
        plt.title('Inter-syllable Interval Durations (min=%0.3f, max=%0.3f, mean=%0.3f +/- %0.3f, median=%0.3f)' %
                  (all_isi.min(), all_isi.max(), all_isi.mean(), all_isi.std(ddof=1), np.median(all_isi)))

        # make some more plots
        plist = list()
        for stim_id in stim_ids:

            spec = self.spec_by_stim[stim_id]

            rms = self.spec_rms_by_stim[stim_id]
            rms -= rms.min()
            rms /= rms.max()

            i = self.segment_df['stim_id'] == stim_id
            stim_type = self.segment_df['stim_type'][i].values[0]
            segment_times = list(zip(self.segment_df['start_time'][i].values, self.segment_df['end_time'][i].values))

            plist.append({'spec':spec, 'rms':rms, 'stim_type':stim_type, 'stim_id':stim_id, 'times':segment_times})

        # sort by stim type
        plist.sort(key=operator.itemgetter('stim_type'))

        # make a multi-plot of the data
        def _plot_stim(pdata, ax):
            plt.sca(ax)
            ax.set_axis_bgcolor('black')
            spec_t = np.arange(0, pdata['spec'].shape[1]) / self.lfp_sample_rate
            plot_spectrogram(spec_t, self.spec_freq, pdata['spec'], ax=ax, ticks=True, fmin=300.0, fmax=8000.0,
                             colormap=plt.cm.afmhot, colorbar=False)
            plt.plot(spec_t, pdata['rms']*self.spec_freq.max(), 'c-', linewidth=1.0)

            for stime,etime in pdata['times']:
                plt.axvline(stime, c='g', linewidth=1.5)
                plt.axvline(etime, c='b', linewidth=1.5)
            plt.axis('tight')
            plt.title('%d (%s)' % (pdata['stim_id'], pdata['stim_type']))

        multi_plot(plist, _plot_stim, nrows=4, ncols=3)

    def save(self, output_file, overwrite=False):

        if not os.path.exists(output_file):
            overwrite = True

        mode = 'a'
        if overwrite:
            mode = 'w'

        hf = h5py.File(output_file, mode)

        if overwrite:
            # write the StimEvent attributes

            hf.attrs['experiment_file'] = self.experiment_file
            hf.attrs['stimulus_file'] = self.stimulus_file
            hf.attrs['rcg_names'] = self.rcg_names

            hf.attrs['bird'] = self.bird
            hf.attrs['block_name'] = self.block_name
            hf.attrs['segment_name'] = self.segment_name
            hf.attrs['seg_uname'] = self.seg_uname

            hf.attrs['lfp_sample_rate'] = self.lfp_sample_rate

            hf.attrs['pre_stim_time'] = self.pre_stim_time
            hf.attrs['post_stim_time'] = self.post_stim_time

            hf.attrs['ncells'] = self.ncells
            hf.attrs['sort_code'] = self.sort_code
            hf.attrs['spec_freq'] = self.spec_freq
            hf.attrs['index2electrode'] = self.index2electrode

            # write the trial data
            grp = hf.create_group('trial_data')
            col_names = list(self.trial_data.keys())
            grp.attrs['col_names'] = col_names
            for cname in list(self.trial_data.keys()):
                grp[cname] = np.array(self.trial_data[cname])

            # write the electrode data
            grp = hf.create_group('electrode_data')
            col_names = list(self.electrode_data.keys())
            grp.attrs['col_names'] = col_names
            for cname in list(self.electrode_data.keys()):
                grp[cname] = np.array(self.electrode_data[cname])

            # write the cell data
            grp = hf.create_group('cell_data')
            col_names = list(self.cell_data.keys())
            grp.attrs['col_names'] = col_names
            for cname in list(self.cell_data.keys()):
                grp[cname] = np.array(self.cell_data[cname])

        # save representation data
        rep_types = list(self.lfp_reps_by_stim.keys())
        if overwrite:
            grp = hf.create_group('rep_params')
        else:
            grp = hf['rep_params']
            old_rep_types = grp.attrs['rep_types']
            rep_types = np.unique(np.r_[rep_types, old_rep_types])

        grp.attrs['rep_types'] = rep_types
        for rtype in rep_types:

            # only write params for new representations
            if rtype in grp:
                continue

            # create a group, even if there are no parameters for the representation
            rgrp = grp.create_group(rtype)

            if rtype not in self.lfp_rep_params:
                continue

            for aname,aval in list(self.lfp_rep_params[rtype].items()):
                rgrp.attrs[aname] = aval

        stim_ids = self.trial_df['stim_id'].unique()
        # write the stimulus conditioned data out, one group per stim
        for stim_id in stim_ids:
            if overwrite:
                grp = hf.create_group('stim%d' % stim_id)
                grp.attrs['id'] = stim_id

                # save the spectrogram
                grp['spec'] = self.spec_by_stim[stim_id]
                grp['spec_rms'] = self.spec_rms_by_stim[stim_id]

                # save the representations
                rep_grp = grp.create_group('representations')
                for rtype in rep_types:
                    rep_grp[rtype] = self.lfp_reps_by_stim[rtype][stim_id]

                # save the spikes
                spikes = self.spikes_by_stim[stim_id]
                ntrials = len(spikes)
                ncells = len(spikes[0])
                spikes_grp = grp.create_group('spikes')
                spikes_grp.attrs['ncells'] = ncells
                spikes_grp.attrs['ntrials'] = ntrials
                for k in range(ntrials):
                    tgrp = spikes_grp.create_group('trial%d' % k)
                    tgrp.attrs['trial'] = k
                    for n in range(ncells):
                        cgrp = tgrp.create_group('cell%d' % n)
                        cgrp.attrs['index'] = n
                        cgrp['spikes'] = spikes[k][n]
            else:
                grp = hf['stim%d' % stim_id]
                # only save new representations
                rep_grp = grp['representations']
                for rtype in rep_types:
                    if rtype not in rep_grp:
                        rep_grp[rtype] = self.lfp_reps_by_stim[rtype][stim_id]

        hf.close()

    @classmethod
    def load(clz, output_file, rep_types_to_load=None):

        se = StimEventTransform()
        se.file_name = output_file

        hf = h5py.File(output_file, 'r')
        se.experiment_file = hf.attrs['experiment_file']
        se.stimulus_file = hf.attrs['stimulus_file'] 
        se.rcg_names = hf.attrs['rcg_names'] 

        se.bird = hf.attrs['bird']
        se.block_name = hf.attrs['block_name']
        se.segment_name = hf.attrs['segment_name'] 
        se.seg_uname = hf.attrs['seg_uname']

        se.lfp_sample_rate = hf.attrs['lfp_sample_rate'] 

        se.pre_stim_time = hf.attrs['pre_stim_time'] 
        se.post_stim_time = hf.attrs['post_stim_time'] 

        se.ncells = hf.attrs['ncells'] 
        se.sort_code = hf.attrs['sort_code'] 
        se.spec_freq = hf.attrs['spec_freq']
        se.index2electrode = list(hf.attrs['index2electrode'])

        # read the trial data
        grp = hf['trial_data']
        col_names = grp.attrs['col_names']
        se.trial_data = dict()
        for cname in col_names:
            se.trial_data[decode_if_bytes(cname)] = np.array(grp[cname])
        se.trial_df = pd.DataFrame(se.trial_data)
        decode_column_if_bytes(se.trial_df)
        
        # read the electrode data
        grp = hf['electrode_data']
        col_names = grp.attrs['col_names']
        se.electrode_data = dict()
        for cname in col_names:
            se.electrode_data[decode_if_bytes(cname)] = np.array(grp[cname])
        se.electrode_df = pd.DataFrame(se.electrode_data)
        decode_column_if_bytes(se.electrode_df)
        
        # read the cell data
        grp = hf['cell_data']
        col_names = grp.attrs['col_names']
        se.cell_data = dict()
        for cname in col_names:
            se.cell_data[decode_if_bytes(cname)] = np.array(grp[cname])
        se.cell_df = pd.DataFrame(se.cell_data)
        decode_column_if_bytes(se.cell_df)

        # read representation data
        grp = hf['rep_params']
        se.lfp_rep_params = dict()
        rep_types = grp.attrs['rep_types']
        for rtype in rep_types:
            se.lfp_rep_params[decode_if_bytes(rtype)] = dict()
            if rtype in grp:
                rgrp = grp[rtype]
                for aname,aval in list(rgrp.attrs.items()):
                    se.lfp_rep_params[decode_if_bytes(rtype)][decode_if_bytes(aname)] = aval

        # read per-stim data
        if rep_types_to_load is None:
            rep_types_to_load = [decode_if_bytes(s) for s in rep_types]

        stim_ids = se.trial_df['stim_id'].unique()
        se.spec_by_stim = dict()
        se.spec_rms_by_stim = dict()
        se.lfp_reps_by_stim = dict()
        se.spikes_by_stim = dict()
        for rtype in rep_types_to_load:
            se.lfp_reps_by_stim[rtype] = dict()

        for stim_id in stim_ids:
            grp = hf['stim%d' % stim_id]

            # load stimulus spectrogram and amplitude envelope
            se.spec_by_stim[stim_id] = np.array(grp['spec'])
            se.spec_rms_by_stim[stim_id] = np.array(grp['spec_rms'])

            # load representations
            for rtype in rep_types_to_load:
                se.lfp_reps_by_stim[rtype][stim_id] = np.array(grp['representations'][rtype])

            # load spikes
            sgrp = grp['spikes']
            ntrials = sgrp.attrs['ntrials']
            ncells = sgrp.attrs['ncells']

            spikes = list()
            for k in range(ntrials):
                tgrp = sgrp['trial%d' % k]
                trial_spikes = list()
                for n in range(ncells):
                    cgrp = tgrp['cell%d' % n]
                    trial_spikes.append(np.array(cgrp['spikes']))
                spikes.append(trial_spikes)
            se.spikes_by_stim[stim_id] = spikes

        hf.close()

        return se

    def itertrials(self, exclude_types=('mlnoise', 'wnoise'), pre_stim_duration=None, lfp_delay=0.005, rep_type='raw',
                   all_trials=False, electrodes=None):
        if self.segment_df is None:
            self.segment_stims()
        return TrialIterator(self, exclude_types, pre_stim_duration, lfp_delay, rep_type, all_trials=all_trials,
                             electrodes=electrodes)

    def plot(self, output_dir='/auto/tdrive/mschachter/figures/stim_event', with_spikes=True, trial_by_trial=True,
             rep_type='raw'):

        stim_ids = self.trial_df['stim_id'].unique()

        # compute the mean and std of the LFP
        all_lfps = list()
        for stim_id in stim_ids:
            lfp = self.lfp_reps_by_stim[rep_type][stim_id]
            all_lfps.extend(lfp.ravel())
        lfp_mean = np.mean(all_lfps)
        lfp_std = np.std(all_lfps, ddof=1)
        del all_lfps

        # redefine electrode order
        lst = list(zip(self.electrode_df['electrode'].values, self.electrode_df['row'].values,
                  self.electrode_df['col'].values, self.electrode_df['hemisphere'].values))
        lst.sort(key=operator.itemgetter(3))
        lst.sort(key=operator.itemgetter(2))
        lst.sort(key=operator.itemgetter(1))
        electrode_order = [x[0] for x in lst]

        for stim_id in stim_ids:

            i = self.trial_df['stim_id'] == stim_id
            assert i.sum() > 0, "Cannot locate data for stim %d" % stim_id
            stim_type = self.trial_df['stim_type'][i].values[0]

            # get the LFP representation
            lfp = self.lfp_reps_by_stim[rep_type][stim_id]
            if rep_type in ('raw', 'ica'):
                ntrials, nelectrodes, nt = lfp.shape
                lfp = lfp.reshape([ntrials, 1, nelectrodes, nt])

            # reshape the LFP representation according to electrode order
            lfp_reshaped = np.zeros_like(lfp)
            for k,e in enumerate(electrode_order):
                n = self.index2electrode.index(e)
                lfp_reshaped[:, :, k, :] = lfp[:, :, n, :]
            lfp = lfp_reshaped
            ntrials, nbands, nelectrodes, nt = lfp.shape

            # zscore the LFP representation
            lfp -= lfp_mean
            lfp /= lfp_std

            absmax = np.abs(lfp.ravel()).max()

            # construct a stimulus spectrogram that includes silent periods
            i = self.trial_df['stim_id'] == stim_id
            stim_dur = self.trial_df['stim_duration'][i].values[0]
            spec = self.spec_by_stim[stim_id]

            # the duration of spec is greater than or equal to stim_dur, so compute the difference
            # and take that into account
            spec_dur = spec.shape[1] / self.lfp_sample_rate
            excess_dur = spec_dur - stim_dur
            assert excess_dur >= 0, "Excess duration is negative! stim_dur=%0.6fs, spec_dur=%0.6fs, excess_dur=%0.6fs" % \
                                    (stim_dur, spec_dur, excess_dur)

            full_duration = self.pre_stim_time + stim_dur + self.post_stim_time
            idur = int(self.lfp_sample_rate*full_duration)
            full_spec = np.zeros([len(self.spec_freq), idur])
            si = int(self.lfp_sample_rate*self.pre_stim_time)
            ei = min(si + spec.shape[1], full_spec.shape[1])
            print('stim_dur=%0.6fs, full_duration=%0.6fs, spec.shape=%s, idur=%d, full_spec.shape=%s, si=%d, ei=%d' % \
                  (stim_dur, full_duration, spec.shape, idur, full_spec.shape, si, ei))
            full_spec[:, si:ei] = spec[:, :(ei-si)]
            spec_t = np.linspace(-self.pre_stim_time, full_duration-self.pre_stim_time, idur)

            # get the multi-electrode spike trains
            spikes_by_trial = self.spikes_by_stim[stim_id]
            assert len(spikes_by_trial) == ntrials, "Something is weird about spikes_by_trial :("

            # determine an ordering for the cell indices
            lst = list()
            for ci,electrode,hemi in zip(self.cell_df['index'].values, self.cell_df['electrode'].values, self.cell_df['hemisphere'].values):
                # get row and column of electrode
                i = self.electrode_df['electrode'] == electrode
                row = self.electrode_df['row'][i].values[0]
                col = self.electrode_df['col'][i].values[0]
                lst.append( (ci, electrode, row, col, hemi))
            lst.sort(key=operator.itemgetter(4))
            lst.sort(key=operator.itemgetter(3))
            lst.sort(key=operator.itemgetter(2))
            plotindex2cellindex = [x[0] for x in lst]

            # compute the PSTH across trials
            bin_size = 0.001
            psth = list()
            cell_psth_t = None
            for nn in range(self.ncells):
                n = plotindex2cellindex[nn]
                cell_spikes_by_trial = [spikes_by_trial[trial_num][n] for trial_num in range(ntrials)]
                cell_psth_t, cell_psth = compute_psth(cell_spikes_by_trial, full_duration, bin_size=bin_size,
                                                      time_offset=-self.pre_stim_time)
                psth.append(cell_psth)
            # min_psth_len = np.min([len(r) for r in psth])
            # psth = np.array([r[:min_psth_len] for r in psth])
            psth = np.array(psth)

            # get the stimulus averaged LFPs
            stim_avg_lfps = list()
            for band in range(nbands):
                band_lfp = lfp[:, band, :, :].squeeze()

                """
                # normalize the amplitude of each electrode individually
                for n in range(nelectrodes):
                    # get max amp across all trials and time points
                    max_amp = np.abs(band_lfp[:, n, :]).max()
                    band_lfp[:, n, :] /= max_amp
                """

                band_lfp_mean = band_lfp.mean(axis=0)
                stim_avg_lfps.append(band_lfp_mean)

            if with_spikes:
                # make a figure for the spikes
                fig = plt.figure(figsize=(24.0, 13.5), facecolor='gray')
                plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

                # plot spectrogram
                nrows = ntrials + 2

                ax = plt.subplot(nrows, 1, 1)
                ax.set_axis_bgcolor('black')
                plot_spectrogram(spec_t, self.spec_freq, full_spec, ax=ax, colormap=plt.cm.afmhot, colorbar=False, fmax=8000.0)
                plt.axvline(stim_dur, c='w')
                plt.axis('tight')
                plt.ylabel('')
                plt.yticks([])

                # plot the spike raster for each trial
                for trial_num in range(ntrials):
                    spikes = spikes_by_trial[trial_num]
                    ax = plt.subplot(nrows, 1, trial_num+2)
                    plot_raster(spikes, ax=ax, duration=full_duration, bin_size=bin_size, ylabel='Trial %d' % trial_num,
                                bgcolor='black', spike_color='r', time_offset=-self.pre_stim_time)

                # plot the PSTH
                ax = plt.subplot(nrows, 1, ntrials+2)
                ax.set_axis_bgcolor('black')
                plt.imshow(psth, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, vmin=0.0, vmax=1.0,
                           extent=[cell_psth_t.min(), cell_psth_t.max(), 0, self.ncells])
                plt.ylabel('PSTH')
                plt.axis('tight')
                fname = os.path.join(output_dir, 'spikes_%s_stim%d.png' % (stim_type, stim_id))
                print('Writing file %s' % fname)
                plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')

            # make a figure for the stimulus averaged response
            fig = plt.figure(figsize=(24.0, 13.5), facecolor='gray')
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

            # plot spectrogram
            nrows = nbands + 2
            ax = plt.subplot(nrows, 1, 1)
            ax.set_axis_bgcolor('black')
            plot_spectrogram(spec_t, self.spec_freq, full_spec, ax=ax, colormap=plt.cm.afmhot, colorbar=False, fmax=8000.0)
            plt.axis('tight')
            plt.ylabel('')
            plt.yticks([])

            # plot the PSTH
            ax = plt.subplot(nrows, 1, 2)
            ax.set_axis_bgcolor('black')
            plt.imshow(psth, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                       vmin=0.0, vmax=1.0, extent=[spec_t.min(), spec_t.max(), 0, self.ncells])
            plt.ylabel('PSTH')
            plt.axis('tight')

            # plot each band
            for n in range(nbands):
                stim_avg_lfp = stim_avg_lfps[n]

                ax = plt.subplot(nrows, 1, n + 3)
                plt.imshow(stim_avg_lfp, interpolation='nearest', aspect='auto', origin='upper', cmap=plt.cm.seismic,
                           extent=[-self.pre_stim_time, full_duration-self.pre_stim_time, 1, nelectrodes],
                           vmin=-1, vmax=1)
                plt.axis('tight')
                plt.ylabel('Band %d' % n)

            fname = os.path.join(output_dir, '%s_%s_stim%d_avg.png' % (rep_type, stim_type, stim_id))
            print('Writing file %s' % fname)
            plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close('all')

            if not trial_by_trial:
                continue

            # make a figure for each LFP band
            for band in range(nbands):

                band_lfp = lfp[:, band, :, :].squeeze()

                # normalize the amplitude of each electrode individually
                for n in range(nelectrodes):
                    # get max amp across all trials and time points
                    max_amp = np.abs(band_lfp[:, n, :]).max()
                    band_lfp[:, n, :] /= max_amp

                fig = plt.figure(figsize=(24.0, 13.5), facecolor='gray')
                plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

                # plot spectrogram
                nrows = ntrials + 2

                ax = plt.subplot(nrows, 1, 1)
                ax.set_axis_bgcolor('black')
                plot_spectrogram(spec_t, self.spec_freq, full_spec, ax=ax, colormap=plt.cm.afmhot, colorbar=False,
                                 fmax=8000.0)
                plt.axis('tight')
                plt.ylabel('')
                plt.yticks([])

                # plot each trial for this band
                for trial_num in range(ntrials):
                    z = band_lfp[trial_num, :, :].squeeze()

                    ax = plt.subplot(nrows, 1, trial_num+2)
                    im = plt.imshow(z, interpolation='nearest', aspect='auto', origin='upper', cmap=plt.cm.seismic,
                                    extent=[-self.pre_stim_time, full_duration-self.pre_stim_time, 1, nelectrodes],
                                    vmin=-1, vmax=1)
                    plt.axis('tight')
                    plt.ylabel('Trial %d' % trial_num)

                # plot the stimulus-averaged lfp
                stim_avg_lfp = band_lfp.mean(axis=0)
                t = np.arange(stim_avg_lfp.shape[1]) / self.lfp_sample_rate
                ax = plt.subplot(nrows, 1, ntrials+2)
                plt.imshow(stim_avg_lfp, interpolation='nearest', aspect='auto', origin='upper', cmap=plt.cm.seismic,
                           extent=[-self.pre_stim_time, full_duration-self.pre_stim_time, 1, nelectrodes],
                           vmin=-1, vmax=1)
                plt.axis('tight')
                plt.ylabel('Mean')

                fname = os.path.join(output_dir, '%s_%s_stim%d_zband%d.png' % (rep_type, stim_type, stim_id, band))
                print('Writing file %s' % fname)
                plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')


class TrialIterator(object):

    def __init__(self, stim_event, exclude_types=('mlnoise', 'wnoise'), pre_stim_duration=0.250, lfp_delay=0.005,
                 rep_type='raw', all_trials=False, electrodes=None):

        assert isinstance(stim_event, StimEventTransform)

        # build a list of trials, and their respective indices
        self.trial_list = list()
        self.current_index = 0
        self.stim_event = stim_event
        self.rep_type = rep_type
        self.all_trials = all_trials
        self.electrode_indices = np.ones([len(stim_event.index2electrode)], dtype='bool')
        if electrodes is not None:
            self.electrode_indices = np.zeros([len(stim_event.index2electrode)], dtype='bool')
            for e in electrodes:
                self.electrode_indices[stim_event.index2electrode.index(e)] = True

        for jj,row in stim_event.segment_df.iterrows():
            stim_id = row['stim_id']
            stim_type = row['stim_type']

            if stim_type in exclude_types:
                continue

            order = row['order']
            stime = row['start_time']
            etime = row['end_time']

            lfp_stime = stim_event.pre_stim_time + stime + lfp_delay
            lfp_etime = stim_event.pre_stim_time + etime + lfp_delay

            i = stim_event.trial_data['stim_id'] == stim_id
            ntrials = i.sum()

            if all_trials:
                new_stim_id = '%d_%d' % (stim_id, order)
                tdata = {'stim_id':new_stim_id, 'stim_type':stim_type, 'stim_stime':stime, 'stim_etime':etime,
                         'lfp_stime':lfp_stime, 'lfp_etime':lfp_etime, 'trial':list(range(ntrials)), 'base_stim_id':stim_id}
                self.trial_list.append(tdata)

                if pre_stim_duration is not None:
                    # grab a sample of the LFP prior to stim presentation
                    tdata_pre = {'stim_id':new_stim_id, 'stim_type':'pre', 'lfp_stime':0, 'lfp_etime':pre_stim_duration,
                                 'trial':list(range(ntrials)), 'base_stim_id':stim_id}
                    self.trial_list.append(tdata_pre)

            else:
                for k in range(ntrials):
                    new_stim_id = '%d_%d' % (stim_id, order)
                    tdata = {'stim_id':new_stim_id, 'stim_type':stim_type, 'stim_stime':stime, 'stim_etime':etime,
                             'lfp_stime':lfp_stime, 'lfp_etime':lfp_etime, 'trial':k, 'base_stim_id':stim_id}
                    self.trial_list.append(tdata)

                    if pre_stim_duration is not None:
                        # grab a sample of the LFP prior to stim presentation
                        tdata_pre = {'stim_id':new_stim_id, 'stim_type':'pre', 'lfp_stime':0, 'lfp_etime':pre_stim_duration,
                                     'trial':k, 'base_stim_id':stim_id}
                        self.trial_list.append(tdata_pre)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.trial_list):
            raise StopIteration
        else:
            self.current_index += 1
            tdata = self.trial_list[self.current_index-1]

            base_stim_id = tdata['base_stim_id']

            # grab the spectrogram for this slice
            spec = None
            if tdata['stim_type'] != 'pre':
                stim_si = int(tdata['stim_stime']*self.stim_event.lfp_sample_rate)
                stim_ei = int(tdata['stim_etime']*self.stim_event.lfp_sample_rate)
                spec = self.stim_event.spec_by_stim[base_stim_id][:, stim_si:stim_ei]

            # grab the LFP across electrodes/(bands) for this slice
            lfp = self.stim_event.lfp_reps_by_stim[self.rep_type][base_stim_id]
            lfp_si = int(tdata['lfp_stime']*self.stim_event.lfp_sample_rate)
            lfp_ei = int(tdata['lfp_etime']*self.stim_event.lfp_sample_rate)
            if self.rep_type in ['memd', 'bandpass']:
                X = lfp[tdata['trial'], :, :, lfp_si:lfp_ei]
                X = X[:, :, self.electrode_indices, :]
            elif self.rep_type == 'wavelet':
                increment = (1.0 / self.stim_event.lfp_sample_rate)*2
                if 'increment' in self.stim_event.lfp_rep_params['wavelet']:
                    increment = self.stim_event.lfp_rep_params['wavelet']['increment']
                lfp_si = int(tdata['lfp_stime']/increment)
                lfp_ei = int(tdata['lfp_etime']/increment)
                X = lfp[tdata['trial'], :, :, lfp_si:lfp_ei]
                X = X[:, :, self.electrode_indices, :]
                nbands,nelectrodes,nt = X.shape
                if nt == 0:
                    print('lfp.shape=',lfp.shape)
                    print('X.shape=',X.shape)
                    print('increment=',increment)
                    print('lfp_si=%d, lfp_ei=%d' % (lfp_si, lfp_ei))
                    raise Exception("nt is zero!")
            else:
                X = lfp[tdata['trial'], :, lfp_si:lfp_ei]
                X = X[:, self.electrode_indices, :]
                if not self.all_trials:
                    nelectrodes,nt = X.shape
                    X = X.reshape([1, nelectrodes, nt])
                else:
                    ntrials,nelectrodes,nt = X.shape
                    X = X.reshape([1, ntrials, nelectrodes, nt])

            iter_data = (tdata['stim_id'], tdata['stim_type'], tdata['trial'], spec, X)

            return iter_data


if __name__ == '__main__':

    exp_name = 'GreBlu9508M'
    data_dir = '/auto/tdrive/mschachter/data/%s' % exp_name
    exp_file = os.path.join(data_dir, '%s.h5' % exp_name)
    stim_file = os.path.join(data_dir, 'stims.h5')
    output_dir = os.path.join(data_dir, 'transforms')

    start_time = None
    end_time = None
    hemis = ['L']

    block_name = 'Site4'
    segment_name = 'Call1'

    file_ext = '%s_%s_%s_%s' % (exp_name, block_name, segment_name, ','.join(hemis))

    # experiment = Experiment.load(exp_file, stim_file)
    # segment = experiment.get_segment(block_name, segment_name)

    sefile = os.path.join(output_dir, 'StimEvent_%s.h5' % file_ext)

    """
    se = StimEventTransform()
    se.set_segment(experiment, segment, hemis, pre_stim_time=1.0, post_stim_time=1.0)
    se.attach_raw(experiment, segment, sort_code='single')
    se.save(sefile)
    del se

    se = StimEventTransform.load(sefile, rep_types_to_load=('raw',))
    se.attach_bandpass()
    se.save(sefile, overwrite=False)
    del se

    se = StimEventTransform.load(sefile, rep_types_to_load=[])
    memd_file = os.path.join(output_dir, 'MEMD_%s.h5' % file_ext)
    se.attach_memd(memd_file)
    se.save(sefile, overwrite=False)
    del se

    se = StimEventTransform.load(sefile, rep_types_to_load=[])
    ica_file = os.path.join(output_dir, 'ICA_%s.h5' % file_ext)
    se.attach_ica(ica_file)
    se.save(sefile, overwrite=False)
    del se
    """

    """
    se = StimEventTransform.load(sefile, rep_types_to_load=['raw'])
    se.attach_wavelet()
    se.save(sefile, overwrite=False)
    del se
    """

    # bs_file = os.path.join(output_dir, 'biosounds.h5')
    se = StimEventTransform.load(sefile, rep_types_to_load=('raw',))
    # se.segment_stims_from_biosound(bs_file)
    # se.segment_stims(plot=True)
    # plt.show()
    # se.plot(with_spikes=True, trial_by_trial=True, rep_type='raw',
    #         output_dir='/auto/tdrive/mschachter/figures/stim_event/GreBlu9508M/Site4_Call1_L')

    # strf_preproc_file = os.path.join(data_dir, 'preprocess', 'cell_strfs_%s.h5' % file_ext)
    # se.export_cells_for_strfs(strf_preproc_file)



