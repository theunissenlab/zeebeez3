import os
import h5py
from matplotlib import cm
from matplotlib.colors import LogNorm
from soundsig.sound import plot_spectrogram
import numpy as np

from scipy.ndimage import convolve1d

import matplotlib.pyplot as plt

from soundsig.signal import break_envelope_into_events, find_extrema
from soundsig.spikes import plot_raster, spike_envelope

from zeebeez3.core.experiment import Experiment, segment_to_unique_name, segment_from_unique_name


class SpikeEventTransform(object):
    """ Breaks a multielectrode spike train into events with well defined start and end times, and
        quantifies those events.
    """

    def __init__(self):
        self.experiment = None
        self.rcg_names = None
        self.bin_size = None
        self.sort_code = None
        self.events = dict()
        self.envelopes = dict()
        self.spec_envelopes = dict()
        self.epoch_tables = dict()

    def transform(self, experiment, bin_size=None, rcg_names=('L',), sort_code='single',
                  amp_thresh=0.050, duration_thresh=0.050, merge_thresh=0.030):
        """ Converts the spike trains in each segment to start and end times.
            :param experiment:
            :return:
        """

        assert isinstance(experiment, Experiment)

        self.experiment = experiment
        assert len(rcg_names) == 1, "Only one recording channel group is supported."
        self.rcg_names = rcg_names

        if bin_size is None:
            bin_size = 1.0 / self.experiment.lfp_sample_rate

        self.bin_size = bin_size
        self.sort_code = sort_code

        self.experiment.init_segment_spectrograms(0.007, sample_rate=1.0/self.bin_size, log=True)

        all_segments = self.experiment.get_all_segments()
        for seg in all_segments:
            seg_uname = segment_to_unique_name(seg)
            start_time = 0.0
            duration = seg.annotations['duration']

            # get the epoch table that lists the stimuli and their presentation times
            etable = self.experiment.epoch_table[seg_uname]
            edata = dict()
            edata['id'] = etable['id'].values.astype('int')
            edata['trial'] = etable['trial'].values.astype('int')
            edata['start_time'] = etable['start_time'].values.astype('float')
            edata['end_time'] = etable['end_time'].values.astype('float')
            self.epoch_tables[seg_uname] = edata

            print('Loading spike raster for segment %s' % seg_uname)

            # load up the spike raster for this time slice
            spike_rasters = self.experiment.get_spike_slice(seg, start_time, duration,
                                                            rcg_names=self.rcg_names, as_matrix=False,
                                                            sort_code=self.sort_code, bin_size=self.bin_size)
            spike_trains,spike_train_group = spike_rasters[self.rcg_names[0]]

            # compute the amplitude envelope for the population, threshold the signal
            spike_env = spike_envelope(spike_trains, start_time, duration, bin_size=self.bin_size, smooth=False,
                                       thresh_percentile=None)
            self.envelopes[seg_uname] = spike_env

            # segment into events
            merge_threshi = int(merge_thresh/self.bin_size)
            events = break_envelope_into_events(spike_env, threshold=0.0, merge_thresh=merge_threshi)

            threshold = True
            if threshold:
                durations = (events[:, 1] - events[:, 0])*self.bin_size
                amplitudes = events[:, -1]

                # threshold data based on joint amplitude and duration conditions
                di = (durations >= duration_thresh) & (amplitudes >= amp_thresh)
                print('Thresholding reduced # of events from %d to %d' % (len(durations), di.sum()))
                events = events[di, :]

            # convert units of events to seconds
            events[:, 0] *= self.bin_size
            events[:, 1] *= self.bin_size
            self.events[seg_uname] = events

            # compute the stimulus protocol amplitude envelope
            spec_t,spec_freq,spec = self.experiment.get_spectrogram_slice(seg, 0, duration)
            self.spec_envelopes[seg_uname] = spec.sum(axis=0)

            # clean up
            del spike_trains
            
    def plot(self, start_time, end_time):

        for seg_uname,events in list(self.events.items()):

            durations = events[:, 1] - events[:, 0]
            amplitudes = events[:, -1]
            event_start_times = events[:, 0]
            event_end_times = events[:, 1]

            inter_event_intervals = event_start_times[1:] - event_end_times[:-1]

            # print some event info
            print('Segment: %s' % seg_uname)
            print('\tIdentified %d events' % len(durations))
            print('\tDuration: min=%0.6f, max=%0.6f, mean=%0.6f, median=%0.6f' % (durations.min(), durations.max(), durations.mean(), np.median(durations)))
            print('\tIEI: min=%0.6f, max=%0.6f, mean=%0.6f, median=%0.6f' % (inter_event_intervals.min(), inter_event_intervals.max(), inter_event_intervals.mean(), np.median(inter_event_intervals)))
            print('\tAmplitude: min=%0.6f, max=%0.6f, mean=%0.6f, median=%0.6f' % (amplitudes.min(), amplitudes.max(), amplitudes.mean(), np.median(amplitudes)))

            # plot some event statistics
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.hist(durations, bins=100, color='r')
            plt.axis('tight')
            plt.title('Event Duration (s)')

            plt.subplot(3, 1, 2)
            plt.hist(inter_event_intervals, bins=100, color='k')
            plt.axis('tight')
            plt.title('Inter-event Interval (s)')

            plt.subplot(3, 1, 3)
            plt.hist(amplitudes, bins=100, color='g')
            plt.title('Peak Amplitude')
            plt.suptitle(seg_uname)
            plt.axis('tight')

            # plot event duration vs amplitude histogram
            plt.figure()
            plt.hist2d(durations, amplitudes, bins=[40, 30], cmap=cm.Greys, norm=LogNorm())
            plt.xlim(0, 2.)
            plt.xlabel('Duration')
            plt.ylabel('Amplitude')
            plt.colorbar(label="# of Joint Events")
            plt.suptitle(seg_uname)

            if self.experiment is not None:
                # get the segment
                seg = segment_from_unique_name(self.experiment.blocks, seg_uname)

                # get a slice of the spike raster
                spike_rasters = self.experiment.get_spike_slice(seg, start_time, end_time,
                                                                rcg_names=self.rcg_names, as_matrix=False,
                                                                sort_code=self.sort_code, bin_size=self.bin_size)
                spike_trains,spike_train_group = spike_rasters[self.rcg_names[0]]

                # get the spike envelope for this time slice
                si = int(start_time / self.bin_size)
                ei = int(end_time / self.bin_size)
                spike_env = self.envelopes[seg_uname][si:ei]

                # get the stim envelope for this time slice
                stim_env = self.spec_envelopes[seg_uname][si:ei]

                # get stimulus spectrogram
                stim_spec_t,stim_spec_freq,stim_spec = self.experiment.get_spectrogram_slice(seg, start_time, end_time)

                # compute the amplitude envelope for the spectrogram
                stim_spec_env = stim_envelope(stim_spec)

                plt.figure()
                plt.suptitle(seg_uname)

                # make a plot of the spectrogram
                ax = plt.subplot(3, 1, 1)
                ax.set_axis_bgcolor('black')
                plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.afmhot, colorbar=False, fmax=8000.0)
                plt.plot(stim_spec_t, stim_spec_env*stim_spec_freq.max(), 'w-', linewidth=3.0, alpha=0.75)
                plt.axis('tight')
                plt.ylabel('')
                plt.yticks([])

                # plot the stimulus amplitude envelope
                tenv = np.arange(len(spike_env))*self.bin_size + start_time
                ax = plt.subplot(3, 1, 2)
                ax.set_axis_bgcolor('black')
                plt.plot(tenv, stim_env, 'w-', linewidth=2.0)
                plt.axis('tight')

                # make a plot of the spike raster, the envelope, and the events
                ax = plt.subplot(3, 1, 3)
                plot_raster(spike_trains, ax=ax, duration=end_time-start_time, bin_size=self.bin_size,
                            time_offset=start_time, ylabel='', bgcolor='k', spike_color='#ff0000')
                plt.plot(tenv, spike_env*len(spike_trains), 'w-', alpha=0.75)

                # plot start events that are within plotting range
                sei = (event_start_times >= start_time) & (event_start_times <= end_time)
                plt.plot(event_start_times[sei], [1]*sei.sum(), 'g^', markersize=10)

                # plot end events that are within plotting range
                eei = (event_end_times >= start_time) & (event_end_times <= end_time)
                plt.plot(event_end_times[eei], [1]*eei.sum(), 'bv', markersize=10)

                plt.axis('tight')
                plt.yticks([])
                plt.ylabel('Spikes')

    def plot_envelopes(self, seg_uname, start_time, end_time):
        print(list(self.spec_envelopes.keys()))
        spec_env = self.spec_envelopes[seg_uname]
        spike_env = self.envelopes[seg_uname]

        spec_env /= spec_env.max()

        si = int(start_time/self.bin_size)
        ei = int(end_time/self.bin_size)

        t = np.arange(ei-si)*self.bin_size + start_time

        plt.figure()
        plt.plot(t, spec_env[si:ei], 'k-', linewidth=4.0, alpha=0.7)
        plt.plot(t, spike_env[si:ei], 'r-', linewidth=4.0, alpha=0.7)
        plt.axis('tight')
        plt.show()

    def save(self, output_file):
        hf = h5py.File(output_file, 'w')
        hf.attrs['sort_code'] = self.sort_code
        hf.attrs['bin_size'] = self.bin_size
        hf.attrs['rcg_names'] = self.rcg_names
        hf.attrs['experiment_file'] = self.experiment.file_name
        hf.attrs['stim_file'] = self.experiment.stimulus_file_name
        for seg_uname,spike_env in list(self.envelopes.items()):
            grp = hf.create_group(seg_uname)
            grp['envelope'] = spike_env
            grp['events'] = self.events[seg_uname]
            grp['stim_envelope'] = self.spec_envelopes[seg_uname]
            egrp = grp.create_group('epoch_table')
            for key,val in list(self.epoch_tables[seg_uname].items()):
                egrp[key] = val
        hf.close()

    @classmethod
    def load(clz, output_file, load_experiment=False):
        se = SpikeEventTransform()
        hf = h5py.File(output_file, 'r')
        se.experiment_file = hf.attrs['experiment_file']
        se.stim_file = hf.attrs['stim_file']
        se.bin_size = hf.attrs['bin_size']
        se.sort_code = hf.attrs['sort_code']
        se.rcg_names = hf.attrs['rcg_names']
        for seg_name in list(hf.keys()):
            se.envelopes[seg_name] = np.array(hf[seg_name]['envelope'])
            se.events[seg_name] = np.array(hf[seg_name]['events'])
            se.spec_envelopes[seg_name] = np.array(hf[seg_name]['stim_envelope'])
            edata = dict()
            egrp = hf[seg_name]['epoch_table']
            for key in list(egrp.keys()):
                edata[key] = np.array(egrp[key])
            se.epoch_tables[seg_name] = edata
        hf.close()

        if load_experiment:
            se.experiment = Experiment.load(se.experiment_file, se.stim_file)

        return se


def stim_envelope(the_matrix, log=False):
    tm_env = np.abs(the_matrix).sum(axis=0)
    tm_env -= tm_env.min()
    tm_env /= tm_env.max()

    if log:
        nz = tm_env > 0.0
        tm_env[nz] = np.log10(tm_env[nz])
        tm_env_thresh = -np.percentile(np.abs(tm_env[nz]), 95)
        tm_env[~nz] = tm_env_thresh
        tm_env[tm_env <= tm_env_thresh] = tm_env_thresh
        tm_env -= tm_env_thresh
        tm_env /= tm_env.max()

    return tm_env


if __name__ == '__main__':

    exp_name = 'GreBlu9508M'
    exp_file = '/auto/tdrive/mschachter/data/GreBlu9508M/%s.h5' % exp_name
    stim_file = '/auto/tdrive/mschachter/data/GreBlu9508M/stims.h5'
    output_dir = '/auto/tdrive/mschachter/data/GreBlu9508M/transforms'
    # exp = Experiment.load(exp_file, stim_file)

    start_time = None
    end_time = None
    hemis = ['L']

    output_file = os.path.join(output_dir, 'SpikeEventTransform_%s_%s.h5' % (exp_name, ','.join(hemis)))

    # sst = SpikeEventTransform()
    # sst.transform(exp)
    # sst.save(output_file)

    sst = SpikeEventTransform.load(output_file, load_experiment=False)
    # sst.plot(start_time=21.0, end_time=51.0)
    sst.plot_envelopes('GreBlu9508M_Site4_Call1', 21., 51.)
    plt.show()
