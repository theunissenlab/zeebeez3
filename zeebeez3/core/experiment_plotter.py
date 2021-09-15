import operator
import os

from soundsig.spikes import plot_raster
from matplotlib import cm
from soundsig.sound import plot_spectrogram

import numpy as np
import matplotlib.pyplot as plt

from soundsig.timefreq import gaussian_stft, log_spectrogram
from zeebeez3.core.experiment import segment_to_unique_name, Experiment


class ExperimentPlotter(object):

    def __init__(self, experiment):

        self.experiment = experiment

    def plot_segment_slice(self, seg, start_time, end_time, output_dir=None, sort_code='0'):
        """ Plot all the data for a given slice of time in a segment. This function will plot the spectrogram,
            multi-electrode spike trains, and LFPs for each array all on one figure.

        :param seg:
        :param start_time:
        :param end_time:
        :return:
        """

        num_arrays = len(seg.block.recordingchannelgroups)

        nrows = 1 + 2*num_arrays
        ncols = 1

        plt.figure()

        #get the slice of spectrogram to plot
        stim_spec_t,stim_spec_freq,stim_spec = self.experiment.get_spectrogram_slice(seg, start_time, end_time)

        #get the spikes to plot
        spikes = self.experiment.get_spike_slice(seg, start_time, end_time, sort_code=sort_code)

        #get the multielectrode LFPs for each channel
        lfps = self.experiment.get_lfp_slice(seg, start_time, end_time)

        #plot the spectrogram
        ax = plt.subplot(nrows, ncols, 1)
        plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.afmhot_r, colorbar=False, fmax=8000.0)
        plt.ylabel('')
        plt.yticks([])

        #plot the LFPs and spikes for each recording array
        for k,rcg in enumerate(seg.block.recordingchannelgroups):
            spike_trains,spike_train_groups = spikes[rcg.name]
            ax = plt.subplot(nrows, ncols, 2 + k*2)
            plot_raster(spike_trains, ax=ax, duration=end_time-start_time, bin_size=0.001, time_offset=start_time, ylabel='Electrode', groups=spike_train_groups)

            electrode_indices,lfp_matrix,sample_rate = lfps[rcg.name]
            nelectrodes = len(electrode_indices)
            
            #zscore the LFP
            LFPmean = lfp_matrix.T.mean(axis=0)
            LFPstd = lfp_matrix.T.std(axis=0, ddof=1)
            nz = LFPstd > 0.0
            lfp_matrix.T[:, nz] -= LFPmean[nz]
            lfp_matrix.T[:, nz] /= LFPstd[nz]

            ax = plt.subplot(nrows, ncols, 2 + k*2 + 1)
            plt.imshow(lfp_matrix, interpolation='nearest', aspect='auto', cmap=cm.seismic, extent=[start_time, end_time, 1, nelectrodes])
            lbls = ['%d' % e for e in electrode_indices]
            lbls.reverse()
            plt.yticks(range(nelectrodes), lbls)
            plt.axis('tight')
            plt.ylabel('Electrode')

if __name__ == '__main__':

    # exp = Experiment.load('/auto/tdrive/mschachter/data/GreBlu9508M/GreBlu9508M.h5', '/auto/tdrive/mschachter/data/GreBlu9508M/stims.h5')
    exp_dir = '/auto/tdrive/fdata/billewood/tdt_h5/GreWhi1242F'
    exp_file = os.path.join(exp_dir, 'GreWhi1242F.h5')
    stim_file = os.path.join(exp_dir, 'stims.h5')

    exp = Experiment.load(exp_file, stim_file)
    plotter = ExperimentPlotter(exp)

    start_time = 600.0
    end_time = 610.0

    seg = exp.get_segment('Site1', 'Call1')

    plotter.plot_segment_slice(seg, start_time, end_time)

    plt.show()
