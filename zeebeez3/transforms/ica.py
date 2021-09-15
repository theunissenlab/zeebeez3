import os
import time
import operator

import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

from neo import Segment
from soundsig.plots import multi_plot
from soundsig.sound import plot_spectrogram

from zeebeez3.core.experiment import Experiment, segment_to_unique_name


class ICALFPTransform(object):

    def __init__(self):
        self.experiment_file = None
        self.stimulus_file = None
        self.block_name = None
        self.segment_name = None
        self.seg_uname = None
        self.lfp_sample_rate = None
        self.duration = None

        self.X = None
        self.data = None
        self.df = None
        self.components = None
        self.components_transformed = None

    def transform(self, experiment, segment, electrodes=None):
        """ Perform ICA on the raw multielectrode LFP, for both hemispheres on a segment. """

        assert isinstance(experiment, Experiment)
        assert isinstance(segment, Segment)

        self.experiment_file = experiment.file_name
        self.stimulus_file = experiment.stimulus_file_name
        self.block_name = segment.block.name
        self.segment_name = segment.name
        self.seg_uname = segment_to_unique_name(segment)
        self.duration = segment.annotations['duration']
        self.lfp_sample_rate = experiment.lfp_sample_rate

        self.data = {'electrode':list(), 'hemisphere':list(), 'region':list(),
                     'row':list(), 'col':list(), 'index':list()}

        # map the electrodes to their hemisphere, anatomical region, and (row,col)
        all_electrodes = list()
        electrode2hemi = dict()
        electrode2region = dict()
        electrode2coord = dict()

        for rgc in segment.block.recordingchannelgroups:
            all_electrodes.extend(rgc.channel_indexes)

            for electrode in rgc.channel_indexes:
                rgc2,rc = experiment.get_channel(segment.block, electrode)
                row = int(rc.annotations['row'])
                col = int(rc.annotations['col'])

                electrode2hemi[electrode] = rgc.name
                electrode2coord[electrode] = (row, col)
                electrode2region[electrode] = rc.annotations['region']
        all_electrodes.sort()

        if electrodes is None:
            electrodes = all_electrodes

        print('Electrodes: %s' % str(electrodes))

        # determine segment duration and construct matrix to hold LFPs
        nt = int(self.duration*self.lfp_sample_rate)
        nelectrodes = len(electrodes)

        # get the raw LFP on each electrode
        the_index = 0
        lfps = np.zeros([nt, nelectrodes])
        for k,electrode in enumerate(electrodes):
            lfps[:, k] = experiment.get_single_lfp_slice(segment, electrode, 0, self.duration)

            self.data['electrode'].append(electrode)
            self.data['hemisphere'].append(electrode2hemi[electrode])
            self.data['region'].append(electrode2region[electrode])
            self.data['row'].append(electrode2coord[electrode][0])
            self.data['col'].append(electrode2coord[electrode][1])
            self.data['index'].append(the_index)
            the_index += 1

        # delete the experiment to save on space
        del segment
        del experiment

        # create a data frame
        self.df = pd.DataFrame(self.data)

        stime = time.time()

        # z-score the LFPs over time
        print('Z-scoring...')
        lfps -= lfps.mean(axis=0)
        lfps /= lfps.std(axis=0, ddof=1)

        # do PCA on the LFPs and whiten them
        print('Doing PCA...')
        pca = PCA(whiten=True)
        lfps = pca.fit_transform(lfps)

        # do ICA on the whitened PC-projected LFPs
        print('Doing ICA...')
        ica = FastICA()
        self.X = ica.fit_transform(lfps)

        etime = time.time() - stime
        print('Elapsed time: %0.0f seconds' % etime)

        # delete the LFPs
        del lfps

        # preserve the ICs
        self.components = ica.components_

        # transform the ICs back into anatomical space
        self.components_transformed = pca.inverse_transform(self.components)

    def save(self, output_file):
        hf = h5py.File(output_file, 'w')

        hf.attrs['experiment_file'] = self.experiment_file
        hf.attrs['stimulus_file'] = self.stimulus_file
        hf.attrs['block_name'] = self.block_name
        hf.attrs['segment_name'] = self.segment_name
        hf.attrs['seg_uname'] = self.seg_uname
        hf.attrs['lfp_sample_rate'] = self.lfp_sample_rate
        hf.attrs['duration'] = self.duration

        col_names = list(self.data.keys())
        hf.attrs['col_names'] = col_names
        for cname in col_names:
            hf[cname] = np.array(self.data[cname])

        hf['X'] = self.X

        hf['components'] = self.components
        hf['components_transformed'] = self.components_transformed

        hf.close()

    @classmethod
    def load(clz, output_file):
        hf = h5py.File(output_file, 'r')

        icat = ICALFPTransform()
        icat.experiment_file = hf.attrs['experiment_file']
        icat.stimulus_file = hf.attrs['stimulus_file']
        icat.block_name = hf.attrs['block_name']
        icat.segment_name = hf.attrs['segment_name']
        icat.seg_uname = hf.attrs['seg_uname']
        icat.lfp_sample_rate = hf.attrs['lfp_sample_rate']
        icat.duration = hf.attrs['duration']

        icat.components = np.array(hf['components'])
        icat.components_transformed = np.array(hf['components_transformed'])
        icat.X = np.array(hf['X'])

        col_names = hf.attrs['col_names']
        icat.data = dict()
        for cname in col_names:
            icat.data[cname] = np.array(hf[cname])
        icat.df = pd.DataFrame(icat.data)

        hf.close()

        return icat

    def plot(self):

        # rescale the components
        self.components_transformed /= np.abs(self.components_transformed).max()
        absmax = np.abs(self.components_transformed).max()

        electrodes = self.df['electrode'].unique()

        ncols = 4
        nrows = 8
        if len(electrodes) < 32:
            ncols = 2

        # plot the independent components
        plist = list()
        for ic in self.components_transformed:

            # re-organize IC into matrix
            icmat = np.zeros([nrows, ncols])
            txt = np.zeros([nrows, ncols], dtype='S20')

            for k,icval in enumerate(ic):
                i = self.df['index'] == k
                assert i.sum() == 1, "Zero or none electrodes at index %d"
                electrode = self.df[i]['electrode'].values[0]
                region = self.df[i]['region'].values[0]
                row = self.df[i]['row'].values[0]
                col = self.df[i]['col'].values[0]
                if electrode > 16 and ncols > 2:
                    col += 2

                icmat[row, col] = icval
                txt[row, col] = "%d\n%s" % (electrode, region)

                print('txt=')
                print(txt)

            plist.append({'IC':icmat, 'txt':txt})

        def _plot_ic(pdata, ax):
            absmax = np.abs(pdata['IC']).max()
            plt.sca(ax)
            plt.imshow(pdata['IC'], aspect='auto', interpolation='nearest', cmap=plt.cm.seismic,
                       vmin=-absmax, vmax=absmax, origin='upper')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            for row in range(nrows):
                for col in range(ncols):
                    # plt.text(col, (7-row)+0.25, pdata['txt'][row, col], horizontalalignment='center', fontsize=8)
                    plt.text(col, row+0.25, pdata['txt'][row, col], horizontalalignment='center', fontsize=8)

        multi_plot(plist, _plot_ic, nrows=ncols, ncols=nrows)

    def plot_all_slices(self, slice_len=3.0, output_dir=None):

        # zcsore the data
        self.X -= self.X.mean(axis=0)
        self.X /= self.X.std(axis=0, ddof=1)
        absmax = np.percentile(np.abs(self.X), 99)

        # generate random colors
        C = np.random.rand(len(self.components), 3)

        # rescale random colors so they're light on a black background
        C *= 0.75
        C += 0.25

        print('Loading experiment...')
        exp = Experiment.load(self.experiment_file, self.stimulus_file, read_only_stims=True)
        seg = exp.get_segment(self.block_name, self.segment_name)

        slices = np.arange(0, self.duration, slice_len)
        for stime,etime in zip(slices[:-1], slices[1:]):
            print('Plotting slice from %0.6fs to %0.6fs' % (stime, etime))
            self.plot_slice(stime, etime, exp=exp, seg=seg, colors=C, absmax=absmax, output_dir=output_dir)
            plt.close('all')

    def plot_slice(self, start_time, end_time, exp=None, seg=None, absmax=None, colors=None, output_dir=None):

        #get stimulus spectrogram
        stim_spec_t,stim_spec_freq,stim_spec = exp.get_spectrogram_slice(seg, start_time, end_time)

        fig = plt.figure(figsize=(24.0, 13.5), facecolor='gray')
        fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02, hspace=0.05)
        gs = plt.GridSpec(100, 1)

        #plot the stimulus spectrogram
        ax = plt.subplot(gs[:10, 0])
        ax.set_axis_bgcolor('black')
        plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=plt.cm.afmhot, colorbar=False, fmax=8000.0)
        plt.axis('tight')
        plt.ylabel('')
        plt.yticks([])
        plt.xticks([])

        si = int(self.lfp_sample_rate*start_time)
        ei = int(self.lfp_sample_rate*end_time)
        nt = ei - si
        ncomps = self.components.shape[0]

        if absmax is None:
            absmax = np.abs(self.X[si:ei, :]).max()
        padding = 0.05*absmax
        plot_height = (2*absmax+padding)*ncomps

        t = np.linspace(start_time, end_time, nt)
        ax = plt.subplot(gs[10:, 0])
        ax.set_axis_bgcolor('black')
        for k in range(ncomps):

            offset = absmax + padding + k*(padding+2*absmax)

            x = self.X[si:ei, k] + offset
            c = 'k'
            if colors is not None:
                c = colors[k, :]

            plt.plot(t, x, '-', c=c, linewidth=2.0)

        plt.axis('tight')
        plt.ylim(0, plot_height)
        plt.yticks([])

        if output_dir is not None:
            ofile = os.path.join(output_dir, 'ica_%0.6f_%0.6f.png' % (start_time, end_time))
            plt.savefig(ofile, facecolor=fig.get_facecolor(), edgecolor='none')
        else:
            plt.show()


if __name__ == '__main__':

    exp_name = 'GreBlu9508M'
    exp_file = '/auto/tdrive/mschachter/data/GreBlu9508M/%s.h5' % exp_name
    stim_file = '/auto/tdrive/mschachter/data/GreBlu9508M/stims.h5'
    output_dir = '/auto/tdrive/mschachter/data/GreBlu9508M/transforms'

    block_name = 'Site4'
    segment_name = 'Call1'

    exp = Experiment.load(exp_file, stim_file, read_only_stims=True)
    segment = exp.get_segment(block_name, segment_name)
    seg_uname = segment_to_unique_name(segment)

    ofname = 'ICA_%s_L.h5' % seg_uname
    ofile = os.path.join(output_dir, ofname)

    icat = ICALFPTransform()
    icat.transform(exp, segment, electrodes=np.arange(16)+1)
    icat.save(ofile)

    # icat = ICALFPTransform.load(ofile)
    # icat.plot()
    # icat.plot_all_slices(output_dir='/auto/tdrive/mschachter/figures/ica/GreBlu9508M/L')
    # plt.show()
