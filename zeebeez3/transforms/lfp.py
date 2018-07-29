import os

import numpy as np
from numpy.fft import fftfreq

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter1d, convolve1d

from soundsig.plots import make_phase_image
from soundsig.signal import compute_instantaneous_frequency, lowpass_filter, demodulate
from soundsig.sound import plot_spectrogram
from soundsig.spikes import plot_raster

from zeebeez3.core.experiment import segment_to_unique_name
from zeebeez3.transforms.spike_event import spike_envelope


class LFPTransform(object):

    def __init__(self):
        self.experiment = None
        self.segment = None
        self.index2electrode = None
        self.start_time = None
        self.end_time = None
        self.num_bands = None
        self.rcg_names = None
        self.X = None
        self.Z = None
        self.spike_rasters = None
        self.file_name = None

    def get_bands(self):
        return self.X

    def get_complex_bands(self):
        return self.Z

    def get_sample_rate(self):
        return float(self.experiment.lfp_sample_rate)

    def clear_data(self):
        del self.X
        del self.Z

    def plot(self, **kwargs):

        seg_uname = segment_to_unique_name(self.segment)

        #set keywords to defaults
        kw_params = {'start_time':self.start_time, 'end_time':self.end_time, 'bands_to_plot':list(range(self.num_bands)), 'output_dir':None}
        for key,val in kw_params.items():
            if key not in kwargs:
                kwargs[key] = val

        bands_to_plot = kwargs['bands_to_plot']
        start_time = kwargs['start_time']
        end_time = kwargs['end_time']
        output_dir = kwargs['output_dir']

        sr = self.get_sample_rate()
        stim_spec_t,stim_spec_freq,stim_spec = self.experiment.get_spectrogram_slice(self.segment, start_time, end_time)
        t1 = int((start_time-self.start_time)*sr)
        t2 = int((end_time-self.start_time)*sr)

        #compute the average power spectrum of each band across electrodes
        band_ps_list = list()
        band_ps_freq = None
        for n in bands_to_plot:
            ps = list()
            for k,enumber in enumerate(self.index2electrode):
                band = self.X[n, k, t1:t2].squeeze()
                band_fft = fft(band)
                freq = fftfreq(len(band), d=1.0/self.get_sample_rate())
                findex = freq > 0.0
                band_ps = np.real(band_fft[findex]*np.conj(band_fft[findex]))
                ps.append(band_ps)
                band_ps_freq = freq[findex]
                ps.append(band_ps)

            band_ps_list.append(np.array(ps))

        band_ps = np.array(band_ps_list)

        #plot the average power spectrums
        thresh = 0.0
        plt.figure(figsize=(24.0, 13.5))
        max_pow = -np.inf
        clrs = ['b', 'g', 'r', 'c', 'm', 'y']
        for k,n in enumerate(bands_to_plot):
            #compute mean across electrodes
            band_ps_mean = band_ps[k, :].mean(axis=0)
            band_ps_mean_filt = gaussian_filter1d(band_ps_mean, 10)
            nzindex = band_ps_mean_filt > 0.0
            band_ps_mean_filt[nzindex] = np.log10(band_ps_mean_filt)
            #plt.subplot(len(bands_to_plot), 1, k+1)

            #threshold the power spectrum at zero
            band_ps_mean_filt[band_ps_mean_filt < thresh] = 0.0

            cnorm = float(k) / len(bands_to_plot)
            a = 0.75*cnorm + 0.25
            c = [cnorm, 1.0 - cnorm, 1.0 - cnorm]
            max_pow = max(max_pow, band_ps_mean_filt.max())
            c = clrs[k]
            plt.plot(band_ps_freq, band_ps_mean_filt, '-', c=c, linewidth=3.0)
        plt.ylim(0.0, max_pow)
        plt.legend(['%d' % (n+1) for n in bands_to_plot])
        plt.title("Band Average Power Spectrum Across Electrodes")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Power (dB)')
        plt.axis('tight')

        if output_dir is not None:
            fname = 'band_ps_%s_%s_start%0.6f_end%0.6f.png' % (seg_uname, ','.join(self.rcg_names), start_time, end_time)
            plt.savefig(os.path.join(output_dir, fname))
            plt.close('all')

        #plot the bands
        t = np.arange(t2-t1) / self.get_sample_rate()

        subplot_nrows = 8 + 1
        subplot_ncols = len(self.rcg_names)*2

        for n in bands_to_plot:
            plt.figure(figsize=(24.0, 13.5))
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)
            plt.suptitle('Band %d' % (n+1))

            for k in range(subplot_ncols):
                ax = plt.subplot(subplot_nrows, subplot_ncols, k+1)
                plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.gist_yarg, colorbar=False)
                plt.ylabel('')
                plt.yticks([])

            for k,electrode in enumerate(self.index2electrode):

                rcg,rc = self.experiment.get_channel(self.segment.block, electrode)
                row = rc.annotations['row']
                col = rc.annotations['col']

                if len(self.rcg_names) > 1:
                    sp = (row+1)*subplot_ncols + col + 1
                else:
                    sp = (row+1)*subplot_ncols + (col % 2) + 1

                plt.subplot(subplot_nrows, subplot_ncols, sp)

                band = self.X[n, k, t1:t2].squeeze()
                plt.plot(t, band, 'k-', linewidth=2.0)
                plt.ylabel('E%d' % electrode)
                plt.axis('tight')

            if output_dir is not None:
                fname = 'band_raw_%s_%s_%d_start%0.6f_end%0.6f.png' % (seg_uname, ','.join(self.rcg_names), n+1, start_time, end_time)
                plt.savefig(os.path.join(output_dir, fname))
                plt.close('all')

        if output_dir is None:
            plt.show()

    def plot_complex(self, **kwargs):

        #set keywords to defaults
        kw_params = {'start_time':self.start_time, 'end_time':self.end_time, 'output_dir':None, 'include_spec':False,
                     'include_ifreq':False, 'spikes':False, 'nbands_to_plot':None, 'sort_code':'0', 'demodulate':True,
                     'phase_only':False, 'log_amplitude':False}
        for key,val in kw_params.items():
            if key not in kwargs:
                kwargs[key] = val

        nbands,nelectrodes,nt = self.X.shape
        start_time = kwargs['start_time']
        end_time = kwargs['end_time']
        duration = end_time - start_time
        output_dir = kwargs['output_dir']

        sr = self.get_sample_rate()
        t1 = int((start_time-self.start_time)*sr)
        d = int((end_time-start_time)*sr)
        t2 = t1 + d
        t = (np.arange(d)/sr) + kwargs['start_time']

        spike_trains = None
        bin_size = 1e-3
        if kwargs['spikes']:
            if self.spike_rasters is None:
                #load up the spike raster for this time slice
                self.spike_rasters = self.experiment.get_spike_slice(self.segment, 0.0, self.segment.annotations['duration'],
                                                                     rcg_names=self.rcg_names, as_matrix=False,
                                                                     sort_code=kwargs['sort_code'], bin_size=bin_size)
            if len(self.spike_rasters) > 1:
                print("WARNING: plot_complex doesn't work well when more than one electrode array is specified.")
            spike_trains_full,spike_train_group = self.spike_rasters[self.rcg_names[0]]
            #select out the spikes for the interval to plot
            spike_trains = list()
            for st in spike_trains_full:
                sindex = (st >= start_time) & (st <= end_time)
                spike_trains.append(st[sindex])

        colors = np.array([[244.0, 244.0, 244.0], #white
                           [241.0, 37.0, 9.0],    #red
                           [238.0, 113.0, 25.0],  #orange
                           [255.0, 200.0, 8.0],   #yellow
                           [19.0, 166.0, 50.0],   #green
                           [1.0, 134.0, 141.0],   #blue
                           [244.0, 244.0, 244.0], #white
                          ])
        colors /= 255.0

        #get stimulus spectrogram
        stim_spec_t,stim_spec_freq,stim_spec = self.experiment.get_spectrogram_slice(self.segment, kwargs['start_time'], kwargs['end_time'])

        #compute the amplitude, phase, and instantaneous frequency of the complex signal
        amplitude = np.abs(self.Z[:, :, t1:t2])
        phase = np.angle(self.Z[:, :, t1:t2])

        # rescale the amplitude of each electrode so it ranges from 0 to 1
        for k in range(nbands):
            for n in range(nelectrodes):
                amplitude[k, n, :] /= amplitude[k, n].max()

        if kwargs['phase_only']:
            # make sure the amplitude is equal to 1
            nz = amplitude > 0
            amplitude[nz] /= amplitude[nz]

        if kwargs['log_amplitude']:
            nz = amplitude > 0
            amplitude[nz] = np.log10(amplitude[nz])
            amplitude[nz] -= amplitude[nz].min()
            amplitude /= amplitude.max()

        nbands_to_plot = nbands
        if kwargs['nbands_to_plot'] is not None:
            nbands_to_plot = kwargs['nbands_to_plot']

        seg_uname = segment_to_unique_name(self.segment)

        if kwargs['include_ifreq']:
            ##################
            ## make plots of the joint instantaneous frequency per band
            ##################
            rcParams.update({'font.size':10})
            plt.figure(figsize=(24.0, 13.5))
            plt.subplots_adjust(top=0.98, bottom=0.01, left=0.03, right=0.99, hspace=0.10)
            nsubplots = nbands_to_plot + 2

            #plot the stimulus spectrogram
            ax = plt.subplot(nsubplots, 1, 1)
            plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.afmhot_r, colorbar=False, fmax=8000.0)
            plt.ylabel('')
            plt.yticks([])

            ifreq = np.zeros([nbands, nelectrodes, d])
            sr = self.get_sample_rate()
            for k in range(nbands):
                for j in range(nelectrodes):
                    ifreq[k, j, :] = compute_instantaneous_frequency(self.Z[k, j, t1:t2], sr)
                    ifreq[k, j, :] = lowpass_filter(ifreq[k, j, :], sr, cutoff_freq=50.0)

            #plot the instantaneous frequency along with it's amplitude
            for k in range(nbands_to_plot):
                img = np.zeros([nelectrodes, d, 4], dtype='float32')

                ifreq_min = np.percentile(ifreq[k, :, :], 5)
                ifreq_max = np.percentile(ifreq[k, :, :], 95)
                ifreq_dist = ifreq_max - ifreq_min
                #print 'ifreq_max=%0.3f, ifreq_min=%0.3f' % (ifreq_max, ifreq_min)

                for j in range(nelectrodes):
                    max_amp = np.percentile(amplitude[k, j, :], 85)

                    #set the alpha and color for the bins
                    alpha = amplitude[k, j, :] / max_amp
                    alpha[alpha > 1.0] = 1.0 #saturate
                    alpha[alpha < 0.05] = 0.0 #nonlinear threshold

                    cnorm = (ifreq[k, j, :] - ifreq_min) / ifreq_dist
                    cnorm[cnorm > 1.0] = 1.0
                    cnorm[cnorm < 0.0] = 0.0
                    img[j, :, 0] = 1.0 - cnorm
                    img[j, :, 1] = 1.0 - cnorm
                    img[j, :, 2] = 1.0 - cnorm
                    #img[j, :, 3] = alpha
                    img[j, :, 3] = 1.0

                ax = plt.subplot(nsubplots, 1, k+2)
                ax.set_axis_bgcolor('black')
                im = plt.imshow(img, interpolation='nearest', aspect='auto', origin='upper', extent=[t.min(), t.max(), 1, nelectrodes])
                plt.axis('tight')
                plt.ylabel('Electrode')
                plt.title('band %d' % (k+1))
            plt.suptitle('Instantaneous Frequency')

            if output_dir is not None:
                fname = 'band_ifreq_%s_%s_start%0.6f_end%0.6f.png' % (seg_uname, ','.join(self.rcg_names), start_time, end_time)
                plt.savefig(os.path.join(output_dir, fname))

        ##################
        ## make plots of the joint phase per band
        ##################

        def compute_envelope(the_matrix, log=False):
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

        #compute the amplitude envelope for the spectrogram
        stim_spec_env = compute_envelope(stim_spec)

        rcParams.update({'font.size':10})
        fig = plt.figure(figsize=(24.0, 13.5), facecolor='gray')
        plt.subplots_adjust(top=0.98, bottom=0.01, left=0.03, right=0.99, hspace=0.10)
        nsubplots = nbands_to_plot + 1 + int(kwargs['spikes'])

        #plot the stimulus spectrogram
        ax = plt.subplot(nsubplots, 1, 1)
        ax.set_axis_bgcolor('black')
        plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.afmhot, colorbar=False, fmax=8000.0)
        plt.plot(stim_spec_t, stim_spec_env*stim_spec_freq.max(), 'w-', linewidth=3.0, alpha=0.75)
        plt.axis('tight')
        plt.ylabel('')
        plt.yticks([])

        #plot the spike raster
        if kwargs['spikes']:
            spike_count_env = spike_envelope(spike_trains, start_time, duration, bin_size=bin_size, win_size=30.0)
            ax = plt.subplot(nsubplots, 1, 2)
            plot_raster(spike_trains, ax=ax, duration=duration, bin_size=bin_size, time_offset=start_time,
                        ylabel='Cell', bgcolor='k', spike_color='#ff0000')
            tenv = np.arange(len(spike_count_env))*bin_size + start_time
            plt.plot(tenv, spike_count_env*len(spike_trains), 'w-', linewidth=1.5, alpha=0.5)
            plt.axis('tight')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Spikes')

        #phase_min = phase.min()
        #phase_max = phase.max()
        #print 'phase_max=%0.3f, phase_min=%0.3f' % (phase_max, phase_min)

        for k in range(nbands_to_plot):

            the_phase = phase[k, :, :]
            if kwargs['demodulate']:
                Ztemp = amplitude[k, :, :]*(np.cos(the_phase) + complex(0, 1)*np.sin(the_phase))
                the_phase,complex_pcs = demodulate(Ztemp, depth=1)
                del Ztemp

            img = make_phase_image(amplitude[k, :, :], the_phase, normalize=True, threshold=True, saturate=True)
            amp_env = compute_envelope(amplitude[k, :, :], log=False)

            ax = plt.subplot(nsubplots, 1, k+2+int(kwargs['spikes']))
            ax.set_axis_bgcolor('black')
            im = plt.imshow(img, interpolation='nearest', aspect='auto', origin='upper', extent=[t.min(), t.max(), 1, nelectrodes])
            if not kwargs['phase_only']:
                plt.plot(t, amp_env*nelectrodes, 'w-', linewidth=2.0, alpha=0.75)
            plt.axis('tight')
            plt.ylabel('Electrode')
            plt.title('band %d' % (k+1))

        plt.suptitle('Phase')

        if output_dir is not None:
            fname = 'band_phase_%s_%s_start%0.6f_end%0.6f.png' % (seg_uname, ','.join(self.rcg_names), start_time, end_time)
            plt.savefig(os.path.join(output_dir, fname), facecolor=fig.get_facecolor(), edgecolor='none')
            del fig

        if not kwargs['include_spec']:
            return

        ##################
        ## make plots of the band spectrograms
        ##################
        subplot_nrows = 8 + 1
        subplot_ncols = len(self.rcg_names)*2
        plt.figure()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

        for k in range(subplot_ncols):
            ax = plt.subplot(subplot_nrows, subplot_ncols, k+1)
            plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.gist_yarg, colorbar=False)
            plt.ylabel('')
            plt.yticks([])

        for j in range(nelectrodes):

            plt.figure(figsize=(24.0, 13.5))
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

            electrode = self.index2electrode[j]
            rcg,rc = self.experiment.get_channel(self.segment.block, electrode)
            row = rc.annotations['row']
            col = rc.annotations['col']
            if len(self.rcg_names) > 1:
                sp = (row+1)*subplot_ncols + col + 1
            else:
                sp = (row+1)*subplot_ncols + (col % 2) + 1

            #ax = plt.subplot(subplot_nrows, subplot_ncols, sp)
            gs = GridSpec(100, 1)
            ax = plt.subplot(gs[:20])
            plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, colormap=cm.gist_yarg, colorbar=False)
            plt.ylabel('')
            plt.yticks([])

            ax = plt.subplot(gs[20:])
            ax.set_axis_bgcolor('black')

            #get the maximum frequency and set the resolution
            max_freq = ifreq[:, j, :].max()
            nf = 150
            df = max_freq / nf

            #create an image to hold the frequencies
            img = np.zeros([nf, ifreq.shape[-1], 4], dtype='float32')

            #fill in the image for each band
            for k in range(nbands):
                max_amp = np.percentile(amplitude[k, j, :], 85)

                freq_bin = (ifreq[k, j, :] / df).astype('int') - 1
                freq_bin[freq_bin < 0] = 0

                #set the color and alpha for the bins
                alpha = amplitude[k, j, :] / max_amp
                alpha[alpha > 1.0] = 1.0 #saturate
                alpha[alpha < 0.05] = 0.0 #nonlinear threshold

                for m,fbin in enumerate(freq_bin):
                    #print 'm=%d, fbin=%d, colors[k, :].shape=%s' % (m, fbin, str(colors[k, :].shape))
                    img[fbin, m, :3] = colors[k, :]
                    img[fbin, m, 3] = alpha[m]

            #plot the image
            im = plt.imshow(img, interpolation='nearest', aspect='auto', origin='lower', extent=[t.min(), t.max(), 0.0, max_freq])
            plt.ylabel('E%d' % electrode)
            plt.axis('tight')
            plt.ylim(0.0, 140.0)

            if output_dir is not None:
                fname = 'band_spec_e%d_%s_%s_start%0.6f_end%0.6f.png' % (electrode, seg_uname, ','.join(self.rcg_names), start_time, end_time)
                plt.savefig(os.path.join(output_dir, fname))
                plt.close('all')

        if output_dir is None:
            plt.show()

    def plot_single_electrode(self, enumber, start_time, end_time):

        #get stimulus spectrogram
        spec_t,spec_freq,spec = self.experiment.get_spectrogram_slice(self.segment, start_time, end_time)

        lfp = self.experiment.get_single_lfp_slice(self.segment, enumber, start_time, end_time)

        sr = self.get_sample_rate()
        si = int(start_time*sr)
        ei = int(end_time*sr)
        t = (np.arange(ei-si) / sr) + start_time

        #get the bands
        index2electrode = list(self.index2electrode)
        eindex = index2electrode.index(enumber)
        bands_to_plot = list(range(6))
        X = self.get_bands()
        bands = np.array([X[n, eindex, si:ei] for n in bands_to_plot])

        #make plots
        nrows = len(bands_to_plot) + 2
        ncols = 1
        rcParams.update({'font.size':18})
        plt.figure()
        plt.subplots_adjust(top=0.95, bottom=0.07, left=0.03, right=0.99, hspace=0.10)

        #plot spectrogram
        ax = plt.subplot(nrows, ncols, 1)
        plot_spectrogram(spec_t, spec_freq, spec, ax=ax, colormap=cm.afmhot_r, colorbar=False, fmin=300.0, fmax=8000.0)
        plt.ylabel('Spectrogram')
        plt.yticks([])
        plt.xticks([])

        #plot raw LFP
        ax = plt.subplot(nrows, ncols, 2)
        plt.plot(t, lfp, 'k-', linewidth=3.0)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('LFP')
        plt.axis('tight')

        #plot the bands
        for n in bands_to_plot:
            ax = plt.subplot(nrows, ncols, 2 + n + 1)
            plt.plot(t, bands[n, :], 'r-', linewidth=3.0)
            plt.yticks([])
            plt.ylabel('band %d' % n)
            if n < len(bands_to_plot)-1:
                plt.xticks([])
            else:
                plt.xlabel('Time (s)')
            plt.axis('tight')

    def compute_power_spectrum(self, segment_size=60.0, output_dir=None):
        """ Compute the average power spectrum across the entire transform, by averaging the power spectrum of segments.
        :param segment_size:
        :return:
        """

        seg_uname = segment_to_unique_name(self.segment)
        duration = self.end_time - self.start_time
        nsegs = int(duration / segment_size) #skip the last segment

        nbands,nelectrodes,nt = self.X.shape

        ps_by_band_and_electrode = dict()
        std_by_band_and_electrode = dict()
        freq = None
        nbands_to_plot = 6
        for n in range(nelectrodes):
            for bi in range(nbands_to_plot):
                print('Computing spectrums for electrode %d, band %d' % (self.index2electrode[n], bi))
                #compute the power spectrum for every segment in this electrode's band
                spectrums = list()
                band = self.X[bi, n, :]
                #zscore the band
                band -= band.mean()
                band /= band.std(ddof=1)
                for k in range(nsegs):
                    si = int(k*segment_size*self.experiment.lfp_sample_rate)
                    ei = si + int(segment_size*self.experiment.lfp_sample_rate)
                    x = band[si:ei]
                    xfft = fft(x)
                    xfreq = fftfreq(len(xfft), d=1.0 / self.experiment.lfp_sample_rate)
                    findex = xfreq > 0.0
                    ps = np.abs(xfft[findex])
                    freq = xfreq[findex]
                    spectrums.append(ps)
                spectrums = np.array(spectrums)

                key = (self.index2electrode[n], bi)
                ps_by_band_and_electrode[key] = spectrums.mean(axis=0)
                std_by_band_and_electrode[key] = spectrums.std(axis=0, ddof=1)

        clrs = ['b', 'g', 'r', 'c', 'm', 'k']

        figsize = None
        if output_dir is not None:
            figsize=(24.0, 13.5)

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)
        rcParams.update({'font.size':20})
        for n in range(nelectrodes):
            enumber = self.index2electrode[n]

            ymax = -np.inf
            plt.figure(figsize=figsize)
            for bi in range(nbands_to_plot):
                key = (enumber, bi)
                ps = ps_by_band_and_electrode[key]
                std = std_by_band_and_electrode[key]
                #plt.errorbar(freq, ps, yerr=std, ecolor='#aaaaaa', capsize=0, color=clrs[bi], linewidth=3, alpha=1.0)
                plt.plot(freq, ps, '-', c=clrs[bi], linewidth=4.0)
                ymax = max(ymax, ps.max())
            ax = plt.gca()
            ax.set_xscale('log')
            plt.axis('tight')
            plt.ylim(0.0, ymax)
            plt.xlim(1.0, freq.max())

            xticks = np.logspace(0.0, np.log10(freq.max()), 10)
            xticklbls = ['%d' % x for x in xticks]
            plt.xticks(xticks, xticklbls)

            plt.legend(['%d' % bi for bi in range(nbands_to_plot)])
            plt.suptitle('Electrode %d' % enumber)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            if output_dir is not None:
                ofile = os.path.join(output_dir, '%s_%s_power_spectrum_e%d.png' % (seg_uname, ','.join(self.rcg_names), enumber))
                plt.savefig(ofile)

    def plot_entire_segment(self, plot_length=10.0, output_dir='/tmp', sort_code='0', demodulate=True, phase_only=False):

        duration = self.segment.annotations['duration']
        num_plots = int(np.ceil(duration / plot_length))
        for k in range(num_plots):
            start_time = k*plot_length
            end_time = min(duration, (k+1)*plot_length)
            print('Plotting from %0.3fs to %0.3fs to %s' % (start_time, end_time, output_dir))

            self.plot_complex(output_dir=output_dir, start_time=start_time, end_time=end_time, include_spec=False,
                              spikes=True, demodulate=demodulate, sort_code=sort_code, phase_only=True)
            plt.close('all')
