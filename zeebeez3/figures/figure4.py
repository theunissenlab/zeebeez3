import os
import sys
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from soundsig.sound import spec_colormap, plot_spectrogram
from soundsig.spikes import compute_psth, plot_raster

from zeebeez3.core.utils import USED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, \
    ACOUSTIC_PROP_COLORS_BY_TYPE, ACOUSTIC_PROP_NAMES, to_hex
from zeebeez3.transforms.biosound import BiosoundTransform
from zeebeez3.transforms.pairwise_cf import PairwiseCFTransform
from zeebeez3.transforms.stim_event import StimEventTransform


def get_this_dir():
    """ Get the directory that contains the python file that is calling this function. """

    f = sys._current_frames().values()[0]
    calling_file_path = f.f_back.f_globals['__file__']
    root_dir,fname = os.path.split(calling_file_path)
    return root_dir


def get_full_data(bird, block, segment, hemi, stim_id, data_dir='/auto/tdrive/mschachter/data',
                  biosound=None, stim_event=None, pairwise_cf=None):
    """ This function aggregates data for a specific stimulus presentation and returns it in a
        dictionary.

    :param bird: The name of the bird (string)
    :param block: The block name (string)
    :param segment: The segment name (string)
    :param hemi: The hemisphere (string)
    :param stim_id: The stimulus id (integer?)
    :param biosound: Pre-loaded BiosoundTransform
    :param stim_event: Pre-loaded StimEventTransform
    :param pairwise_cf: Pre-loaded PairwiseCFTransform

    :return: A dictionary with the following elements:

        {
            'stim_id': the stimulus id
            'spec_t': a time vector for the spectrogram
            'spec_freq': the frequency vector for the spectrogram
            'spec': the log transformed stimulus spectrogram,
            'lfp': the raw multielectrode LFP, for all trials, for the specified stimulus, of shape (num_trials, num_electrodes, num_time_points)
            'spikes': a list of spike trains, each spike train is a list of spike times, the list is of shape (num_trials, num_neurons)
            'lfp_sample_rate': the sample rate of the LFP
            'psth': the trial-averaged PSTH for the stimulus
            'electrode_order': The electrodes that correspond to the indices of 'lfp'
            'cell_index2electrode': The electrode that corresponds to each cell in 'spikes'
            'psd_freq': The frequency vector that corresponds to the LFP power spectra
            'aprops': A list of acoustic features used
            'syllable_props': A dictionary of properties for each syllable of the stimulus:
                {
                    'start_time': The start time of the syllable
                    'end_time': The end time of the syllable
                    'order': The order of the syllable
                    'lfp_psd': An array of trial-averaged LFP power spectra, of shape (num_electrodes, len(psd_freq)
                }
        }

    """

    bdir = os.path.join(data_dir, bird)
    tdir = os.path.join(bdir, 'transforms')

    aprops = USED_ACOUSTIC_PROPS

    if biosound is None:
        # load the BioSound
        bs_file = os.path.join(tdir, 'BiosoundTransform_%s.h5' % bird)
        biosound = BiosoundTransform.load(bs_file)

    if stim_event is None:
        # load the StimEvent transform
        se_file = os.path.join(tdir, 'StimEvent_%s_%s_%s_%s.h5' % (bird,block,segment,hemi))
        print('Loading %s...' % se_file)
        stim_event = StimEventTransform.load(se_file, rep_types_to_load=['raw'])
        stim_event.zscore('raw')
        stim_event.segment_stims_from_biosound(bs_file)

    if pairwise_cf is None:
        # load the pairwise CF transform
        pcf_file = os.path.join(tdir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird,block,segment,hemi))
        print('Loading %s...' % pcf_file)
        pairwise_cf = PairwiseCFTransform.load(pcf_file)

    def log_transform(x, dbnoise=100.):
        x /= x.max()
        zi = x > 0
        x[zi] = 20*np.log10(x[zi]) + dbnoise
        x[x < 0] = 0
        x /= x.max()

    all_lfp_psds = deepcopy(pairwise_cf.psds)
    log_transform(all_lfp_psds)
    all_lfp_psds -= all_lfp_psds.mean(axis=0)
    all_lfp_psds /= all_lfp_psds.std(axis=0, ddof=1)

    # get overall biosound stats
    bs_stats = dict()
    for aprop in aprops:
        amean = biosound.stim_df[aprop].mean()
        astd = biosound.stim_df[aprop].std(ddof=1)
        bs_stats[aprop] = (amean, astd)

    for (stim_id2,stim_type2),gdf in stim_event.segment_df.groupby(['stim_id', 'stim_type']):
        print('%d: %s' % (stim_id2, stim_type2))

    # get the spectrogram
    i = stim_event.segment_df.stim_id == stim_id
    last_end_time = stim_event.segment_df.end_time[i].max()
    print('last_end_time=',last_end_time)

    spec_freq = stim_event.spec_freq
    stim_spec = stim_event.spec_by_stim[stim_id]
    spec_t = np.arange(stim_spec.shape[1]) / stim_event.lfp_sample_rate
    speci = np.max(np.where(spec_t <= last_end_time)[0])
    spec_t = spec_t[:speci]
    stim_spec = stim_spec[:, :speci]
    stim_dur = spec_t.max() - spec_t.min()

    # get the raw LFP
    si = int(stim_event.pre_stim_time*stim_event.lfp_sample_rate)
    ei = int(stim_dur*stim_event.lfp_sample_rate) + si
    lfp = stim_event.lfp_reps_by_stim['raw'][stim_id][:, :, si:ei]
    ntrials,nelectrodes,nt = lfp.shape

    # get the raw spikes, spike_mat is ragged array of shape (num_trials, num_cells, num_spikes)
    spike_mat = stim_event.spikes_by_stim[stim_id]
    assert ntrials == len(spike_mat)

    ncells = len(stim_event.cell_df)
    print('ncells=%d' % ncells)
    ntrials = len(spike_mat)

    # compute the PSTH
    psth = list()
    for n in range(ncells):
        # get the spikes across all trials for neuron n
        spikes = [spike_mat[k][n] for k in range(ntrials)]
        # make a PSTH
        _psth_t,_psth = compute_psth(spikes, stim_dur, bin_size=1.0/stim_event.lfp_sample_rate)
        psth.append(_psth)
    psth = np.array(psth)

    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    # get acoustic props and LFP/spike power spectra for each syllable
    syllable_props = list()

    i = biosound.stim_df.stim_id == stim_id
    orders = sorted(biosound.stim_df.order[i].values)
    cell_index2electrode = None
    for o in orders:
        i = (biosound.stim_df.stim_id == stim_id) & (biosound.stim_df.order == o)
        assert i.sum() == 1

        d = dict()
        d['start_time'] = biosound.stim_df.start_time[i].values[0]
        d['end_time'] = biosound.stim_df.end_time[i].values[0]
        d['order'] = o

        for aprop in aprops:
            amean,astd = bs_stats[aprop]
            d[aprop] = (biosound.stim_df[aprop][i].values[0] - amean) / astd

        # get the LFP power spectra
        lfp_psd = list()
        for k,e in enumerate(electrode_order):
            i = (pairwise_cf.df.stim_id == stim_id) & (pairwise_cf.df.order == o) & (pairwise_cf.df.decomp == 'full') & \
                (pairwise_cf.df.electrode1 == e) & (pairwise_cf.df.electrode2 == e)

            assert i.sum() == 1, "i.sum()=%d, stim_id=%s, order=%s, electrode1=%d, electrode2=%d" % \
                                 (i.sum(), stim_id, o, e, e)

            index = pairwise_cf.df[i]['index'].values[0]
            lfp_psd.append(all_lfp_psds[index, :])
        d['lfp_psd'] = np.array(lfp_psd)

        syllable_props.append(d)

    return {'stim_id':stim_id, 'spec_t':spec_t, 'spec_freq':spec_freq, 'spec':stim_spec,
            'lfp':lfp, 'spikes':spike_mat, 'lfp_sample_rate':stim_event.lfp_sample_rate, 'psth':psth,
            'syllable_props':syllable_props, 'electrode_order':electrode_order, 'psd_freq':pairwise_cf.freqs,
            'cell_index2electrode':cell_index2electrode, 'aprops':aprops}



def plot_full_data(d, syllable_index):

    syllable_start = d['syllable_props'][syllable_index]['start_time'] - 0.020
    syllable_end = d['syllable_props'][syllable_index]['end_time'] + 0.030

    figsize = (24.0, 10)
    fig = plt.figure(figsize=figsize, facecolor='w')
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20, wspace=0.20)

    gs = plt.GridSpec(100, 100)
    left_width = 55
    top_height = 30
    middle_height = 40
    # bottom_height = 40
    top_bottom_sep = 20

    # plot the spectrogram
    ax = plt.subplot(gs[:top_height+1, :left_width])
    spec = d['spec']
    spec[spec < np.percentile(spec, 15)] = 0
    plot_spectrogram(d['spec_t'], d['spec_freq']*1e-3, spec, ax=ax, colormap='SpectroColorMap', colorbar=False,
                     ticks=True, log=False, dBNoise=None)
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (s)')

    # plot the LFPs
    sr = d['lfp_sample_rate']
    # lfp_mean = d['lfp'].mean(axis=0)
    lfp_mean = d['lfp'][2, :, :]
    lfp_t = np.arange(lfp_mean.shape[1]) / sr
    nelectrodes,nt = lfp_mean.shape
    gs_i = top_height + top_bottom_sep
    gs_e = gs_i + middle_height + 1

    ax = plt.subplot(gs[gs_i:gs_e, :left_width])

    voffset = 5
    for n in range(nelectrodes):
        plt.plot(lfp_t, lfp_mean[nelectrodes-n-1, :] + voffset*n, 'k-', linewidth=3.0, alpha=0.75)
    plt.axis('tight')
    ytick_locs = np.arange(nelectrodes) * voffset
    plt.yticks(ytick_locs, list(reversed(d['electrode_order'])))
    plt.ylabel('Electrode')
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.xlabel('Time (s)')

    # plot the PSTH
    """
    gs_i = gs_e + 5
    gs_e = gs_i + bottom_height + 1
    ax = plt.subplot(gs[gs_i:gs_e, :left_width])
    ncells = d['psth'].shape[0]
    plt.imshow(d['psth'], interpolation='nearest', aspect='auto', origin='upper', extent=(0, lfp_t.max(), ncells, 0),
               cmap=psth_colormap(noise_level=0.1))

    cell_i2e = d['cell_index2electrode']
    print 'cell_i2e=',cell_i2e
    last_electrode = cell_i2e[0]
    for k,e in enumerate(cell_i2e):
        if e != last_electrode:
            plt.axhline(k, c='k', alpha=0.5)
            last_electrode = e

    ytick_locs = list()
    for e in d['electrode_order']:
        elocs = np.array([k for k,el in enumerate(cell_i2e) if el == e])
        emean = elocs.mean()
        ytick_locs.append(emean+0.5)
    plt.yticks(ytick_locs, d['electrode_order'])
    plt.ylabel('Electrode')

    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    """

    # plot the biosound properties
    sprops = d['syllable_props'][syllable_index]
    aprops = USED_ACOUSTIC_PROPS

    vals = [sprops[a] for a in aprops]
    ax = plt.subplot(gs[:top_height, (left_width+5):])
    plt.axhline(0, c='k')
    for k,(aprop,v) in enumerate(zip(aprops,vals)):
        bx = k
        rgb = np.array(ACOUSTIC_PROP_COLORS_BY_TYPE[aprop]).astype('int')
        clr_hex = to_hex(*rgb)
        plt.bar(bx, v, color=clr_hex, alpha=0.7)

    # plt.bar(range(len(aprops)), vals, color='#c0c0c0')
    plt.axis('tight')
    plt.ylim(-1.5, 1.5)
    plt.xticks(np.arange(len(aprops))+0.5, [ACOUSTIC_PROP_NAMES[aprop] for aprop in aprops], rotation=90)
    plt.ylabel('Z-score')

    # plot the LFP power spectra
    gs_i = top_height + top_bottom_sep
    gs_e = gs_i + middle_height + 1

    f = d['psd_freq']
    ax = plt.subplot(gs[gs_i:gs_e, (left_width+5):])
    plt.imshow(sprops['lfp_psd'], interpolation='nearest', aspect='auto', origin='upper',
               extent=(f.min(), f.max(), nelectrodes, 0), cmap=plt.cm.viridis, vmin=-2., vmax=2.)
    plt.colorbar(label='Z-scored Log Power')
    plt.xlabel('Frequency (Hz)')
    plt.yticks(np.arange(nelectrodes)+0.5, d['electrode_order'])
    plt.ylabel('Electrode')

    # fname = os.path.join(get_this_dir(), 'figure.svg')
    # plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def plot_single_trial_data(d, syllable_index, trial_index):

    syllable_start = d['syllable_props'][syllable_index]['start_time'] - 0.030
    syllable_end = d['syllable_props'][syllable_index]['end_time'] + 0.030

    # set the figure width proportional to the length of the syllable for uniformity across stimuli
    max_fig_width = 12.0
    max_stim_duration = 2.5

    fig_width = ((syllable_end - syllable_start) / max_stim_duration) * max_fig_width

    figsize = (fig_width, 10)
    fig = plt.figure(figsize=figsize, facecolor='w')
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20, wspace=0.20)

    gs = plt.GridSpec(100, 1)

    # plot the biosound features
    ax = plt.subplot(gs[:10])
    sprops = d['syllable_props'][syllable_index]
    aprops = USED_ACOUSTIC_PROPS

    vals = [sprops[a] for a in aprops]
    plt.axhline(0, c='k')
    for k, (aprop, v) in enumerate(zip(aprops, vals)):
        bx = k
        rgb = np.array(ACOUSTIC_PROP_COLORS_BY_TYPE[aprop]).astype('int')
        clr_hex = to_hex(*rgb)
        plt.bar(bx, v, color=clr_hex, alpha=0.7)
    ax.xaxis.tick_top()
    # plt.xticks(range(len(aprops)), aprops, rotation=45, fontsize=6)
    plt.xticks([])

    # plot the spectrogram
    ax = plt.subplot(gs[15:40])
    spec = d['spec']
    spec[spec < np.percentile(spec, 15)] = 0

    # the spectogram is already log transformed, make sure log=False and dBNoise=None
    plot_spectrogram(d['spec_t'], d['spec_freq'] * 1e-3, spec, ax=ax, colormap='SpectroColorMap',
                                       ticks=False, log=False, dBNoise=None, colorbar=False)

    plt.xlim(syllable_start, syllable_end)

    # plot the raw LFP
    ax = plt.subplot(gs[45:70])

    sr = d['lfp_sample_rate']
    raw_lfp = d['lfp'][trial_index, :, :]
    lfp_t = np.arange(raw_lfp.shape[1]) / sr
    nelectrodes, nt = raw_lfp.shape
    lfp_i = (lfp_t >= syllable_start) & (lfp_t <= syllable_end)

    voffset = 5
    for n in range(nelectrodes):
        plt.plot(lfp_t[lfp_i], raw_lfp[nelectrodes - n - 1, :][lfp_i] + voffset * n, 'k-', linewidth=2.0, alpha=0.75)
    plt.axis('tight')
    ytick_locs = np.arange(nelectrodes) * voffset
    plt.yticks(ytick_locs, list(reversed(d['electrode_order'])))
    plt.xticks([])

    # plot the spike train raster
    ax = plt.subplot(gs[75:])

    spike_mat = d['spikes'] # list-of-lists with shape (num_trials, num_neurons)
    print('# of neurons: ',len(spike_mat))
    raw_spikes = list()
    for spike_train in spike_mat[trial_index]:
        i = (spike_train >= syllable_start) & (spike_train <= syllable_end)
        raw_spikes.append(spike_train[i] - syllable_start)
    plt.xticks([])

    plot_raster(raw_spikes, ax=ax, duration=syllable_end-syllable_start,
                bin_size=0.001, time_offset=0.0, ylabel='', groups=None,
                bgcolor=None, spike_color='k')
    plt.xticks([])


def draw_figures():

    bird = 'GreBlu9508M'
    block = 'Site1'
    segment = 'Call1'
    hemi = 'L'

    data_dir='/auto/tdrive/mschachter/data'
    bdir = os.path.join(data_dir, bird)
    tdir = os.path.join(bdir, 'transforms')

    # pre-load the data files
    bs_file = os.path.join(tdir, 'BiosoundTransform_%s.h5' % bird)
    biosound = BiosoundTransform.load(bs_file)

    se_file = os.path.join(tdir, 'StimEvent_%s_%s_%s_%s.h5' % (bird,block,segment,hemi))
    print('Loading %s...' % se_file)
    stim_event = StimEventTransform.load(se_file, rep_types_to_load=['raw'])
    stim_event.zscore('raw')
    stim_event.segment_stims_from_biosound(bs_file)

    pcf_file = os.path.join(tdir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird,block,segment,hemi))
    print('Loading %s...' % pcf_file)
    pairwise_cf = PairwiseCFTransform.load(pcf_file)

    # stimulus 1, a distance call
    d1 = get_full_data('GreBlu9508M', 'Site1', 'Call1', 'L', 268, biosound=biosound, stim_event=stim_event, pairwise_cf=pairwise_cf)

    # stimulus 2, a tet
    d2 = get_full_data('GreBlu9508M', 'Site1', 'Call1', 'L', 258, biosound=biosound, stim_event=stim_event, pairwise_cf=pairwise_cf)

    # stimulus 3, a begging call
    d3 = get_full_data('GreBlu9508M', 'Site1', 'Call1', 'L', 188, biosound=biosound, stim_event=stim_event, pairwise_cf=pairwise_cf)

    # plot stimulus 1, syllable 1, trial 2
    plot_single_trial_data(d1, 1, 2)

    # plot stimulus 2, syllable 0, trial 4
    plot_single_trial_data(d2, 0, 4)

    # plot stimulus 3, syllable 0, trial 0
    plot_single_trial_data(d3, 0, 0)

    plt.show()

    # old figure 4 plot
    # plot_full_data(d, 1)


if __name__ == '__main__':
    spec_colormap()
    draw_figures()