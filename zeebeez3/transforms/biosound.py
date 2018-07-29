import os
import h5py

import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt

from soundsig.signal import break_envelope_into_events
from soundsig.sound import temporal_envelope, BioSound, plot_spectrogram, log_transform, spec_colormap
from soundsig.timefreq import gaussian_stft
from zeebeez3.core.experiment import Experiment

from zeebeez3.core.utils import CALL_TYPE_COLORS, CALL_TYPE_SHORT_NAMES


class BiosoundTransform(object):
    """
        Segments all the stimuli in an Experiment and then computes their acoustic features using the
        soundsig.sound.BioSound class.
    """

    def __init__(self):

        self.stim_data = None
        self.stim_df = None
        self.acoustic_props = ['fund', 'fund2', 'sal', 'voice2percent', 'maxfund', 'minfund', 'cvfund',
                               'meanspect', 'stdspect', 'skewspect', 'kurtosisspect', 'entropyspect',
                               'q1', 'q2', 'q3', 'meantime',
                               'stdtime', 'skewtime', 'kurtosistime', 'entropytime', 'maxAmp']
        self.bird = None

    def transform(self, experiment, stim_types_to_segment=('Ag', 'Di', 'Be', 'DC', 'Te', 'Ne', 'LT', 'Th', 'song'),
                  plot=False, excluded_types=tuple()):

        assert isinstance(experiment, Experiment), 'experiment argument must be an instance of class Experiment!'

        self.bird = experiment.bird_name
        all_stim_ids = list()
        # iterate through the segments and get the stim ids from each epoch table
        for seg in experiment.get_all_segments():
            etable = experiment.get_epoch_table(seg)
            stim_ids = etable['id'].unique()
            all_stim_ids.extend(stim_ids)

        stim_ids = np.unique(all_stim_ids)

        stim_data = {'stim_id': list(), 'stim_type': list(), 'start_time': list(), 'end_time': list(), 'order': list()}

        for aprop in self.acoustic_props:
            stim_data[aprop] = list()

        # specify type-specific thresholds for segmentation
        seg_params = {'default': {'min_thresh': 0.05, 'max_thresh': 0.25},
                      'Ag': {'min_thresh': 0.05, 'max_thresh': 0.10},
                      'song': {'min_thresh': 0.05, 'max_thresh': 0.10},
                      'Di': {'min_thresh': 0.15, 'max_thresh': 0.20}}

        for stim_id in stim_ids:

            print('Transforming stimulus {}'.format(stim_id))

            # get sound type
            si = experiment.stim_table['id'] == str(stim_id)
            assert si.sum() == 1, "Zero or more than one stimulus defined for id=%d, (si.sum()=%d)" % (
            stim_id, si.sum())
            stim_type = experiment.stim_table['type'][si].values[0]
            if stim_type == 'call':
                stim_type = experiment.stim_table['callid'][si].values[0]

            if stim_type in excluded_types:
                continue

            # get the stimulus waveform and sample rate
            sound = experiment.sound_manager.reconstruct(stim_id)
            waveform = np.array(sound.squeeze())
            sample_rate = float(sound.samplerate)
            stim_dur = len(waveform) / sample_rate

            if stim_type in stim_types_to_segment:
                # compute the spectrogram of the stim
                spec_sample_rate = 1000.
                spec_t, spec_freq, spec_stft, spec_rms = gaussian_stft(waveform, sample_rate, 0.007,
                                                                       1.0 / spec_sample_rate)
                spec = np.abs(spec_stft)
                nz = spec > 0
                spec[nz] = 20 * np.log10(spec[nz]) + 50
                spec[spec < 0] = 0

                # compute the amplitude envelope
                amp_env = spec_rms
                amp_env -= amp_env.min()
                amp_env /= amp_env.max()

                # segment the amplitude envelope
                minimum_isi = int(4e-3 * spec_sample_rate)
                if stim_type in seg_params:
                    min_thresh = seg_params[stim_type]['min_thresh']
                    max_thresh = seg_params[stim_type]['max_thresh']
                else:
                    min_thresh = seg_params['default']['min_thresh']
                    max_thresh = seg_params['default']['max_thresh']
                syllable_times = break_envelope_into_events(amp_env, threshold=min_thresh, merge_thresh=minimum_isi,
                                                            max_amp_thresh=max_thresh)

                if plot:
                    plt.figure()
                    ax = plt.subplot(111)
                    plot_spectrogram(spec_t, spec_freq, np.abs(spec), ax=ax, fmin=300.0, fmax=8000.0,
                                     colormap=plt.cm.afmhot,
                                     colorbar=False)
                    sfd = spec_freq.max() - spec_freq.min()
                    amp_env *= sfd
                    amp_env += spec_freq.min()

                    tline = sfd * min_thresh + amp_env.min()
                    tline2 = sfd * max_thresh + amp_env.min()
                    plt.axhline(tline, c='w', alpha=0.50)
                    plt.axhline(tline2, c='w', alpha=0.50)

                    plt.plot(spec_t, amp_env, 'w-', linewidth=2.0, alpha=0.75)
                    for k, (si, ei, max_amp) in enumerate(syllable_times):
                        plt.plot(spec_t[si], 0, 'go', markersize=8)
                        plt.plot(spec_t[ei], 0, 'ro', markersize=8)
                    plt.title('stim %d, %s, minimum_isi=%d' % (stim_id, stim_type, minimum_isi))
                    plt.axis('tight')
                    plt.show()

                the_order = 0
                for k, (si, ei, max_amp) in enumerate(syllable_times):

                    sii = int((si / spec_sample_rate) * sample_rate)
                    eii = int((ei / spec_sample_rate) * sample_rate)

                    s = waveform[sii:eii]
                    if len(s) < 1024:
                        continue

                    bs = BioSound(soundWave=s, fs=sample_rate)
                    bs.spectrum(f_high=8000.)
                    bs.ampenv()
                    bs.fundest()

                    stime = sii / sample_rate
                    etime = eii / sample_rate

                    stim_data['stim_id'].append(stim_id)
                    stim_data['stim_type'].append(stim_type)
                    stim_data['start_time'].append(stime)
                    stim_data['end_time'].append(etime)
                    stim_data['order'].append(the_order)
                    the_order += 1

                    for aprop in self.acoustic_props:
                        aval = getattr(bs, aprop)
                        if aval is None:
                            aval = -1
                        stim_data[aprop].append(aval)

            else:

                bs = BioSound(soundWave=waveform, fs=sample_rate)
                bs.spectrum(f_high=8000.)
                bs.ampenv()
                bs.fundest()

                stim_data['stim_id'].append(stim_id)
                stim_data['stim_type'].append(stim_type)
                stim_data['start_time'].append(0)
                stim_data['end_time'].append(stim_dur)
                stim_data['order'].append(0)

                for aprop in self.acoustic_props:
                    aval = getattr(bs, aprop)
                    stim_data[aprop].append(aval)

            self.stim_data = stim_data
            self.stim_df = pd.DataFrame(self.stim_data)

    def save(self, output_file):
        hf = h5py.File(output_file, 'w')

        col_names = list(self.stim_data.keys())
        hf.attrs['col_names'] = col_names
        hf.attrs['bird'] = self.bird

        for cname in col_names:
            try:
                hf[cname] = np.array(self.stim_data[cname])
            except TypeError:
                print('TypeError for column {}'.format(cname))
                print(self.stim_data[cname])
                raise

        hf.close()

    @classmethod
    def load(clz, bs_file):
        hf = h5py.File(bs_file, 'r')
        bst = BiosoundTransform()
        bst.bird = hf.attrs['bird']
        col_names = hf.attrs['col_names']
        bst.stim_data = dict()
        for cname in col_names:
            bst.stim_data[cname] = np.array(hf[cname])
        bst.stim_df = pd.DataFrame(bst.stim_data)
        hf.close()

        return bst

    def plot(self, exclude_stats=('kurtosisspect', 'skewtime', 'kurtosistime'), exclude_types=('mlnoise',)):
        nrows = 3
        ncols = 6

        i = np.ones(len(self.stim_df), dtype='bool')
        for stype in exclude_types:
            i &= self.stim_df['stim_type'] != stype

        df = self.stim_df[i]
        dur = df['end_time'] - df['start_time']

        stimsets = list(zip(df['stim_id'].values, df['stim_type'].values, dur))
        stimsets.sort(key=operator.itemgetter(-1), reverse=True)
        print('Stim durations by id/type:')
        for sid, stype, sdur in stimsets:
            print('{} (P{): {}ms'.format(sid, stype, sdur * 1e3))

        print([sid for sid, stype, sdur in stimsets if sdur > 1])

        stim_types = df['stim_type'].unique()
        nstims_per_type = list()
        for st in stim_types:
            i = df['stim_type'] == st
            nstims_per_type.append(i.sum())
        lst = list(zip(stim_types, nstims_per_type))
        lst.sort(key=operator.itemgetter(1))
        stim_types = [st for st, n in lst]
        nstims_per_type = [n for st, n in lst]

        # plot barplot of # of examples per stim type
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.bar(np.arange(len(nstims_per_type)), nstims_per_type, facecolor='#7E56C3', width=0.60, align='center')
        for k, (st, n) in enumerate(lst):
            plt.text(k - 0.25, n + 5, '%d' % n)
        plt.xticks(np.arange(len(stim_types)), stim_types)
        plt.xlim(-1, len(stim_types))
        plt.ylabel('# of Examples')

        # plot histogram of syllable durations
        fig = plt.figure()
        plt.hist(dur, bins=60, color='g')
        plt.title('Syllable Durations (# of syllables = %d, min=%dms, median=%dms, q3=%dms)' %
                  (len(dur), dur.min() * 1e3, np.median(dur) * 1e3, np.percentile(dur, 75) * 1e3))

        # plot acoustic properties boxplots
        fig = plt.figure()
        fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20)
        sp = 0
        for k, aprop in enumerate(self.acoustic_props):
            if aprop in exclude_stats:
                continue
            ax = plt.subplot(nrows, ncols, sp + 1)
            sp += 1
            self.boxplot_for_stim_stat(df, aprop, ax=ax, title=aprop)

    def boxplot_for_stim_stat(self, df, key_name, ax=None, title=None):

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        stim_classes = df['stim_type'].unique()

        d = dict()
        for sc in stim_classes:
            i = df['stim_type'] == sc
            x = df[key_name][i].values
            ii = x != -1
            d[sc] = x[ii]

        dmeans = [(sc, vals.mean()) for sc, vals in list(d.items())]
        dmeans.sort(key=operator.itemgetter(1))
        key_order = [x[0] for x in dmeans]
        key_labels = [CALL_TYPE_SHORT_NAMES[key] for key in key_order]

        vals = [d[key] for key in key_order]

        bp = plt.boxplot(vals)
        plt.setp(bp['boxes'], lw=0, color='k')
        plt.setp(bp['whiskers'], lw=3.0, color='k')
        plt.xticks(list(range(1, len(key_labels) + 1)), key_labels)
        if title is not None:
            plt.title(title)

        for k, stim_class in enumerate(key_order):
            box = bp['boxes'][k]
            boxX = list()
            boxY = list()
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            boxPolygon = plt.Polygon(boxCoords, facecolor=CALL_TYPE_COLORS[stim_class])
            ax.add_patch(boxPolygon)

    def export_for_classifier(self, output_file, exclude_types=('mlnoise',)):

        # read the aggregate stim file to get information about bird that emitted each call
        agg_stim_df = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/stim_data.csv')
        i = agg_stim_df.bird == self.bird
        index2emitter = list(agg_stim_df[i].emitter.unique())
        print('index2emitter=', index2emitter)

        stim_ids = self.stim_df['stim_id'].values
        stim_orders = self.stim_df['order'].values
        full_stim_ids = list(zip(stim_ids, stim_orders))

        index2id = full_stim_ids
        stim_types = self.stim_df['stim_type'].values
        index2type = [st for st in np.unique(stim_types) if st not in exclude_types]

        # construct a feature matrix and output matrix
        X = list()
        Y = list()
        S = list()

        for k, ((stim_id, stim_order), stim_type) in enumerate(zip(full_stim_ids, stim_types)):

            if stim_type in exclude_types:
                continue

            # get the bird that emitted this call
            i = (agg_stim_df['id'] == stim_id)
            emitter = agg_stim_df['emitter'][i].values[0]

            # append integer-valued information about the stim id, type,
            Y.append(
                (index2type.index(stim_type), index2id.index((stim_id, stim_order)), 0, index2emitter.index(emitter)))

            ii = (self.stim_df['stim_id'] == stim_id) & (self.stim_df['order'] == stim_order)
            assert ii.sum() == 1, "Too many results for stim_id=%d and stim_order=%d" % (stim_id, stim_order)
            x = list()
            for j, aprop in enumerate(self.acoustic_props):
                x.append(self.stim_df[aprop][ii].values[0])

            X.append(np.array(x))
            S.append(0)

        hf = h5py.File(output_file, 'w')
        hf['X'] = np.array(X)
        hf['S'] = np.array(S)
        hf['Y'] = np.array(Y)
        hf.attrs['integer2type'] = index2type
        hf.attrs['integer2id'] = index2id
        hf.attrs['integer2bird'] = index2emitter
        hf.close()


class ManualSegmenter(object):

    def __init__(self):
        pass

    def segment(self, exp_file, stim_file, output_file):

        exp = Experiment.load(exp_file, stim_file)
        spec_colormap()

        all_stim_ids = list()
        for ekey in list(exp.epoch_table.keys()):
            etable = exp.epoch_table[ekey]
            stim_ids = etable['id'].unique()
            all_stim_ids.extend(stim_ids)
        stim_ids = np.unique(all_stim_ids)

        # read all the stims that are already segmented
        finished_stims = list()
        if os.path.exists(output_file):
            f = open(output_file, 'r')
            lns = f.readlines()
            f.close()
            finished_stims = [int(x.split(',')[0]) for x in lns if len(x) > 0]
            print('finished_stims=', finished_stims)

        # manually segment the stimuli
        print('# of stims: %d' % len(stim_ids))
        with open(output_file, 'a') as ofd:
            for stim_id in stim_ids:
                if stim_id in finished_stims:
                    continue
                stimes = self.show_stim(exp, stim_id)
                ofd.write('{},{}\n'.format(stim_id, ','.join(['%f' % s for s in stimes])))
                ofd.flush()

    def show_stim(self, exp, stim_id):

        # get the raw sound pressure waveform
        wave = exp.sound_manager.reconstruct(stim_id)
        wave_sr = wave.samplerate
        wave = np.array(wave).squeeze()
        wave_t = np.arange(len(wave)) / wave_sr

        # compute the amplitude envelope
        amp_env = temporal_envelope(wave, wave_sr, cutoff_freq=200.0)
        amp_env /= amp_env.max()

        # compute the spectrogram
        spec_sr = 1000.
        spec_t, spec_freq, spec, spec_rms = gaussian_stft(wave, float(wave_sr), 0.007, 1. / spec_sr,
                                                          min_freq=300.,
                                                          max_freq=8000.)
        spec = np.abs(spec) ** 2
        log_transform(spec, dbnoise=70)

        spec_ax = None
        click_points = list()
        fig = None

        def _render():
            plt.sca(spec_ax)
            plt.cla()
            plot_spectrogram(spec_t, spec_freq, spec, ax=spec_ax, colorbar=False, fmin=300., fmax=8000.,
                             colormap='SpectroColorMap')
            plt.plot(wave_t, amp_env * 8000, 'k-', linewidth=3.0, alpha=0.7)
            plt.axis('tight')
            for k, cp in enumerate(click_points):
                snum = int(k / 2)
                plt.axvline(cp, c='k', linewidth=2.0, alpha=0.8)
                plt.text(cp, 7000., str(snum), fontsize=14)
            plt.draw()

        def _onclick(event):
            click_points.append(event.xdata)
            _render()

        def _onkey(event):
            _k = str(event.key)
            if _k == 'delete':
                if len(click_points) > 0:
                    click_points.remove(click_points[-1])
                    _render()

        figsize = (23, 12)
        fig = plt.figure(figsize=figsize)
        bpress_cid = fig.canvas.mpl_connect('button_press_event', _onclick)
        kpress_cid = fig.canvas.mpl_connect('key_press_event', _onkey)

        gs = plt.GridSpec(100, 1)

        ax = plt.subplot(gs[:30, 0])
        plt.plot(wave_t, wave, 'k-')
        plt.plot(wave_t, amp_env * wave.max(), 'r-', linewidth=2.0, alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Waveform')
        plt.axis('tight')

        spec_ax = plt.subplot(gs[45:, 0])
        _render()

        plt.show()

        assert len(click_points) % 2 == 0, "Must have an even number of click points!"
        return click_points


def compare_stims(exp_file, stim_file, seg_file, bs_file):
    exp = Experiment.load(exp_file, stim_file)
    spec_colormap()

    all_stim_ids = list()
    for ekey in list(exp.epoch_table.keys()):
        etable = exp.epoch_table[ekey]
        stim_ids = etable['id'].unique()
        all_stim_ids.extend(stim_ids)
    stim_ids = np.unique(all_stim_ids)

    # read the manual segmentation data
    man_segs = dict()
    with open(seg_file, 'r') as f:
        lns = f.readlines()
        for ln in lns:
            x = ln.split(",")
            stim_id = int(x[0])
            stimes = [float(f) for f in x[1:]]
            assert len(stimes) % 2 == 0, "Uneven # of syllables for stim %d" % stim_id
            ns = len(stimes) / 2
            man_segs[stim_id] = np.array(stimes).reshape([ns, 2])

    # get the automated segmentation
    algo_segs = dict()
    bst = BiosoundTransform.load(bs_file)
    for stim_id in stim_ids:
        i = bst.stim_df.stim_id == stim_id
        d = list(zip(bst.stim_df[i].start_time, bst.stim_df[i].end_time))
        d.sort(key=operator.itemgetter(0))
        algo_segs[stim_id] = np.array(d)

    for stim_id in stim_ids:

        # get the raw sound pressure waveform
        wave = exp.sound_manager.reconstruct(stim_id)
        wave_sr = wave.samplerate
        wave = np.array(wave).squeeze()
        wave_t = np.arange(len(wave)) / wave_sr

        # compute the amplitude envelope
        amp_env = temporal_envelope(wave, wave_sr, cutoff_freq=200.0)
        amp_env /= amp_env.max()

        # compute the spectrogram
        spec_sr = 1000.
        spec_t, spec_freq, spec, spec_rms = gaussian_stft(wave, float(wave_sr), 0.007, 1. / spec_sr,
                                                          min_freq=300.,
                                                          max_freq=8000.)
        spec = np.abs(spec) ** 2
        log_transform(spec, dbnoise=70)

        figsize = (23, 12)
        plt.figure(figsize=figsize)

        ax = plt.subplot(2, 1, 1)
        plot_spectrogram(spec_t, spec_freq, spec, ax=ax, colorbar=False, fmin=300., fmax=8000.,
                         colormap='SpectroColorMap')
        for k, (stime, etime) in enumerate(algo_segs[stim_id]):
            plt.axvline(stime, c='k', linewidth=2.0, alpha=0.8)
            plt.axvline(etime, c='k', linewidth=2.0, alpha=0.8)
            plt.text(stime, 7000., str(k + 1), fontsize=14)
            plt.text(etime, 7000., str(k + 1), fontsize=14)
        plt.title('Algorithm Segmentation')
        plt.axis('tight')

        ax = plt.subplot(2, 1, 2)
        plot_spectrogram(spec_t, spec_freq, spec, ax=ax, colorbar=False, fmin=300., fmax=8000.,
                         colormap='SpectroColorMap')
        for k, (stime, etime) in enumerate(man_segs[stim_id]):
            plt.axvline(stime, c='k', linewidth=2.0, alpha=0.8)
            plt.axvline(etime, c='k', linewidth=2.0, alpha=0.8)
            plt.text(stime, 7000., str(k + 1), fontsize=14)
            plt.text(etime, 7000., str(k + 1), fontsize=14)
        plt.title('Manual Segmentation')
        plt.axis('tight')

        plt.show()


if __name__ == '__main__':
    exp_name = 'GreBlu9508M'
    data_dir = '/auto/tdrive/mschachter/data/%s' % exp_name
    exp_file = os.path.join(data_dir, '%s.h5' % exp_name)
    stim_file = os.path.join(data_dir, 'stims.h5')

    bst_file = os.path.join(data_dir, 'transforms', 'BiosoundTransform_%s.h5' % exp_name)

    # experiment = Experiment.load(exp_file, stim_file)
    # bst = BiosoundTransform()
    # bst.transform(experiment, plot=False)
    # bst.save(bst_file)

    # bst = BiosoundTransform.load(bst_file)
    # bst.export_for_classifier('/tmp/preprocess_biosound_GreBlu9508M.h5')
    # bst.plot()
    # plt.show()

    # ms = ManualSegmenter()
    # ms.segment(exp_file, stim_file, os.path.join(data_dir, 'stim_segmentation.csv'))
    compare_stims(exp_file, stim_file, os.path.join(data_dir, 'stim_segmentation.csv'), bst_file)
