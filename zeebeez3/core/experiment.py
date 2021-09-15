import os
import time
import operator

import numpy as np
from neo import NeoHdf5IO
from pandas import DataFrame

from soundsig.spikes import spike_trains_to_matrix
from soundsig.timefreq import gaussian_stft, log_spectrogram


class Experiment(object):

    def __init__(self):
        self.blocks = None
        self.bird_name = None
        self.file_name = None
        self.stimulus_file_name = None
        self.segment_durations = None
        self.stim_table = None
        self.epoch_table = None
        self.stim_manager = None
        self.lfp_sample_rate = None
        self.segment_specs = None

    def init_from_file(self, neo_file_name, stim_file):

        self.file_name = neo_file_name
        self.stimulus_file_name = os.path.join(stims_dir, 'stims.csv')
        assert os.path.exists(neo_file_name), "No such file: {}".format(neo_file_name)
        assert os.path.exists(self.stimulus_file_name), "No such file: {}".format(self.stimulus_file_name)

        #read the neo file
        with NeoHdf5IO(neo_file_name) as f:
            self.blocks = f.read_all_blocks()

        #construct a sound manager and construct a DataFrame of stimulus data
        if stim_file_name is not None:
            self.sound_manager = SoundManager(HDF5Store, stim_file_name, db_args={'read_only':read_only_stims})
            self.stim_table = stim_file_to_pandas(stim_file_name)
        else:
            self.sound_manager = None
            self.stim_table = None

        #gather up the epoch data (stim start and end times, # of trials) for each segment
        self.epoch_table = dict()
        for blk in self.blocks:
            for seg in blk.segments:
                sname = segment_to_unique_name(seg)
                self.epoch_table[sname] = epoch_data_to_pandas(seg)

                sr,dur = self.get_sample_rate_and_duration(seg)

                if self.lfp_sample_rate is None:
                    self.lfp_sample_rate = sr

                if 'duration' not in seg.annotations:
                    seg.annotations['duration'] = dur

        #set the bird name
        self.bird_name = self.blocks[0].annotations['bird_name']

        if stim_file_name is not None:
            #create SegmentSpectrogram objects that can serve up slices of spectrogram
            self.init_segment_spectrograms()

    def get_sample_rate_and_duration(self, seg):
        try:
            #get duration from an example LFP
            sig = seg.analogsignals[0]
        except IndexError:
            #signals aren't attached to segments, they're attached to RecordingChannelGroups,
            #this might be a bug that needs to be fixed.
            rcg = seg.block.recordingchannelgroups[0]
            rc = rcg.recordingchannels[0]
            sig = rc.analogsignals[0]

        sr = sig.sampling_rate
        return float(sr), float(len(sig) / sr)

    def init_segment_spectrograms(self, window_length=0.007, sample_rate=1000.0, min_freq=300.0, max_freq=8000.0, log=False):
        """ A SegmentSpectrogram object is computed when the Experiment object is initialized, which
            prevents someone from setting the spectrogram sample rate. This function overrides the
            defaut spectrogram sample rate (which is 1000Hz).
        """

        if self.segment_specs is not None:
            del self.segment_specs
        self.segment_specs = dict()
        for block in self.blocks:
            for seg in block.segments:
                seg_uname = segment_to_unique_name(seg)
                self.segment_specs[seg_uname] = SegmentSpectrogram(self, seg, sample_rate=sample_rate,
                                                                   window_length=window_length,
                                                                   min_freq=min_freq, max_freq=max_freq)
                if log:
                    self.segment_specs[seg_uname].load_all()
                    self.segment_specs[seg_uname].log_transform()

    def get_epoch_table(self, seg):
        """ Get a Pandas DataFrame that contains data about the start and end times of stimulus presentations, as well
            as trial number.
        """

        seg_uname = segment_to_unique_name(seg)
        assert seg_uname in self.epoch_table, "Segment %s doesn't exist in experiment %s!" % (seg.name, self.bird_name)
        return self.epoch_table[seg_uname]

    def get_segment(self, block_name, segment_name):

        #find the right block
        blist = [b for b in self.blocks if b.name == block_name]
        if len(blist) == 0:
            return None
        block = blist[0]

        #find the right segment
        slist = [s for s in block.segments if s.name == segment_name]
        if len(slist) == 0:
            return None
        return slist[0]

    def get_all_segments(self):
        all_segments = list()
        for blk in self.blocks:
            for seg in blk.segments:
                all_segments.append(seg)
        return all_segments

    def get_channel(self, block, electrode):
        """ Get a RecordingChannel and RecordingChannelGroup objects for a given electrode. """

        #get the recording channel group that contains the electrode
        rcg_list = [rcg for rcg in block.recordingchannelgroups if electrode in rcg.channel_indexes]
        if len(rcg_list) == 0:
            return None

        rcg = rcg_list[0]

        rc = None
        rclist = [rc for rc in rcg.recordingchannels if rc.index == electrode]
        if len(rclist) > 0:
            rc = rclist[0]

        return rcg,rc

    def get_electrodes(self, block, rcg_name):
        """ Get a list of electrode names that are on a given RecordingChannelGroup.

        :param rcg_name: The name of the RecordingChannelGroup
        """

        rcg_list = [rcg for rcg in block.recordingchannelgroups if rcg.name == rcg_name]
        assert len(rcg_list) == 1, "Can't find RecordingChannelGroup with name %s" %  rcg_name

        return rcg_list[0].channel_indexes

    def get_units(self, block, electrode, sort_code):
        """ Get Unit objects on a block for a given electrode and TDT sort_code. """

        rcg,rc = self.get_channel(block, electrode)

        def is_the_unit(u):
            # a little subfunction to find units whose spike trains match the electrode and sort_code given, across segments
            if str(u.annotations['sort_code']) != sort_code:
                return False
            it_is = True
            for st in u.spiketrains:
                it_is &= st.annotations['channel_index'] == electrode
            if len(u.spiketrains) == 0:
                it_is = False
            return it_is

        ulist = [u for u in rcg.units if is_the_unit(u)]

        return ulist

    def get_spectrogram_slice(self, seg, start_time, end_time):
        """ Retrieves the stimulus spectrogram over a duration of time for a given segment. """
        seg_uname = segment_to_unique_name(seg)
        seg_spec = self.segment_specs[seg_uname]
        stim_spec_t,stim_spec_freq,stim_spec = seg_spec.get_spectrogram(float(start_time), float(end_time))

        return stim_spec_t,stim_spec_freq,stim_spec

    def get_spike_shapes(self, seg, sort_code='single', rcg_names=None, flip=False):
        """ Get the spike shapes on all electrodes for a given segment.

        :param seg: The Segment to get spike shapes from
        :param sort_code: The sort code used to identify units on the electrodes.
        :param rcg_names: The RecordingChannelGroup names to select electrodes from.
        :param flip: Choose one spike shape as a template, and flip all the other spike shapes so they have a
                     positive correlation with the template.
        """

        spike_shapes = list()
        index2electrode = list()
        blk = seg.block
        for rcg in blk.recordingchannelgroups:

            if rcg.name not in rcg_names:
                continue

            for electrode in rcg.channel_indexes:
                units = self.get_units(seg.block, electrode, sort_code)
                for u in units:
                    st = u.spiketrains[0]
                    W = np.array(st.waveforms)
                    nt,nw = W.shape
                    spike_shape = W.mean(axis=0)
                    spike_shapes.append(spike_shape)
                    index2electrode.append(electrode)

        spike_shapes = np.array(spike_shapes)
        if flip:
            ss_template = spike_shapes[0, :]
            for k,ss in enumerate(spike_shapes[1:, :]):
                cc = np.corrcoef(ss_template, ss)[0, 1]
                if cc < 0:
                    spike_shapes[k+1, :] = -ss

        return spike_shapes,index2electrode

    def get_spike_slice(self, seg, start_time, end_time, sort_code="0", as_matrix=False, bin_size=1e-3, rcg_names=None):
        """ Get the multielectrode spike times for all channels on each array.

        :param seg: The Segment to get the data from.
        :param start_time: The start time of the slice.
        :param end_time: The end time of the slice.
        :param sort_code: The unit sort code to match.
        :param as_matrix: Whether to return the spike times as a matrix
        :param bin_size: if as_matrix=True, then the bin size in seconds used to create the spike matrix
        :param rcg_names: A list of RecordingChannelGroup names to obtain spikes from.

        :return: spikes: a dictionary where the key is the name of a RecordingChannelGroup. The value
            is a tuple (spike_trains, spike_train_groups). spike_trains is a list of arrays of spike
            times, one for each Unit on the array. spike_groups is a dictionary that maps a channel
            index to the indices in spike_trains that belong to that electrode.
        """

        if rcg_names is None:
            rcg_names = [rcg.name for rcg in seg.block.recordingchannelgroups]
        rcgs = [rcg for rcg in seg.block.recordingchannelgroups if rcg.name in rcg_names]

        spikes = dict()
        for rcg in rcgs:
            spike_train_index = 0
            spike_trains = list()
            spike_train_groups = dict()
            for electrode in rcg.channel_indexes:
                units = self.get_units(seg.block, electrode, sort_code)
                for u in units:
                    # get the spike train that corresponds to the segment
                    stlist = [st for st in u.spiketrains if st.segment.name == seg.name]
                    if len(stlist) > 0:
                        # get the spike times and append them to the array
                        st = np.array(stlist[0].times)
                        spike_trains.append(st[(st >= start_time) & (st <= end_time)])
                        # assign this spike train to a group defined by it's electrode
                        key = electrode
                        if key not in spike_train_groups:
                            spike_train_groups[key] = list()
                        spike_train_groups[key].append(spike_train_index)
                        spike_train_index += 1

            spikes[rcg.name] = (spike_trains, spike_train_groups)

        if as_matrix:
            stime = time.time()
            duration = end_time - start_time
            new_spikes = dict()
            for rcg_name,(spike_trains, spike_train_groups) in spikes.iteritems():
                # build a spike count over time matrix
                spike_count = spike_trains_to_matrix(spike_trains, bin_size, start_time, duration)
                new_spikes[rcg_name] = (spike_count, spike_train_groups)
            spikes = new_spikes

        return spikes

    def get_lfp_slice(self, seg, start_time, end_time, zscore=False):
        """ Get a slice of the multi-electrode LFP in matrix form.

        :param seg: The Segment to get the data from.
        :param start_time: The start time of the slice.
        :param end_time: The end time of the slice.

        :return: lfps: a dictionary where the key is the name of a RecordingChannelArray. The value
            is a tuple (electrode_indices, lfp_matrix, sample_rate), where electrode_indices is a list that contains
            the channel index of each row of lfp_matrix. lfp_matrix is a matrix of shape
            (num_electrodes, num_time_points) which contains the raw multi-electrode LFP.
        """

        lfps = dict()
        sample_rate,exp_dur = self.get_sample_rate_and_duration(seg)
        for rcg in seg.block.recordingchannelgroups:

            electrode_indices = list()
            lfp_matrix = list()
            for electrode in rcg.channel_indexes:
                rgc2,rc = self.get_channel(seg.block, electrode)
                slist = [sig for sig in rc.analogsignals if sig.segment.name == seg.name]
                if len(slist) == 0:
                    continue

                sig = slist[0]
                sig_mean = 0
                sig_std = 1
                if zscore:
                    sig_arr = np.array(sig)
                    sig_mean = sig_arr.mean()
                    sig_std = sig_arr.std(ddof=1)
                si = int(start_time*self.lfp_sample_rate)
                ei = int(end_time*self.lfp_sample_rate)
                lfp_matrix.append((np.array(sig[si:ei]) - sig_mean) / sig_std)
                electrode_indices.append(electrode)

            lfps[rcg.name] = (electrode_indices, np.array(lfp_matrix), sample_rate)

        return lfps

    def get_single_lfp_slice(self, segment, electrode, start_time, end_time):
        """ Get a single LFP.

        :param seg: The Segment to get the data from.
        :param electrode: The index of the electrode to select.
        :param start_time: The start time of the slice.
        :param end_time: The end time of the slice.
        """

        rgc,rc = self.get_channel(segment.block, electrode)
        slist = [sig for sig in rc.analogsignals if sig.segment.name == segment.name]
        if len(slist) == 0:
            return None

        sig = slist[0]
        si = int(start_time*self.lfp_sample_rate)
        ei = int(end_time*self.lfp_sample_rate)
        return np.array(sig[si:ei])

    def save(self, file_name=None):
        """ Save the blocks to a neo hdf5 file. """
        if file_name is None:
            assert self.file_name is not None, "You must specify a file name to write the experiment to!"
            file_name = self.file_name

        #reset any relationships if they have been modified
        for block in self.blocks:
            block.create_many_to_one_relationship(force=True, recursive=True)

        if os.path.exists(file_name):
            #delete the old file
            os.remove(file_name)

        #write a new file
        of = NeoHdf5IO(file_name)
        self.blocks.sort(key=operator.attrgetter("name"))
        of.write_all_blocks(self.blocks)
        of.close()

    @classmethod
    def load(cls, neo_file_name, stim_file_name=None, read_only_stims=True):
        """ Load an experiment file from a path on the filesystem.

        :param neo_file_name: The path to the neo hdf5 file.
        :param stim_file_name: The path to the neosound stimulus file.
        :return: an Experiment object.
        """
        assert os.path.exists(neo_file_name), "File %s doesn't exist." % neo_file_name

        if stim_file_name is not None:
            assert os.path.exists(stim_file_name), "File %s doesn't exist." % stim_file_name

        e = Experiment()
        e.init_from_file(neo_file_name, stim_file_name=stim_file_name, read_only_stims=read_only_stims)
        return e


def segment_to_unique_name(seg):
    """ Construct a name that is unique to the segment across all experiments. Segment
        name is of the form: bird_site_protocol.
    """
    return '%s_%s_%s' % (seg.block.annotations['bird_name'], seg.block.name, seg.name)


def segment_from_unique_name(blocks, uname):
    """ Take a list of blocks, and a unique segment name constructed by segment_to_unique_name, and
        return the Segment that corresponds to the name.
    """

    bird,block_name,seg_name = uname.split('_')

    for block in [b for b in blocks if b.annotations['bird_name'] == bird and b.name == block_name]:
        segs = [s for s in block.segments if s.name == seg_name]
        if len(segs) > 0:
            return segs[0]


def stim_file_to_pandas(stim_file):
    """ Reads a neosound stim file and turns it into a Pandas DataFrame. """
    sm = SoundManager(HDF5Store, stim_file)

    #get the annotations of every sound in the database
    sound_annotations = dict()
    all_columns = ['id']
    for sound_id in sm.database.list_ids():
        sound_annotations[sound_id] = sm.database.get_annotations(sound_id)
        sound_annotations[sound_id]['id'] = sound_id
        for name in sound_annotations[sound_id].keys():
            if name not in all_columns:
                all_columns.append(name)

    #construct a dataset with the annotations
    data = {name:list() for name in all_columns}
    for sound_id,annot in sound_annotations.items():
        for cname in all_columns:
            if cname not in annot:
                val = None
            else:
                val = annot[cname]
            data[cname].append(val)
    del sm

    return DataFrame(data)


def epoch_data_to_pandas(segment):
    """ Reads the epoch data for a segment and constructs a Pandas DataFrame from it. """

    #sort the epochs
    segment.epochs.sort(key=operator.attrgetter('time'))

    data = {'id':list(), 'start_time':list(), 'end_time':list(), 'trial':list()}
    trial_number = dict()
    for epoch in segment.epochs:
        stim_id = epoch.annotations['stim_id']
        if stim_id not in trial_number:
            trial_number[stim_id] = 0
        trial_number[stim_id] += 1
        data['id'].append(stim_id)
        data['start_time'].append(epoch.time)
        data['end_time'].append(epoch.time + epoch.duration)
        data['trial'].append(trial_number[stim_id])

    return DataFrame(data)


class SegmentSpectrogram(object):
    """ This class serves up slices of spectrogram for a given Segment. """

    def __init__(self, experiment, seg, window_length=0.007, sample_rate=1000.0, min_freq=300.0, max_freq=8000.0,
                 dbnoise=100, lazy_load=True):

        self.experiment = experiment
        self.segment = seg
        self.window_length = window_length
        self.freq_range = [min_freq, max_freq]
        self.sample_rate = sample_rate
        self.spec_inc = 1.0 / sample_rate
        self.epoch_table = self.experiment.get_epoch_table(self.segment)
        self.sound_manager = experiment.sound_manager
        self.lazy_load = lazy_load
        self.spec_freq = None
        self.dbnoise = dbnoise

        #get the stimulus ids for the segment
        self.stim_ids = np.unique([stim_id for stim_id in self.epoch_table['id']])

        #compute the spectrogram of each sound
        self.specs = dict()
        if not lazy_load:
            self.load_all()

        #load at least one spectrogram so self.spec_freq is initialized
        self.load_spectrogram(self.stim_ids[0])

    def load_all(self):
        for stim_id in self.stim_ids:
            self.load_spectrogram(stim_id)

    def load_spectrogram(self, stim_id):
        sound = self.sound_manager.reconstruct(stim_id)
        t,freq,timefreq,rms = gaussian_stft(sound.squeeze(), float(sound.samplerate), self.window_length, self.spec_inc, min_freq=self.freq_range[0], max_freq=self.freq_range[1])
        timefreq = np.abs(timefreq)**2
        self.specs[stim_id] = (t, freq, timefreq, sound)

        if self.spec_freq is None:
            self.spec_freq = freq
        else:
            assert np.abs(self.spec_freq - freq).sum() == 0, "Spectrogram frequency differs for stim %d" % stim_id

    def log_transform(self):
        # find the max across all loaded spectrograms
        max_pow = -np.inf
        for stim_id in self.stim_ids:
            t, freq, timefreq, sound = self.specs[stim_id]
            max_pow = max(max_pow, np.percentile(timefreq.ravel(), 99))

        for stim_id in self.stim_ids:
            t, freq, timefreq, sound = self.specs[stim_id]
            timefreq = log_spectrogram(timefreq / max_pow, offset=self.dbnoise)
            self.specs[stim_id] = (t, freq, timefreq, sound)

    def get_spectrogram(self, start_time, end_time, fixed_length=None, sample_rate=None):

        si = int(start_time*self.sample_rate)
        ei = int(end_time*self.sample_rate)
        if fixed_length is not None:
            ei = si + fixed_length

        #find all the stimuli that occur in the time interval
        stim_indices = (self.epoch_table['end_time'] > start_time) & (self.epoch_table['start_time'] < end_time)

        #load spectrograms if needed
        if self.lazy_load:
            for stim_id in self.epoch_table[stim_indices]['id'].unique():
                if stim_id not in self.specs:
                    self.load_spectrogram(stim_id)

        #construct empty spectrogram for the segment
        seg_spec = np.zeros([len(self.spec_freq), ei-si])

        for k,(stim_id,stime,etime) in self.epoch_table[stim_indices][ ['id', 'start_time', 'end_time'] ].T.iteritems():
            stime = float(stime)
            etime = float(etime)

            #print 'get_spectrogram: stim_id=%d, stime=%f, etime=%f' % (stim_id, stime, etime)
            spec_t, spec_freq, stim_spec, sound = self.specs[stim_id]
            tlen = stim_spec.shape[1]

            #get start and end indices for full spectrogram
            if stime >= start_time:
                si = int((stime - start_time) / self.spec_inc)
                ei = min(seg_spec.shape[1], si+tlen)
                #get start and end indices for this event's spectrogram
                eei = ei - si
                ssi = 0
            else:
                offset = int((start_time - stime)/self.spec_inc)
                si = 0
                ei = min(seg_spec.shape[1], tlen-offset)

                ssi = int((start_time - stime)/self.spec_inc)
                eei = ssi + (ei - si)

            #fill in the spectrogram
            try:
                seg_spec[:,si:ei] = stim_spec[:, ssi:eei]
            except ValueError:
                print('seg_spec.shape=',seg_spec.shape)
                print('stim_spec.shape=',stim_spec.shape)
                print('start_time=%0.6f, end_time=%0.6f, stime=%0.6f, etime=%0.6f, si=%d, ei=%d, ssi=%d, eei=%d' % \
                      (start_time, end_time, stime, etime, si, ei, ssi, eei))
                raise

        spec_t = np.arange(seg_spec.shape[1])/self.sample_rate + start_time
        return spec_t,self.spec_freq,seg_spec


if __name__ == '__main__':

    stim_data = stim_file_to_pandas("/tmp/tdt2neo/GreBlu9508M/stims.h5")
    print(stim_data)





