import os
import time

import h5py

import numpy as np

import matplotlib.pyplot as plt

from soundsig.memd import memd
from soundsig.signal import analytic_signal

from zeebeez3.core.experiment import Experiment, segment_to_unique_name
from zeebeez3.transforms.lfp import LFPTransform


class MEMDTransform(LFPTransform):

    def __init__(self):
        LFPTransform.__init__(self)

    def transform(self, experiment, segment, electrodes=None, start_time=None, end_time=None, **kwargs):
        """Run the multivariate empirical mode decomposition (MEMD) on multi-electrode LFPs.

           Args:
            experiment: The Experiment object the data comes from.

            segment: The Segment object that contains the LFP.

            electrodes: The electrodes to include.

            hemishperes (array): the hemispheres to include in the MEMD, possible values are ['L', 'R']

            start_time (float): the start time within the protocol of the LFP (seconds). If no time is specified, defaults to 0.0.

            end_time (float): the end time within the protocol of the LFP (seconds). If no time is given, defaults to end time of the protocol.

            nbands (int): the number of IMFs to find. Defaults to 6.

            num_noise_channels (int): if nonzero, the number of white noise channels to add to the signal before finding the IMFs. This
            method is called "Noise assisted MEMD" and is documented in Rehman and Mandic (2011). Defaults to zero.

            num_samps (int): the number of samples used in computing the mean envelope for the sifting process. Defaults to 100.

           Sets:

            self.imfs (np.ndarray): The IMFs, with dimensions NxT, where N is the number of electrodes, T is the number of time points.
            self.index2electrode (list): maps an electrode index from 0 to N-1 to an electrode number.
            self.sample_rate (float): the sample rate of the LFP in Hz.
        """

        #set keywords to defaults
        kw_params = {'nbands':6, 'num_noise_channels':0, 'num_samps':100, 'rseed':None, 'rcg_names':['L'],
                     'resolution':1.0, 'max_iterations':30}
        for key,val in kw_params.items():
            if key not in kwargs:
                kwargs[key] = val

        self.experiment = experiment
        self.segment = segment

        self.num_bands = kwargs['nbands']
        self.num_samps = kwargs['num_samps']
        self.resolution = kwargs['resolution']
        self.max_iterations = kwargs['max_iterations']
        self.num_noise_channels = kwargs['num_noise_channels']

        self.sample_rate,duration = self.experiment.get_sample_rate_and_duration(self.segment)

        self.rcg_names = kwargs['rcg_names']

        #set start and end times of the LFP
        t1 = 0.0
        t2 = duration
        if start_time is not None:
            t1 = start_time
        if end_time is not None:
            t2 = end_time
        self.start_time = t1
        self.end_time = t2

        if electrodes is None:
            electrodes = list()
            for rgc in self.segment.block.recordingchannelgroups:
                if rgc.name in self.rcg_names:
                    electrodes.extend(rgc.channel_indexes)

        #get LFPs
        start_index = int(t1*self.experiment.lfp_sample_rate)
        end_index = int(t2*self.experiment.lfp_sample_rate)
        lfps = np.zeros([len(electrodes), end_index-start_index])
        index2electrode = list()
        for k,electrode in enumerate(electrodes):
            index2electrode.append(electrode)
            lfps[k, :] = self.experiment.get_single_lfp_slice(self.segment, electrode, self.start_time, self.end_time)
        self.index2electrode = index2electrode

        #set random seed
        rseed = kwargs['rseed']
        if rseed is not None:
            np.random.seed(rseed)

        #perform MEMD on the electrodes
        stime = time.time()
        self.X = memd(lfps, self.num_bands, nsamps=self.num_samps, resolution=self.resolution,
                         max_iterations=self.max_iterations, num_noise_channels=self.num_noise_channels)

        # compute the analytic signal on each electrode wtihin each band
        self.Z = np.zeros_like(self.X, dtype='complex')
        for k in range(self.num_bands):
            for n in range(len(electrodes)):
                self.Z[k, n, :] = analytic_signal(self.X[k, n, :])

        etime = time.time() - stime
        print('MEMD elapsed time: %d seconds' % int(etime))

    def save(self, output_file):
        hf = h5py.File(output_file, 'w')

        hf['bands'] = self.X
        hf['bands_analytic'] = self.Z

        hf.attrs['num_noise_channels'] = self.num_noise_channels
        hf.attrs['num_bands'] = self.num_bands
        hf.attrs['resolution'] = self.resolution
        hf.attrs['max_iterations'] = self.max_iterations
        hf.attrs['num_samples'] = self.num_samps

        hf.attrs['index2electrode'] = self.index2electrode
        hf.attrs['rcg_names'] = self.rcg_names
        hf.attrs['sample_rate'] = self.sample_rate

        hf.attrs['start_time'] = self.start_time
        hf.attrs['end_time'] = self.end_time

        hf.attrs['experiment_file'] = self.experiment.file_name
        hf.attrs['stimulus_file'] = self.experiment.stimulus_file_name

        hf.attrs['block_name'] = self.segment.block.name
        hf.attrs['segment_name'] = self.segment.name

        hf.close()

    @classmethod
    def load(clz, output_file, load_real=True, load_complex=True):

        hf = h5py.File(output_file, 'r')

        mt = MEMDTransform()

        mt.num_samples = hf.attrs['num_samples']
        mt.num_noise_channels = hf.attrs['num_noise_channels']
        mt.num_bands = hf.attrs['num_bands']
        mt.resolution = hf.attrs['resolution']
        mt.max_iterations = hf.attrs['max_iterations']

        mt.index2electrode = hf.attrs['index2electrode']
        mt.sample_rate = hf.attrs['sample_rate']
        mt.rcg_names = hf.attrs['rcg_names']

        mt.start_time = hf.attrs['start_time']
        mt.end_time = hf.attrs['end_time']

        mt.X = None
        mt.Z = None
        mt.file_name = output_file

        # load up Experiment object
        stim_file_name = hf.attrs['stimulus_file']
        exp_file_name = hf.attrs['experiment_file']
        mt.experiment = Experiment.load(exp_file_name, stim_file_name)

        # get the right segment
        seg_name = hf.attrs['segment_name']
        mt.block_name = hf.attrs['block_name']
        mt.segment = mt.experiment.get_segment(mt.block_name, seg_name)

        if load_real:
            mt.X = np.array(hf['bands'])
        if load_complex:
            mt.Z = np.array(hf['bands_analytic'])

        hf.close()
        return mt


def split_segment_into_parts(segment, njobs=10, overlap=10.0):
    """
        MEMD is pretty expensive, so to speed things up, we split a single site into
        overlapping time segments, and run on each one separately.

        Args:
            segment: The Neo Segment to split up.
            jobs (int): the number of jobs to run in parallel, the number of segments to split the protocol response into
            overlap (float): time in seconds that segments overlap, used to deal with end effects

        Returns: segment_times (list): a list of pairs of start and end times, one pair for each segment

    """

    duration = segment.annotations['duration']
    time_per_segment = duration / njobs
    print('duration=%0.2fs, time_per_segment=%0.2f' % (duration, time_per_segment))

    #determine start and end time of each segment
    segment_times = list()
    for k in range(njobs):
        start_time = k*time_per_segment
        if k > 0:
            start_time -= overlap
        end_time = min(duration, (k+1)*time_per_segment)
        segment_times.append( (start_time, end_time) )

    return segment_times


def get_memd_output_filename(bird_name, block_name, segment_name, rcg_names, start_time=None, end_time=None):

    if start_time is None and end_time is None:
        return 'MEMD_%s_%s_%s_%s.h5' % (bird_name, block_name, segment_name, ','.join(rcg_names))
    else:
        return 'MEMD_%s_%s_%s_%s_%0.9f_%0.9f.h5' % (bird_name, block_name, segment_name, ','.join(rcg_names), start_time, end_time)


def find_missing(bird_name, segment, output_dir, njobs=10, overlap=10.0):

    seg_times = split_segment_into_parts(segment, njobs=njobs, overlap=overlap)

    missing_times = list()

    for rcg in ['L', 'R']:
        for start_time,end_time in seg_times:
            fname = get_memd_output_filename(bird_name, segment.block.name, segment.name, [rcg], start_time=start_time, end_time=end_time)
            output_file = os.path.join(output_dir, fname)

            if not os.path.exists(output_file):
                print('%s is missing!' % output_file)
                missing_times.append( (rcg, start_time, end_time) )

    return missing_times


def merge_segments(bird_name, segment, rcg_name='L', data_dir='/auto/tdrive/mschachter/data', njobs=10, overlap=10.0):

    seg_times = split_segment_into_parts(segment, njobs=njobs, overlap=overlap)
    output_dir = os.path.join(data_dir, bird_name, 'transforms')

    #load up the transform for each segment
    output_files = list()
    lowest_start_time = np.min([x[0] for x in seg_times])
    highest_end_time = np.max([x[1] for x in seg_times])
    for start_time,end_time in seg_times:
        fname = get_memd_output_filename(bird_name, segment.block.name, segment.name, [rcg_name], start_time, end_time)

        output_file = os.path.join(output_dir, fname)
        output_files.append(output_file)
        if not os.path.exists(output_file):
            raise Exception('Missing file %s, breaking!' % output_file)

    print('lowest_start_time=%0.6f, highest_end_time=%0.6f' % (lowest_start_time, highest_end_time))

    merged_arrays = dict()
    props_to_merge = ['bands', 'bands_analytic']
    for prop_name in props_to_merge:

        #peek into the first transform to get the shape of the data
        hf = h5py.File(output_files[0], 'r')
        data = np.array(hf[prop_name])
        data_dtype = data.dtype
        nbands,nelectrodes,nt = data.shape
        sample_rate = hf.attrs['sample_rate']
        hf.close()
        del data

        #construct an array to hold the merged data
        t1 = int(lowest_start_time*sample_rate)
        t2 = int(np.ceil(highest_end_time*sample_rate))
        d = t2 - t1
        merged_data = np.zeros([nbands, nelectrodes, d], dtype=data_dtype)

        for k,output_file in enumerate(output_files):

            #get properties from the segment file
            print('Reading file %s...' % output_files[k])
            hf = h5py.File(output_file, 'r')
            stime = hf.attrs['start_time']
            etime = hf.attrs['end_time']

            #get data from the segment file
            si = int(stime*sample_rate)
            ei = int(etime*sample_rate)
            data = np.array(hf[prop_name])
            hf.close()

            fnbands,fnelectrodes,fnt = data.shape
            assert fnbands == nbands
            assert fnelectrodes == nelectrodes
            assert fnt == ei - si

            if k > 0:
                #find the time that the overlap ends
                overlap_size = int(overlap*sample_rate)
            else:
                overlap_size = 0

            #get the non-overlapping portion of the data
            print('k=%d, overlap_size=%d, si=%d, ei=%d, ei-si=%d, (ei - si+overlap_size)=%d' % (k, overlap_size, si, ei, ei-si, ei-(si+overlap_size)))
            print('data.shape=',data.shape)
            merged_data[:, :, (si+overlap_size):ei] = data[:, :, overlap_size:]

            if k > 0:
                #linearly blend the overlapping segments from this output file and the previous one
                right_blend = np.arange(overlap_size, dtype='float') / (overlap_size-1) # goes from 0 to 1
                left_blend = right_blend[::-1]                                          # goes from 1 to 0
                print('k=%d, overlap_size=%d, si=%d, ei=%d' % (k, overlap_size, si, ei))

                merged_data[:, :, si:(si+overlap_size)] = left_blend*merged_data[:, :, si:(si+overlap_size)] + right_blend*data[:, :, :overlap_size]

        merged_arrays[prop_name] = merged_data

    #write the data to an output file
    output_file = os.path.join(output_dir, get_memd_output_filename(bird_name, segment.block.name, segment.name, [rcg_name]))
    print('Writing to file %s' % output_file)

    hf = h5py.File(output_file, 'w')
    hf.attrs['start_time'] = 0.0
    hf.attrs['end_time'] = highest_end_time

    #open the first file and copy it's attributes
    hf2 = h5py.File(output_files[0], 'r')
    for key,val in hf2.attrs.items():
        if key in ['start_time', 'end_time']:
            continue
        hf.attrs[key] = val
    hf2.close()

    #set the data properties
    for prop_name in props_to_merge:
        hf[prop_name] = merged_arrays[prop_name]

    hf.close()


if __name__ == '__main__':

    exp_name = 'YelBlu6903F'
    exp_file = '/auto/tdrive/mschachter/data/%s/%s.h5' % (exp_name, exp_name)
    stim_file = '/auto/tdrive/mschachter/data/%s/stims.h5' % exp_name
    output_dir = '/auto/tdrive/mschachter/data/%s/transforms' % exp_name
    exp = Experiment.load(exp_file, stim_file)

    #start_time = 1280.0
    #end_time = 1300.0
    start_time = None
    end_time = None
    hemis = ['R']

    block_name = 'Site3'
    segment_name = 'Call3'

    segment = exp.get_segment(block_name, segment_name)
    seg_uname = segment_to_unique_name(segment)

    if start_time is not None and end_time is not None:
        ofname = 'MEMD_%s_%s_%0.3f_%0.3f.h5' % (seg_uname, ','.join(hemis), start_time, end_time)
    else:
        ofname = 'MEMD_%s_%s.h5' % (seg_uname, ','.join(hemis))
    ofile = os.path.join(output_dir, ofname)

    #mt = MEMDTransform()
    #mt.transform(exp, segment, rcg_names=hemis, start_time=start_time, end_time=end_time)
    #mt.save(ofile)
    
    mt = MEMDTransform.load(ofile)
    mt.plot_entire_segment(plot_length=10.0, output_dir='/auto/tdrive/mschachter/figures/memd', demodulate=False, phase_only=False)
    #mt.plot_complex(output_dir=None, start_time=start_time, end_time=end_time, include_spec=False, spikes=True, demodulate=False, sort_code='single', log_amplitude=True)
    #plt.show()
