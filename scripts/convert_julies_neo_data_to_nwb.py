import os
import re
import json
import operator

from dateutil import parser as date_parser
import h5py

import numpy as np
import pandas as pd

from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, SpikeEventSeries

bird_info = {
    'WhiWhi4522M': {
        'recording_date': '2012/03/05',
        'recording_start': '13:40',
        'recording_end': '22:50',
        'anaesthesia_start': '9:35',
        'fasting_start': '8:55'
    },
    'GreBlu9508M': {
        'recording_date': '2014/03/14',
        'recording_start': '13:40',
        'recording_end': '10:30',
        'anaesthesia_start': '10:00',
        'fasting_start': '9:15'
    },
    'YelBlu6903F': {
        'recording_date': '2012/03/19',
        'recording_start': '14:10',
        'recording_end': '21:35',
        'anaesthesia_start': '10:20',
        'fasting_start': '9:30'
    }
}


def clean_string(s):
    return str(s).replace("b'", '').replace("'", '')


def read_old_neo(hf, block_name):
    block_grp = None
    for block_grp2_name, block_grp2 in hf.items():
        blk_grp_attr_name = clean_string(block_grp2.attrs['name'])
        if blk_grp_attr_name == block_name:
            block_grp = block_grp2
            break

    sample_rate = None
    duration = None
    segment_name = clean_string(block_grp['segments']['Segment_0'].attrs['name'])

    print('-----------------------------------')
    print('Processing block {}, segment {}'.format(block_name, segment_name))

    recchan_grp = block_grp['recordingchannelgroups']

    neo_data = dict()
    neo_data['electrodes'] = dict()

    for rgrp_name, rgrp in recchan_grp.items():
        index2electrode = list(np.array(rgrp['channel_indexes']))
        print('\tProcessing recording channel group {}: {}'.format(rgrp_name, index2electrode))

        # get LFP data for each electrode
        for chan_grp_name_orig, chan_grp in rgrp['recordingchannels'].items():

            chan_grp_name = clean_string(chan_grp.attrs['name'])
            e_match = re.findall(' [0-9]*', chan_grp_name)
            assert len(e_match) == 1, "weird chan_grp_name={}".format(chan_grp_name)
            electrode = int(e_match[0].strip())
            print('\tProcessing electrode {}'.format(electrode))

            asig_grp = chan_grp['analogsignals']['AnalogSignal_0']

            lfp = np.array(asig_grp['signal'])

            if sample_rate is None:
                sample_rate = float(asig_grp['sampling_rate'][()])

            if duration is None:
                duration = len(lfp) / sample_rate

            neo_data['electrodes'][electrode] = dict()
            neo_data['electrodes'][electrode]['lfp'] = lfp
            neo_data['electrodes'][electrode]['sample_rate'] = sample_rate
            neo_data['electrodes'][electrode]['units'] = list()

        # get unit data
        for ugrp_name, ugrp in rgrp['units'].items():
            unit_name = clean_string(ugrp.attrs['name'])
            print('\tProcessing unit {}'.format(unit_name))
            spike_times = np.array(ugrp['spiketrains']['SpikeTrain_0']['times'])
            waveforms = np.array(ugrp['spiketrains']['SpikeTrain_0']['waveforms'])

            chan_match = re.findall('Chan[\s0-9]*', unit_name)
            assert len(chan_match) > 0, "No channel found in unit name: {}".format(unit_name)

            electrode = int(chan_match[0].replace('Chan', '').replace(' ', ''))

            print('\t{} mapped to electrode {}'.format(unit_name, electrode))

            neo_data['electrodes'][electrode]['units'].append({'electrode': electrode,
                                                               'spike_times': spike_times,
                                                               'waveforms': waveforms.squeeze(),
                                                               'name': unit_name})

    # read epochs
    epochs = list()
    trials = dict()
    for epoch_grp_name, epoch_grp in block_grp['segments']['Segment_0']['epochs'].items():
        stime = float(epoch_grp['time'][()])
        etime = stime + float(epoch_grp['duration'][()])
        stim_str = clean_string(epoch_grp.attrs['label'])
        stim_match = re.findall('Stim[0-9]*', stim_str)
        assert len(stim_match) == 1, "Can't get stim id from string: {}".format(stim_str)
        stim_id = int(stim_match[0].replace('Stim', ''))
        if stim_id not in trials:
            trials[stim_id] = 0
        trials[stim_id] += 1

        epochs.append({'start': stime, 'end': etime, 'stim_id': stim_id, 'trial': trials[stim_id]})

    epochs.sort(key=operator.itemgetter('start'))

    neo_data['sample_rate'] = sample_rate
    neo_data['protocol'] = segment_name
    neo_data['duration'] = duration
    neo_data['epochs'] = epochs

    return neo_data


def convert_from_old_neo(old_file, bird_name, electrode_df):
    root_dir, fname = os.path.split(old_file)

    rec_date = bird_info[bird_name]['recording_date']
    rec_start = bird_info[bird_name]['recording_start']
    rec_end = bird_info[bird_name]['recording_end']

    rec_datetime = date_parser.parse('{} {}'.format(rec_date, rec_start))

    # create an electrode array, electrode groups, and electrode table for the electrodes
    i = electrode_df.bird == bird_name
    edf = electrode_df[i]

    for block, gdf in edf.groupby(['block']):

        lfp_series = dict()
        spike_series = dict()

        print('*************** Processing block {}'.format(block))

        if bird_name == 'WhiWhi4522M' and block == 'Site1':
            continue

        # get the LFP and spike data for the block
        hf = h5py.File(old_file, 'r')
        block_data = read_old_neo(hf, block)
        hf.close()

        recording_name = '{}_{}_{}'.format(bird_name, block, block_data['protocol'])

        nwb_file = os.path.join(root_dir, '{}.nwb'.format(recording_name))

        session_desc = """ A single recording session, roughly one hour long, at a single depth, in a series of
                           recordings from the auditory "cortex" of zebra finch {}. Block {}, stimulus protocol
                           {}.        
                       """.format(bird_name, block, block_data['protocol'])

        exp_desc = """ A randomly interleaved mixture of Zebra finch vocalizations, songs, and modulation limited
                       noise played to a Zebra finch under urethane anaesthesia in an acute experiment. Two 16 microwire
                       electrode arrays were implanted, one in each hemisphere. Experiments  designed and performed by
                       Julie Elie, vocal repertoire recorded by Julie Elie. Data converted to NWB by Mike Schachter. For
                       a full description of methods, please consult and also cite the following publications:

                       Elie, J. E., & Theunissen, F. E. (2015). Meaning in the avian auditory cortex: neural
                       representation of communication calls. European Journal of Neuroscience, 41(5), 546-567.

                       Elie, J. E., & Theunissen, F. E. (2016). The vocal repertoire of the domesticated zebra finch:
                       a data-driven approach to decipher the information-bearing acoustic features of communication
                       signals. Animal cognition, 19(2), 285-315. 
                   """

        nf = NWBFile(recording_name, session_desc,
                     bird_name,
                     rec_datetime,
                     experimenter='Julie Elie',
                     lab='Theunissen Lab',
                     institution='UC Berkeley',
                     experiment_description=exp_desc,
                     session_id=bird_name)

        # create the electrodes and electrode tables
        for hemi, ggdf in gdf.groupby(['hemisphere']):

            electrode_array_name = '16 electrode microwire array on {} hemisphere'.format(hemi)
            electrode_array = nf.create_device(name=electrode_array_name, source='')

            # create an electrode group
            egrp_desc = """ The (x,y) locations of the electrodes refer to the distance from region midline and distance
                            from L2A, respectively, in mm.            
                        """

            electrode_group = nf.create_electrode_group(hemi,
                                                        source=electrode_array_name,
                                                        description=egrp_desc,
                                                        location='{} Hemisphere, Field L, CM, NCM'.format(hemi),
                                                        device=electrode_array)

            # add electrodes to electrode group
            for row_idx, row in ggdf.iterrows():
                electrode_number = row['electrode'] - 1
                dist_l2a = row['dist_l2a']
                dist_midline = row['dist_midline']
                region = row['region']

                if bird_name == 'GreBlu9508M':
                    dist_l2a *= 4  # correct for the error in the original data

                nf.add_electrode(electrode_number,
                                 x=dist_midline, y=dist_l2a, z=0.0,
                                 imp=0.0,
                                 location=region,
                                 filtering='none',
                                 description='Row {}, Column {}'.format(row['row'], row['col']),
                                 group=electrode_group)

            # create an electrode table region
            electrode_numbers = list(np.array(sorted(gdf.electrode.unique())) - 1)
            etable = nf.create_electrode_table_region(electrode_numbers,
                                                      'All electrodes in array for hemisphere {}'.format(hemi))

            lfp_data = np.array([block_data['electrodes'][e + 1]['lfp'] for e in electrode_numbers])
            sr = block_data['sample_rate']
            t = np.arange(lfp_data.shape[1]) / sr

            # add the raw LFP
            lfp_series_name = 'Multi-electrode LFP on {} hemisphere from block {}'.format(hemi, block)
            lfp = ElectricalSeries(
                lfp_series_name,
                electrode_array_name,
                lfp_data,
                etable,
                timestamps=t,
                resolution=1e-12,
                comments='',
                description='Low-passed LFP recorded from microwire array'
            )

            lfp_series[hemi] = lfp

            # add the spikes and their waveforms
            for row_idx, row in ggdf.iterrows():

                # electrode_number ranges from 1-32, same as the keys in block_data['electrodes']
                electrode_number = row['electrode']

                for k, unit_data in enumerate(block_data['electrodes'][electrode_number]['units']):
                    spike_times = unit_data['spike_times']
                    waveforms = unit_data['waveforms']
                    unit_name = unit_data['name']
                    e_num = unit_data['electrode']

                    print('{} electrode_number={}({}), e_num={}, k={}'.format(unit_name, electrode_number, electrode_number-1, e_num, k))

                    assert e_num == electrode_number

                    if unit_name.startswith('RF Sort'):
                        xarr = unit_name.split(' ')
                        unit_num = xarr[-1]
                        sort_type = 'Spike-sorted unit {}'.format(unit_num)

                    elif unit_name.startswith('Chan'):
                        if 'Code0' not in unit_name:
                            print('Skipping multi-unit channel with non-zero sort code')
                            continue
                        sort_type = 'Multi-unit'
                    else:
                        raise Exception('Unknown unit name: {}'.format(unit_name))

                    full_unit_name = '{} on block {} and electrode {}'.format(sort_type, block, e_num-1)
                    spikes = SpikeEventSeries(full_unit_name, lfp_series_name,
                                              waveforms, spike_times, etable,
                                              resolution=1e-12, conversion=1e6,
                                              comments='',
                                              description='',
                                              )

                    print('\tAdding spikes acquisition: {} ({}), waveforms.shape={}'.format(full_unit_name, sort_type,
                                                                                            str(waveforms.shape)))

                    spike_series[(hemi, electrode_number, unit_name)] = spikes

        all_series = list()
        all_series.extend(lfp_series.values())
        all_series.extend(spike_series.values())
        print('len(all_series)=',len(all_series))
        nf.add_acquisition(all_series)

        # create trials for each stim presentation
        nf.add_trial_column('stim_id', 'The ID of the sound played during the trial.')
        nf.add_trial_column('trial', 'The number of times this stimulus has been presented.')
        for stim_epoch_data in block_data['epochs']:
            stim_id = stim_epoch_data['stim_id']
            trial_num = stim_epoch_data['trial']
            stime = stim_epoch_data['start']
            etime = stim_epoch_data['end']
            nf.add_trial({'start':stime, 'end':etime, 'stim_id':stim_id, 'trial':trial_num})

        print('Writing to {}'.format(nwb_file))
        with NWBHDF5IO(nwb_file, mode='w') as io:
            io.write(nf)

        del nf


if __name__ == '__main__':

    _root_dir = '/auto/tdrive/mschachter/nwb'
    # for _bird_name in ['GreBlu9508M', 'YelBlu6903F', 'WhiWhi4522M']:
    for _bird_name in ['GreBlu9508M']:
        _bird_dir = os.path.join(_root_dir, _bird_name)

        _old_file = os.path.join(_bird_dir, '{}.h5'.format(_bird_name))

        _edata_file = os.path.join(_root_dir, 'electrode_data+dist.csv')
        _edata = pd.read_csv(_edata_file)

        convert_from_old_neo(_old_file, _bird_name, _edata)
