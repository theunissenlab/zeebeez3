
import os

from dateutil import parser as date_parser
import h5py

import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, SpikeEventSeries


def write_multiple_nwb_files(root_dir='/tmp'):

    rec_datetime = date_parser.parse('03/22/2018 15:35:22')

    # create an electrode array, electrode groups, and electrode table for the electrodes
    lfp_series = dict()
    spike_series = dict()

    for block in ['Site1', 'Site2']:

        print('*************** Processing block {}'.format(block))

        recording_name = '{}_{}'.format('bird', block)

        nwb_file = os.path.join(root_dir, '{}.nwb'.format(recording_name))

        session_desc = "A single recording session"

        exp_desc = """ A randomly interleaved mixture of Zebra finch vocalizations, songs, and modulation limited
                       noise played to a Zebra finch under urethane anaesthesia in an acute experiment. Two 16 microwire
                       electrode arrays were implanted, one in each hemisphere. Experiments  designed and performed by
                       Julie Elie, vocal repertoire recorded by Julie Elie.
                   """

        nf = NWBFile(recording_name, session_desc,
                     'bird',
                     rec_datetime,
                     experimenter='Julie Elie',
                     lab='Theunissen Lab',
                     institution='UC Berkeley',
                     experiment_description=exp_desc,
                     session_id='bird')

        # create the electrodes and electrode tables
        for hemi in ['R', 'L']:

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
            electrode_start = 0
            if hemi == 'R':
                electrode_start = 16
            electrode_numbers = list(np.arange(electrode_start, electrode_start + 16))
            for electrode_number in electrode_numbers:

                nf.add_electrode(electrode_number-1,
                                 x=np.random.randn(), y=np.random.randn(), z=0.0,
                                 imp=0.0,
                                 location='cortex',
                                 filtering='none',
                                 description='An electrode',
                                 group=electrode_group)

            # create an electrode table region
            etable = nf.create_electrode_table_region(electrode_numbers,
                                                      'All electrodes in array for hemisphere {}'.format(hemi))

            lfp_data = np.random.randn(len(electrode_numbers), 1000)
            sr = 1000.
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
            for electrode_number in electrode_numbers:

                for k in range(3):
                    if k == 0:
                        unit_name = 'Chan{} Code0'.format(electrode_number)
                    else:
                        unit_name = 'RF Sort unit {}'.format(k+1)

                    spike_times = [0.5, 1.5, 3.5]
                    waveforms = np.random.randn(3, 18)

                    print('{} electrode_number={}, e_num={}, k={}'.format(unit_name, electrode_number, e_num, k))

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

                    full_unit_name = '{} on electrode {}'.format(sort_type, electrode_number)
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
        """
        nf.add_trial_column('stim_id', 'The ID of the sound played during the trial.')
        nf.add_trial_column('trial', 'The number of times this stimulus has been presented.')
        for stim_epoch_data in block_data['epochs']:
            stim_id = stim_epoch_data['stim_id']
            trial_num = stim_epoch_data['trial']
            stime = stim_epoch_data['start']
            etime = stim_epoch_data['end']
            nf.add_trial({'start':stime, 'end':etime, 'stim_id':stim_id, 'trial':trial_num})
        """

        print('Writing to {}'.format(nwb_file))
        with NWBHDF5IO(nwb_file, mode='w') as io:
            io.write(nf)

        del nf



if __name__ == '__main__':

    write_multiple_nwb_files()
