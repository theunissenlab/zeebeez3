
import os

from dateutil import parser as date_parser
import h5py

import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, SpikeEventSeries


def write_multiple_nwb_files(root_dir='/tmp'):

    rec_datetime = date_parser.parse('03/22/2018 15:35:22')

    block = 'Site1'

    recording_name = '{}_{}'.format('bird', block)

    nwb_file = os.path.join(root_dir, '{}.nwb'.format(recording_name))

    session_desc = "A single recording session"

    exp_desc = "An experiment."

    nf = NWBFile(recording_name, session_desc,
                 'bird',
                 rec_datetime,
                 experimenter='Experi Menter',
                 lab='The Lab',
                 institution='University of Shaz',
                 experiment_description=exp_desc,
                 session_id='bird')

    hemi = 'L'

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
                                              'All electrodes in array for hemisphere {} with LFP'.format(hemi))

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

    # add the spikes and their waveforms
    electrode_number = 0
    spike_times = [0.5, 1.5, 3.5]
    waveforms = np.random.randn(3, 18)
    sort_type = 'Multi-unit'

    full_unit_name = '{} on electrode {}'.format(sort_type, electrode_number)
    spikes = SpikeEventSeries(full_unit_name, lfp_series_name,
                              waveforms, spike_times, etable,
                              resolution=1e-12, conversion=1e6,
                              comments='',
                              description='',
                              )

    print('\tAdding spikes acquisition: {} ({}), waveforms.shape={}'.format(full_unit_name, sort_type,
                                                                            str(waveforms.shape)))

    # adding the LFP is fine
    nf.add_acquisition(lfp)

    ########################################################
    # adding even a single spike event series causes the error
    ########################################################
    nf.add_acquisition(spikes)

    print('Writing to {}'.format(nwb_file))
    with NWBHDF5IO(nwb_file, mode='w') as io:
        io.write(nf)

    del nf



if __name__ == '__main__':

    write_multiple_nwb_files('/tmp')

    _nwb_file = os.path.join('/tmp/bird_Site1.nwb')
    _io = NWBHDF5IO(_nwb_file, mode='r')
    _nwb = _io.read()




