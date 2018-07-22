"""
    This script converts stims.h5 files that were created by neosound into a stimulus attribute .csv file and
    an a bunch of .wav files.
"""

import os
import h5py
import wave
import argparse

import numpy as np
import pandas as pd


def convert(neosound_file, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cols = ['number', 'callid', 'stim_class', 'source', 'source_sex', 'samplerate', 'original_filename']
    sound_data = {c:list() for c in cols}
    sound_data['filename'] = list()

    bytewidth = 4
    max_amp = 2**(8*bytewidth) - 1

    hf = h5py.File(neosound_file, 'r')
    for stim_name,stim_grp in hf.items():
        if 'waveform' not in stim_grp:
            continue

        print(stim_name)
        for k,v in stim_grp.attrs.items():
            print('\t{}={}'.format(k, v))

        for key in cols:
            val = 'NA'
            if key in stim_grp.attrs:
                val = stim_grp.attrs[key]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            sound_data[key].append(val)

        stim_id = sound_data['number'][-1]
        sr = sound_data['samplerate'][-1]

        fname = 'stim{}.wav'.format(stim_id)
        sound_data['filename'].append(fname)

        out_fname = os.path.join(output_path, fname)

        waveform = np.array(stim_grp['waveform'])

        assert max(np.abs(waveform)) <= 1, "waveform.absmax={}".format(max(np.abs(waveform)))
        waveform_conv = (waveform * (max_amp / 2)).astype('int32')

        with wave.open(out_fname, 'wb') as wf:
            wf.setnchannels(1)
            wf.setframerate(sr)
            wf.setsampwidth(bytewidth)
            wf.writeframes(waveform_conv)

        print('Wrote {}'.format(out_fname))

    hf.close()

    df = pd.DataFrame(sound_data)
    df = df.sort_values(by='number')
    cols.insert(-1, 'filename')
    df.to_csv(os.path.join(output_path, 'stims.csv'), header=True, columns=cols, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--neosound_file", help="The name of a neosound file", required=True)
    parser.add_argument("--output_path", help="The directory to output the .csv file and the .wav files.", required=True)
    args = parser.parse_args()

    convert(args.neosound_file, args.output_path)


