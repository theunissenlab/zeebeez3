import os

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal, SpikeTrain
from pynwb import NWBHDF5IO


class NWBIO(BaseIO):
    """
        A class for reading data from an .nwb file created with pynwb. Writing is not implemented.
    """

    is_readable = True
    is_writable = False
    has_header = False
    is_streameable = False

    supported_objects = [Block, Segment, AnalogSignal, SpikeTrain]
    readable_objects = supported_objects
    writeable_objects = list()

    mode = 'file'
    name = ".nwb electrophysiology file"
    extensions = ['nwb']

    def read_block(self, lazy=False):

        assert os.path.exists(self.filename), "Cannot locate file: {}".format(self.filename)

        with NWBHDF5IO(self.filename, mode='r') as io:
            nwb = io.read()

        blk = Block()


    def write_block(self, block, **kargs):
        raise NotImplementedError('NWBIO is a read-only I/O class.')


if __name__ == '__main__':

    _root_dir = '/auto/tdrive/mschachter/nwb'
    _bird_name = 'GreBlu9508M'
    _block_site = 'Site1_Call1'

    _nwb_file = os.path.join(_root_dir, _bird_name, '{}_{}.nwb'.format(_bird_name, _block_site))

    _io = NWBIO(_nwb_file)
    _blk = _io.read_block()
