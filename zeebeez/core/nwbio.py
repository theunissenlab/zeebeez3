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

        io = NWBHDF5IO(self.filename)
        nwb= io.read()

        # return block

    def write_block(self, block):
        raise NotImplementedError('NWBIO is a read-only I/O class.')