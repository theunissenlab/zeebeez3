from pynwb import NWBHDF5IO

if __name__ == '__main__':


    io = NWBHDF5IO('/auto/tdrive/mschachter/nwb/GreBlu9508M/GreBlu9508M_Site1_Call1.nwb')
    nwb = io.read()