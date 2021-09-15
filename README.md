## Introduction

zeebeez3 is a refactor of the [zeebeez](https://github.com/theunissenlab/zeebeez) project. The code
for zeebeez produced all the analysis and figures for the [Schachter 2016 thesis](http://pqdtopen.proquest.com/pubnum/10192590.html).

In this version, everything is python3 compatible, and utilizes the [soundsig](https://github.com/theunissenlab/soundsig)
package. All dependencies to [neosound](https://github.com/theunissenlab/neosound) have
been removed.

The base file format for zeebeez3 are nwb files, created with [pynwb](https://github.com/NeurodataWithoutBorders/pynwb).
There are a series of transformations applied to the nwb files that produce hdf5 files with highly preprocessed data. There
is one nwb file per recording site.

Event though the original codebase was developed for experiments in anaesthetized Zebra finch with
repeated trials of randomly interleaved stimuli, the representation of the data in the Experiment object
is relatively neutral to the type of recording. The data is represented in it's continuous format.


## Local Installation

The first step is to clone the repository:

    git clone https://github.com/theunissenlab/zeebeez3
    
This project requires that you have [anaconda](https://www.anaconda.com/download/#linux) installed. The
next step is to create a virtual environment with the dependencies
that zeebeez3 needs:

    cd zeebeez3
    conda env create -f environment.yml -n zeebeez3

Now activate that environment:

    source activate zeebeez3
    
and install zeebeez3

    python setup.py install
    
Note that at the moment, you will have to have the latest version of soundsig installed.
To do this, pull the latest version or clone [soundsig](https://github.com/theunissenlab/soundsig),
and then from the soundsig directory:

    source activate zeebeez3
    python setup.py install

## Running Notebooks

To run the notebooks, first start jupyter notebooks from the zeebeez3 notebooks directory:

    cd notebooks
    jupyter notebook
    
A web browser tab will pop up. Select the notebook that you're interested in.
