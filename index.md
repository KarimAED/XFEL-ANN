# XFEL-ANN
The Code and data for my research project on XFEL-Data prediction with ANNs. It uses data from https://www.github.com/alvarosg/DataLCLS2017 and was developed to run on the Imperial College London (ICL) HPC.

## Requirements

In order to run this code, the following packages are needed

* numpy
* scipy
* pandas
* scikit-learn
* tensorflow

Most of these can be installed from the pyproject.toml with poetry. Only tensorflow cannot, as the execution on the ICL HPC used tensorflow-gpu, which was not usable for my
development environment.

## Setup

In order to run the code, you can follow either of two methods:

* clone from branch 'main' and run setup.py
* clone from branch 'Data_incl' with the datasets included

If you intend to run the code on the ICL HPC, cloning from 'Data_incl' is the recommended method, as it includes submission scripts for said HPC.

## Usage

This repository provides a general purpose sklearn estimator for simple dense feed-forward neural networks (FFNNs). It furthermore provides the preprocessing pipelines described in the attached report.

There are 2 cases for which code is currently provided:

* Single Pulse Mean Energy Prediction
* Double Pulse Delay Prediction

the file labelled model_fit.py contains the code to fit a single FFNN with command-line parameters on these datasets.
For feature ranking, run feature_selection.py with the relevant parameters instead.

To execute them on the ICL HPC, use the submission scripts from the 'Data_incl' branch in the Scripts folder.
To instead run them locally, simply execute them in a python environment with the given requirements (see above).

### Model Command line Parameters

* inp-folder: folder name within the XFEL-ANN/Data/ folder
* --shape: list of integers, number of nodes per layer of the Dense FFNN
* --drop_out: The rate of drop_out to be applied between any two network layers (default: 20, 20, 10, 10, 10)
* --verbose: The verbosity level at which to train the network (default: 2)
* --activation: The activation function to apply to the Dense layers (default: relu)
* --loss: The loss function to use for training (default: mae)
* --rate: The learning rate of the (Adagrad) optimiser (default: 0.001)
* --regularizer: The kernel regularizer for the Dense layers (default: None)
* --epochs: The number of epochs to train the model for (default: 10,000)
* --batch_size: The size of each batch for training (default: 1,000)
* --stopper: If early stopping should be applied (default: False)
* --patience: If stopper is True, what patience for the stopper (default: 10)
* --p_delta: Delta for the stopper to apply (default: 1e-6)
* --validation_split: Fraction of training events to use as validation instead (default: 0.1)


## Troubleshooting

In case of a "file not found" error, make sure that the directory into which the project is cloned is called "XFEL-ANN". For any further questions, do not hesitate to raise an issue on this repository or contact me under karimaed@gmx.de
