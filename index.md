The Code and data for my research project on XFEL-Data prediction with ANNs ([first report](https://www.overleaf.com/read/nvqtmzvyhsgf)).  
It uses data from [here](https://www.github.com/alvarosg/DataLCLS2017) and was developed to run on the Imperial College London (ICL) HPC.

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

the files labelled:

delay_est.py
emean_est.py

contain the code to fit a single FFNN with given parameters on these datasets.

To execute them on the ICL HPC, use the submission scripts from the 'Data_incl' branch in the Scripts folder.
To instead run them locally, simply execute them in a python environment with the given requirements (see above).

## Troubleshooting

In case of a "file not found" error, make sure that the directory into which the project is cloned is called "XFEL-ANN". For any further questions, do not hesitate to raise an issue on this repository or contact me under karimaed@gmx.de
