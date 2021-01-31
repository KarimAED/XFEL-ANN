import os

import numpy as np
import scipy.stats as sps
import pandas as pd

from prep_utils import train_test_norm


def prep_delay_data(proj_dir, split=0.15):
    # Load double pulse data
    double_inp = pd.read_table(os.path.join(proj_dir, "Data/double_inputs.tsv.gz"))
    double_out = pd.read_table(os.path.join(proj_dir, "Data/double_outputs.tsv.gz"))

    # select only delay columns
    delay_out = double_out.loc[:, ["Delays", "DelayMask"]].copy()

    # get prepared mask
    delay_mask = delay_out["DelayMask"].values
    delays_nan = delay_out["Delays"].isna().values
    delay_mask = delay_mask - delays_nan  # Also mask NaN values
    delay_argmask = np.argwhere(delay_mask).flatten()  # create arg_mask to apply to inps and outputs

    # apply masking of events
    delay_inp = double_inp.iloc[delay_argmask].copy()
    delay_out = delay_out.loc[delay_argmask, "Delays"]  # only select delay copy

    # Filter input features by variance
    feat_columns = [c for c in delay_inp if len(np.unique(delay_inp[c])) > 10]
    delay_inp = delay_inp[feat_columns]

    # Get mean absolute deviation of outputs
    mad_delays = abs((delay_out.values
                      - np.median(delay_out.values)) / sps.median_abs_deviation(delay_out.values))

    # create mad and beam energy mask from relevant arrays
    mad_mask = mad_delays < 4
    emask = (delay_inp["f_63_ENRC"].values > 0.005) & (delay_inp["f_64_ENRC"].values > 0.005)

    arg_mask = np.argwhere(mad_mask & emask).flatten()  # generate yet another arg_mask

    # Apply arg_mask
    delay_inp = delay_inp.iloc[arg_mask]
    delay_out = delay_out.iloc[arg_mask]

    # Reuse training_test split and normalisation across inputs
    return train_test_norm(delay_inp, delay_out, split)


def save_delays(proj_dir, split=0.15):
    x_tr, x_te, y_tr, y_te, x_ref, y_ref = prep_delay_data(proj_dir, split)
    path = os.path.join(proj_dir, "Data/Delays")
    if not os.path.exists(path):
        d = os.getcwd()
        os.chdir(os.path.join(proj_dir, "Data"))
        os.mkdir("Delays")
        os.chdir(d)
    np.savez_compressed(os.path.join(path, "data.npz"), x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    x_ref.to_pickle(os.path.join(path, "inp_ref.pkl"))
    y_ref.to_pickle(os.path.join(path, "out_ref.pkl"))


def load_delays(proj_dir, split=0.15):
    if not os.path.exists(os.path.join(proj_dir, "Data/Delays")):
        save_delays(proj_dir, split)

    data = np.load(os.path.join(proj_dir, "Data/Delays/data.npz"))

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    x_ref = pd.read_pickle(os.path.join(proj_dir, "Data/Delays/inp_ref.pkl"))
    y_ref = pd.read_pickle(os.path.join(proj_dir, "Data/Delays/out_ref.pkl"))

    return x_train, y_train, x_test, y_test, x_ref, y_ref


if __name__ == "__main__":
    # Run this to generate preprocessed delay files
    save_delays(os.getcwd())
