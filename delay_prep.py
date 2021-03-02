import os
import sys

import numpy as np
import scipy.stats as sps
import pandas as pd

from prep_utils import train_test_norm


def prep_delay_data(proj_dir, split=0.15, new=False):
    # Load double pulse data
    if not new:
        double_inp = pd.read_table(os.path.join(proj_dir, "Data/double_inputs.tsv.gz"))
        double_out = pd.read_table(os.path.join(proj_dir, "Data/double_outputs.tsv.gz"))
    else:
        double_inp = pd.read_table(os.path.join(proj_dir, "Data/new_inputs.tsv.gz"))
        double_out = pd.read_table(os.path.join(proj_dir, "Data/new_outputs.tsv.gz"))

    print("Filtering output columns...")
    # select only delay columns
    delay_out = double_out.loc[:, ["Delays", "DelayMask"]].copy()

    print(delay_out.shape[1], "columns left.")

    print("Filtering events...")
    # get prepared mask
    delay_mask = delay_out["DelayMask"].values
    delays_nan = delay_out["Delays"].notna().values
    delay_mask = delay_mask.astype(np.bool) & delays_nan.astype(np.bool)  # Also mask NaN values
    delay_mask = delay_mask  # create arg_mask to apply to inps and outputs

    # apply masking of events
    delay_inp = double_inp.iloc[delay_mask].copy()
    delay_out = delay_out.loc[delay_mask, "Delays"]  # only select delay copy


    print(delay_inp.shape[0], "events left.")
    print("Filtering input columns...")
    # Filter input features by variance
    if new:
        var_thresh = 50
    else:
        var_thresh = 10
    feat_columns = [c for c in delay_inp if len(np.unique(delay_inp[c])) > var_thresh]
    delay_inp = delay_inp[feat_columns]

    print(delay_inp.shape[1], "columns left.")
    print("Filtering MAD & Energy...")
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
    print(delay_inp.shape[0], "events left.")
    print("Done.")
    # Reuse training_test split and normalisation across inputs
    return train_test_norm(delay_inp, delay_out, split)


def save_delays(proj_dir, split=0.15, new=False):
    x_tr, x_te, y_tr, y_te, x_ref, y_ref = prep_delay_data(proj_dir, split, new)
    if not new:
        path = os.path.join(proj_dir, "Data/Delays")
    else:
        path = os.path.join(proj_dir, "Data/NewDelays")
    if not os.path.exists(path):
        d = os.getcwd()
        os.chdir(os.path.join(proj_dir, "Data"))
        if not new:
            os.mkdir("Delays")
        else:
            os.mkdir("NewDelays")
        os.chdir(d)
    np.savez_compressed(os.path.join(path, "data.npz"), x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    x_ref.to_pickle(os.path.join(path, "inp_ref.pkl"))
    y_ref.to_pickle(os.path.join(path, "out_ref.pkl"))


def load_delays(proj_dir, split=0.15, new=False):
    if not new:
        d = "Data/Delays"
    else:
        d = "Data/NewDelays"
    if not os.path.exists(os.path.join(proj_dir, d)):
        save_delays(proj_dir, split)

    data = np.load(os.path.join(proj_dir, d+"/data.npz"))

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    x_ref = pd.read_pickle(os.path.join(proj_dir, d+"/inp_ref.pkl"))
    y_ref = pd.read_pickle(os.path.join(proj_dir, d+"/out_ref.pkl"))

    return x_train, y_train, x_test, y_test, x_ref, y_ref


if __name__ == "__main__":
    # Run this to generate preprocessed delay files
    if len(sys.argv) > 1:
        new = bool(sys.argv[1])
    else:
        new=False
    save_delays(os.getcwd(), new=new)
