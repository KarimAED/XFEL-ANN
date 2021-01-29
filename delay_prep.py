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


if __name__ == "__main__":
    # for debugging purposes only
    x_tr, x_te, y_tr, y_te, x_ref, y_ref = prep_delay_data()
    print(x_tr.shape, x_te.shape, y_tr.shape, y_te.shape)
