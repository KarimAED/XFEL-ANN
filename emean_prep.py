import os

import numpy as np
import scipy.stats as sps
import pandas as pd

from prep_utils import train_test_norm

def get_energy(pixel):
    m_pix = 466.
    m_ener = 531.5
    a = 9.8
    return (pixel-m_pix)/a + m_ener


def prep_delay_data(proj_dir, split=0.15):
    # Load single pulse data
    single_inp = pd.read_table(os.path.join(proj_dir, "Data/single_inputs.tsv.gz"))
    single_out = pd.read_table(os.path.join(proj_dir, "Data/single_outputs.tsv.gz"))

    # select only fit columns
    emean_out = single_out.loc[:, ["GaussMean_pxl", "FitMask"]].copy()

    # get prepared mask
    emean_mask = emean_out["FitMask"].values
    fit_nan = emean_out["GaussMean_pxl"].isna().values
    emean_mask = emean_mask - fit_nan  # Also mask NaN values
    emean_argmask = np.argwhere(emean_mask).flatten()  # create arg_mask to apply to inps and outputs

    # apply masking of events
    emean_inp = single_inp.iloc[emean_argmask].copy()
    emean_out = emean_out.loc[emean_argmask, "GaussMean_pxl"]  # only select fit copy
    emean_out = emean_out.apply(lambda x: get_energy(x))
    emean_out.name = "GaussMean_eV"

    print(emean_out)

    # Filter input features by variance
    feat_columns = [c for c in emean_inp if len(np.unique(emean_inp[c])) > 10]
    emean_inp = emean_inp[feat_columns]

    # Get mean absolute deviation of outputs
    mad_emean = abs((emean_out.values
                      - np.median(emean_out.values)) / sps.median_abs_deviation(emean_out.values))

    # create mad and beam energy mask from relevant arrays
    mad_mask = mad_emean < 4
    emask = (emean_inp["f_63_ENRC"].values > 0.005) & (emean_inp["f_64_ENRC"].values > 0.005)

    arg_mask = np.argwhere(mad_mask & emask).flatten()  # generate yet another arg_mask

    # Apply arg_mask
    emean_inp = emean_inp.iloc[arg_mask]
    emean_out = emean_out.iloc[arg_mask]

    # Reuse training_test split and normalisation across inputs
    return train_test_norm(emean_inp, emean_out, split)


def save_emean(proj_dir, split=0.15):
    x_tr, x_te, y_tr, y_te, x_ref, y_ref = prep_delay_data(proj_dir, split)
    path = os.path.join(proj_dir, "Data/EMean")
    if not os.path.exists(path):
        d = os.getcwd()
        os.chdir(os.path.join(proj_dir, "Data"))
        os.mkdir("EMean")
        os.chdir(d)
    np.savez_compressed(os.path.join(path, "data.npz"), x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    x_ref.to_pickle(os.path.join(path, "inp_ref.pkl"))
    y_ref.to_pickle(os.path.join(path, "out_ref.pkl"))


def load_emean(proj_dir, split=0.15):
    if not os.path.exists(os.path.join(proj_dir, "Data/EMean")):
        save_delays(proj_dir, split)

    data = np.load(os.path.join(proj_dir, "Data/EMean/data.npz"))

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    x_ref = pd.read_pickle(os.path.join(proj_dir, "Data/EMean/inp_ref.pkl"))
    y_ref = pd.read_pickle(os.path.join(proj_dir, "Data/EMean/out_ref.pkl"))

    return x_train, y_train, x_test, y_test, x_ref, y_ref


if __name__ == "__main__":
    # Run this to generate preprocessed delay files
    save_emean(os.getcwd())
