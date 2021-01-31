import os
import sys
import argparse

import numpy as np
import pandas as pd

from estimator import ANN


def get_args():
    parser = argparse.ArgumentParser(description="Select datasets and model parameters")
    parser.add_argument("inp-folder")
    parser.add_argument("--shape", nargs="+", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--drop_out", "-d", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--verbose", "-v", action="count", default=argparse.SUPPRESS)
    parser.add_argument("--activation", "-a", default=argparse.SUPPRESS)
    parser.add_argument("--loss", "-l", default=argparse.SUPPRESS)
    parser.add_argument("--epochs", "-e", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch_norm", type=bool, default=argparse.SUPPRESS)
    args = parser.parse_args()
    return args.__dict__


def load_inp_folder(path):
    data = np.load(os.path.join(path, "data.npz"))
    iref = pd.read_pickle(os.path.join(path, "inp_ref.pkl"))
    oref = pd.read_pickle(os.path.join(path, "out_ref.pkl"))
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"], iref, oref


def main():
    args = get_args()
    folder = args["inp-folder"]
    del args["inp-folder"]
    loc = os.path.dirname(sys.argv[0])
    data_loc = os.path.join(loc, "Data")
    x_tr, x_te, y_tr, y_te, i_ref, o_ref = load_inp_folder(os.path.join(data_loc, folder))

    print(i_ref)
    print(o_ref)

    ann = ANN(**args)
    ann.fit(x_tr, y_tr)
    print(ann.score(x_te, y_te))

if __name__ == "__main__":
    sys.exit(main())
