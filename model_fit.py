import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from estimator import ann, Layer


def get_args():
    parser = argparse.ArgumentParser(description="Select datasets and model parameters")
    parser.add_argument("inp-folder")
    parser.add_argument("--shape", "-s", nargs="+", type=int, default=[20, 20, 10, 10, 10])
    parser.add_argument("--drop_out", "-d", type=float, default=0.0)
    parser.add_argument("--verbose", "-v", action="count", default=2)
    parser.add_argument("--activation", "-a", default="relu")
    parser.add_argument("--loss", "-l", default="mae")
    parser.add_argument("--rate", "-r", type=float, default=0.001)
    parser.add_argument("--epochs", "-e", type=int, default=10000)
    parser.add_argument("--batch_norm", type=bool, default=False)
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
    loc = os.path.dirname(sys.argv[0])
    data_loc = os.path.join(loc, "Data")
    x_tr, x_te, y_tr, y_te, i_ref, o_ref = load_inp_folder(os.path.join(data_loc, folder))

    print(i_ref)
    print(o_ref)

    layer_list = []
    for i in args["shape"]:
        layer_list.append(Layer(Dense, units=i, activation=args["activation"]))
        layer_list.append(Layer(Dropout, rate=args["drop_out"]))
        if args["batch_norm"]:
            layer_list.append(Layer(BatchNormalisation))

    if len(y_tr.shape) > 1:
        out_sh = y_tr.shape[1]
    else:
        out_sh = 1

    opt = tf.keras.optimizers.Adagrad(learning_rate=args["rate"])
    est = ann(layer_list, out_sh, args["loss"], opt)
    est.fit(x_tr, y_tr, epochs=args["epochs"], verbose=args["verbose"])
    print(est.evaluate(x_te, y_te))


if __name__ == "__main__":
    sys.exit(main())
