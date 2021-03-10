import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from estimator import ann, Layer


def get_args():
    parser = argparse.ArgumentParser(description="Select datasets and model parameters")
    parser.add_argument("inp-folder")
    parser.add_argument("--shape", "-s", nargs="+", type=int, default=[20, 20, 10, 10, 10])
    parser.add_argument("--drop_out", "-d", type=float, default=0.0)
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--activation", "-a", default="relu")
    parser.add_argument("--loss", "-l", default="mae")
    parser.add_argument("--rate", "-r", type=float, default=0.001)
    parser.add_argument("--regularizer", "-reg", choices=["l1", "l2"], default=None)
    parser.add_argument("--epochs", "-e", type=int, default=10000)
    parser.add_argument("--batch_size", "-bs", type=int, default=1000)
    parser.add_argument("--batch_norm", type=bool, default=False)
    parser.add_argument("--stopper", type=bool, default=False)
    parser.add_argument("--patience", "-p", type=int, default=10)
    parser.add_argument("--p_delta", "-pd", type=float, default=1e-6)
    parser.add_argument("--validation_split", "-vs", type=float, default=0.1)
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
        layer_list.append(Layer(Dense, units=i, activation=args["activation"], kernel_regularizer=args["regularizer"]))
        layer_list.append(Layer(Dropout, rate=args["drop_out"]))
        if args["batch_norm"]:
            layer_list.append(Layer(BatchNormalization))

    if len(y_tr.shape) > 1:
        out_sh = y_tr.shape[1]
    else:
        out_sh = 1

    opt = tf.keras.optimizers.Adagrad(learning_rate=args["rate"])
    start = time.time()
    excluded_features = []
    scores = []
    for i in range(len(i_ref.columns)-60):
        est = ann(layer_list, out_sh, args["loss"], opt)
        mask = [not j==i_ref.columns[i] for j in i_ref.columns]
        x_tr_masked = x_tr[:, mask]
        x_te_masked = x_te[:, mask]
        hist = est.fit(x_tr_masked, y_tr, args["batch_size"],
                       epochs=args["epochs"], verbose=args["verbose"],
                       validation_split=args["validation_split"])
        excluded_features.append(i_ref.columns[i])
        scores.append(est.evaluate(x_te_masked, y_te)[1])
    dur = time.time() - start
    print(f"Runtime: {dur}")
    feature_rank = pd.DataFrame({"features": excluded_features, "mae_score": scores})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)
    print(feature_rank.head())
    plt.figure(figsize=(20, 7))
    plt.bar(feature_rank["features"], feature_rank["mae_score"])
    plt.xticks(rotation=90)
    label = time.time()
    plt.ylabel("MAE")
    plt.xlabel("Excluded Feature")
    plt.savefig("feat_"+args["inp-folder"]+"_"+str(label)+".pdf", bbox_inches="tight")

if __name__ == "__main__":
    sys.exit(main())
