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
    parser.add_argument("--verbose", "-v", action="count", default=2)
    parser.add_argument("--activation", "-a", default="relu")
    parser.add_argument("--loss", "-l", default="mae")
    parser.add_argument("--rate", "-r", type=float, default=0.001)
    parser.add_argument("--epochs", "-e", type=int, default=10000)
    parser.add_argument("--batch_size", "-bs", type=int, default=1000)
    parser.add_argument("--batch_norm", type=bool, default=False)
    parser.add_argument("--patience", "-p", type=int, default=10)
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
        layer_list.append(Layer(Dense, units=i, activation=args["activation"]))
        layer_list.append(Layer(Dropout, rate=args["drop_out"]))
        if args["batch_norm"]:
            layer_list.append(Layer(BatchNormalisation))

    if len(y_tr.shape) > 1:
        out_sh = y_tr.shape[1]
    else:
        out_sh = 1

    stopper = tf.keras.callbacks.EarlyStopping(monitor="mae", patience=args["patience"], min_delta=1e-6)
    opt = tf.keras.optimizers.Adagrad(learning_rate=args["rate"])
    est = ann(layer_list, out_sh, args["loss"], opt)
    start = time.time()
    hist = est.fit(x_tr, y_tr, args["batch_size"],
                   epochs=args["epochs"], verbose=args["verbose"],
                   validation_split=args["validation_split"],
                   callbacks=stopper)
    dur = time.time() - start
    print("Finished Fitting after {}s, {}s/epoch.".format(dur, dur/args["epochs"]))
    print(est.evaluate(x_te, y_te))

    label = time.time()

    pred_tr = est.predict(x_tr)*o_ref.loc["train_std",:].values + o_ref.loc["train_mean",:].values
    pred_te = est.predict(x_te)*o_ref.loc["test_std",:].values + o_ref.loc["test_mean",:].values
    d_y_tr = y_tr*o_ref.loc["train_std",:].values + o_ref.loc["train_mean",:].values
    d_y_te = y_te*o_ref.loc["test_std",:].values + o_ref.loc["test_mean",:].values

    plt.figure()
    x = np.arange(np.min(d_y_tr), np.max(d_y_tr), 0.01)
    plt.grid()
    plt.plot(x, x, "k--", label="x=y")
    plt.scatter(d_y_tr, pred_tr, s=1, alpha=0.5, c="blue", label="Training set")
    plt.scatter(d_y_te, pred_te, s=1, alpha=0.5, c="red", label="Test set")
    plt.legend()
    plt.savefig("pvm_"+str(args["inp-folder"])+"_"+str(label)+".pdf")

    plt.figure()
    plt.grid()
    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig("loss_"+str(args["inp-folder"])+"_"+str(label)+".pdf")

    plt.figure()
    plt.grid()
    plt.plot(hist.history["mae"], label="Training MAE")
    plt.plot(hist.history["val_mae"], label="Validation MAE")
    plt.legend()
    plt.savefig("mae_"+str(args["inp-folder"])+"_"+str(label)+".pdf")

    est.save("model_"+str(args["inp-folder"]+"_"+str(label)))

if __name__ == "__main__":
    sys.exit(main())
