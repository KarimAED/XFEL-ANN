import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from estimator import ann, Layer


def get_args():
    parser = argparse.ArgumentParser(description="Select datasets and model parameters")
    parser.add_argument("inp-folder")
    parser.add_argument("model", choices=["ann", "grad-boost"], default="ann")
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
    parser.add_argument("--validation_split", "-vs", type=float, default=0.1)
    args = parser.parse_args()
    return args.__dict__


def load_inp_folder(path):
    data = np.load(os.path.join(path, "data.npz"))
    iref = pd.read_pickle(os.path.join(path, "inp_ref.pkl"))
    oref = pd.read_pickle(os.path.join(path, "out_ref.pkl"))
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"], iref, oref


def get_layers(args):
    layer_list = []
    for i in args["shape"]:
        layer_list.append(Layer(Dense, units=i, activation=args["activation"], kernel_regularizer=args["regularizer"]))
        layer_list.append(Layer(Dropout, rate=args["drop_out"]))
        if args["batch_norm"]:
            layer_list.append(Layer(BatchNormalization))
    return layer_list


def plot_pvm(args, est, x_tr, x_te, y_tr, y_te, o_ref, label):

    baseline = LinearRegression()

    baseline.fit(x_tr, y_tr)

    pred_tr = est.predict(x_tr)*o_ref.loc["train_std",:].values + o_ref.loc["train_mean",:].values
    pred_te = est.predict(x_te)*o_ref.loc["test_std",:].values + o_ref.loc["test_mean",:].values
    d_y_tr = y_tr*o_ref.loc["train_std",:].values + o_ref.loc["train_mean",:].values
    d_y_te = y_te*o_ref.loc["test_std",:].values + o_ref.loc["test_mean",:].values

    baseline_tr = baseline.predict(x_tr)*o_ref.loc["train_std",:].values + o_ref.loc["train_mean",:].values
    baseline_te = baseline.predict(x_te)*o_ref.loc["test_std",:].values + o_ref.loc["test_mean",:].values

    plt.figure()
    x = np.arange(np.min(d_y_tr), np.max(d_y_tr), 0.01)
    plt.plot(x, x, "k--", label="x=y")
    plt.scatter(d_y_tr, baseline_tr, s=0.5, alpha=0.2, c="black",
                label=f"S.-G. et al.; MAE={round(np.mean(np.abs(d_y_te-baseline_te)), 3)}")
    plt.scatter(d_y_te, baseline_te, s=0.5, alpha=0.2, c="black")
    try:
        mae = est.evaluate(x_te, y_te)[1]*o_ref.loc["test_std",:].values[0]
    except:
        mae = np.mean(np.abs(d_y_te-pred_te))
    plt.scatter(d_y_tr, pred_tr, s=0.5, alpha=0.2, c="blue",
                label=f"{args['model']}; MAE={round(mae, 3)}")
    plt.scatter(d_y_te, pred_te, s=0.5, alpha=0.2, c="blue")
    plt.legend()
    plt.xlabel("Measured Mean Pulse Energy (eV)")
    plt.ylabel("Predicted Mean Pulse Energy (eV)")
    plt.savefig("pvm_"+str(args["model"])+"_"+str(args["inp-folder"])+"_"+str(label)+".pdf")


def plot_hist(args, hist, label):
    plt.figure()
    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig("loss_nn_"+str(args["inp-folder"])+"_"+str(label)+".pdf")

    plt.figure()
    plt.plot(hist.history["mae"], label="Training MAE")
    plt.plot(hist.history["val_mae"], label="Validation MAE")
    plt.legend()
    plt.savefig("mae_nn_"+str(args["inp-folder"])+"_"+str(label)+".pdf")


def get_inp(folder):
    loc = os.path.dirname(sys.argv[0])
    data_loc = os.path.join(loc, "Data")
    return load_inp_folder(os.path.join(data_loc, folder))


def main():
    args = get_args()
    x_tr, x_te, y_tr, y_te, i_ref, o_ref = get_inp(args["inp-folder"])

    print(i_ref)
    print(o_ref)

    plt.style.use("mystyle-2.mplstyle")

    if args["model"] == "ann":
        est = fit_ann(args, x_tr, y_tr)
        print(est.evaluate(x_te, y_te))
    elif args["model"] == "grad-boost":
        features = ("ebeamLTU450", "ebeamL3Energy", "ebeamLTU250", "ebeamEnergyBC2")
        filter = [i for i in range(len(i_ref.columns)) if i_ref.columns[i] in features]
        x_tr = x_tr[:, filter]
        x_te = x_te[:, filter]
        est = fit_grad_boost(x_tr, y_tr)
        print(np.mean(np.abs(est.predict(x_te)-y_te)))
    label = time.time()
    plot_pvm(args, est, x_tr, x_te, y_tr, y_te, o_ref, label)


def fit_grad_boost(x_tr, y_tr):
    reg = HistGradientBoostingRegressor()
    reg.fit(x_tr, y_tr)
    return reg

def fit_ann(args, x_tr, y_tr):

    layer_list = get_layers(args)

    if len(y_tr.shape) > 1:
        out_sh = y_tr.shape[1]
    else:
        out_sh = 1

    opt = tf.keras.optimizers.Adagrad(learning_rate=args["rate"])
    est = ann(layer_list, out_sh, args["loss"], opt)
    start = time.time()
    hist = est.fit(x_tr, y_tr, args["batch_size"],
                   epochs=args["epochs"], verbose=args["verbose"],
                   validation_split=args["validation_split"])
    dur = time.time() - start
    print("Finished Fitting after {}s, {}s/epoch.".format(dur, dur/args["epochs"]))

    label = time.time()
    plot_hist(args, hist, label)

    est.save("ann_"+str(args["inp-folder"]+"_"+str(label)))
    return est


if __name__ == "__main__":
    sys.exit(main())
