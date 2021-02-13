import os
import sys
import argparse
import datetime

import numpy as np
import pandas as pd
import kerastuner as kt
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

from estimator import Layer, ann


def hp_estimator(hp):
    layers = []
    act = hp.Choice("activation", ["relu", "tanh"])
    reg = hp.Choice("regularizer", ["l2", "l1"])
    drop_out = hp.Float("drop_out", 0.0, 0.05, step=0.025)
    norm = hp.Boolean("norm")
    init_units = hp.Int("units_1", 10, 50, step=10)
    later_units = hp.Int("units_2", 10, 50, step=10)
    for i in range(hp.Int("layers", 6, 10, default=6)):
        if i < 3:
            units = init_units
        else:
            units = later_units
        dense = Layer(kind=Dense,
                      units=units,
                      activation=act,
                      kernel_regularizer=reg)
        layers.append(dense)
        layers.append(Layer(kind=Dropout, rate=drop_out))
        if norm:
            layers.append(Layer(kind=BatchNormalization))

    l_rate = 0.0015  # hp.Float("learning_rate", 0.001, 0.005, step=0.001)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=l_rate)
    loss = hp.Choice("loss", ["mae", "mse"])
    out_len = hp.Int("out", 1, 10)
    return ann(layers, out_len, loss, optimizer)


def load_inp_folder(path):
    data = np.load(os.path.join(path, "data.npz"))
    iref = pd.read_pickle(os.path.join(path, "inp_ref.pkl"))
    oref = pd.read_pickle(os.path.join(path, "out_ref.pkl"))
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"], iref, oref


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for the model hyperparameter tuning.")
    parser.add_argument("inp-folder")
    return parser.parse_args().__dict__


def main():
    args = get_args()
    folder = args["inp-folder"]
    del args["inp-folder"]

    loc = os.path.dirname(sys.argv[0])
    data_loc = os.path.join(loc, "Data")
    log_dir = os.path.join(loc, "logs")
    log_search_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%m%d-%H%M"))

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if not os.path.exists(log_search_dir):
        os.mkdir(log_search_dir)

    x_tr, x_te, y_tr, y_te, i_ref, o_ref = load_inp_folder(os.path.join(data_loc, folder))

    print(i_ref)
    print(o_ref)

    hp = kt.HyperParameters()
    if len(y_tr.shape) > 1:
        hp.Fixed("out", y_tr.shape[1])
    else:
        hp.Fixed("out", 1)

    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_search_dir,
        histogram_freq=1,
        embeddings_freq=1,
        write_graph=True,
        update_freq='batch')

    tuner = kt.tuners.BayesianOptimization(
        hp_estimator,
        "val_mae",
        500,
        num_initial_points=10,
        directory=log_search_dir,
        hyperparameters=hp,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        project_name="XFEL-ANN"
    )

    print(tuner.search_space_summary())

    x_tr_v, x_val, y_tr_v, y_val = train_test_split(x_tr, y_tr)  # get validation set

    tuner.search(x_tr_v,
                 y_tr_v,
                 epochs=10,
                 verbose=2,
                 validation_data=(x_val, y_val),
                 batch_size=1000,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=3)]
                 )
    best_model = tuner.get_best_models()[0]
    print(tuner.get_best_hyperparameters()[0].values)
    best_model.fit(x_tr, y_tr, batch_size=1000, epochs=20000)
    print(best_model.evaluate(x_te, y_te))


if __name__ == "__main__":
    sys.exit(main())
