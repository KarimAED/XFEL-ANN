import os
import sys
import argparse

import numpy as np
import pandas as pd
import kerastuner as kt
import tensorflow as tf
from sklearn.model_selection import KFold

from estimator import ANN


def hp_estimator(hp):
    shape = []
    for i in range(hp.Int("layers", 3, 10)):
        shape.append(hp.Int("units_"+str(i), 10, 50, step=10))
    drop_out = hp.Float("drop_out", 0.01, 0.1, step=0.01)
    activation = hp.Choice("activation", ["relu", "sigmoid"])
    loss = hp.Choice("loss", ["mae", "mse"])
    batch_norm = hp.Boolean("batch_norm", default=False)

    return ANN(shape=shape, drop_out=drop_out, activation=activation, loss=loss, batch_norm=batch_norm, verbose=0)


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
    x_tr, x_te, y_tr, y_te, i_ref, o_ref = load_inp_folder(os.path.join(data_loc, folder))

    print(i_ref)
    print(o_ref)

    tuner = kt.tuners.Sklearn(
        oracle=kt.oracles.Hyperband(
            objective=kt.Objective('score', 'max'),
            max_epochs=20000),
        hypermodel=hp_estimator,
        cv=KFold(5),
        directory='.',
        project_name=folder+"_hp")

    print(tuner.search_space_summary())

    tuner.search(x_tr,
                 y_tr)
    best_model = tuner.get_best_models()[0]
    print(tuner.get_best_hyperparameters()[0])
    best_model.fit(x_tr, y_tr)
    print(best_model.score(x_te, y_te))


if __name__ == "__main__":
    sys.exit(main())
