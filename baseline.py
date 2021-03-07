import sys, os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def load_inp_folder(path):
    data = np.load(os.path.join(path, "data.npz"))
    iref = pd.read_pickle(os.path.join(path, "inp_ref.pkl"))
    oref = pd.read_pickle(os.path.join(path, "out_ref.pkl"))
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"], iref, oref


def main():
    loc = os.path.dirname(sys.argv[0])
    data_loc = os.path.join(loc, "Data")
    folder = sys.argv[1]

    x_tr, x_te, y_tr, y_te, i_ref, o_ref = load_inp_folder(os.path.join(data_loc, folder))

    print(i_ref)
    print(o_ref)

    lin = LinearRegression()
    lin.fit(x_tr, y_tr)

    pred = lin.predict(x_te)

    print(np.mean(np.abs(pred-y_te)))

if __name__ == '__main__':
    main()