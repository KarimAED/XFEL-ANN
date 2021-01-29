import sys
import os

import pandas as pd

from estimator import ANN
from delay_prep import prep_delay_data

if len(sys.argv) != 2:
    raise ValueError("Wrong # of args")

home = os.path.join(sys.argv[1], "XFEL-ANN")

x_tr, x_te, y_tr, y_te, x_ref, y_ref = prep_delay_data(home)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

print(x_ref)
print(y_ref)

ann = ANN(shape=[50, 50, 20, 20], epochs=20000, verbose=2)

ann.fit(X_tr, y_tr)
score = ann.score(X_te, y_te)

print(score)
