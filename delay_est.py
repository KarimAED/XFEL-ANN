import sys
import os

from estimator import ANN
from delay_prep import prep_delay_data

if len(sys.argv) not in (2, 3):
    raise ValueError("Wrong # of args")

home = os.path.join(sys.argv[1], "XFEL-ANN")
if len(sys.argv) == 3:
    shape = [int(layer) for layer in sys.argv[2].split(",")]
else:
    shape = [50, 50, 20, 20, 20]

x_tr, x_te, y_tr, y_te, x_ref, y_ref = prep_delay_data(home)

ann = ANN(shape=shape, epochs=20000, verbose=2, batch_norm=True)

ann.fit(x_tr, y_tr)
score = ann.score(x_te, y_te)

print(score)
