import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for prediction plotting")
    parser.add_argument("inp-loc")
    args = parser.parse_args()
    return args.__dict__


def main():
    args = get_args()
    path = os.path.join(os.path.dirname(sys.argv[0]), args["inp-loc"])
    data = np.load(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, title="Delay Measurement vs Prediction")
    ax.grid()
    ax.set_xlabel("Measured delay (normalised)")
    ax.set_ylabel("Predicted delay (normalised)")
    ax.set_xlim(-3, 2)
    ax.set_ylim(-3, 2)
    ax.scatter(data["val"], data["pred"], s=1, alpha=0.3, label="MAE = 1.52fs")
    x = np.arange(-3, 2, 0.01)
    ax.plot(x, x, "k--")
    ax.legend()
    fig.show()
    input()
    return 0


if __name__ == "__main__":
    sys.exit(main())
