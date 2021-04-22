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
from model_fit import get_args, get_inp, fit_ann, fit_grad_boost


def main():
    args = get_args()
    x_tr, x_te, y_tr, y_te, i_ref, o_ref = get_inp(args["inp-folder"])

    print(i_ref)
    print(o_ref)

    start = time.time()

    if args["model"] == "ann":
        est = fit_ann(args, x_tr, y_tr)
        print(est.evaluate(x_te, y_te))
    elif args["model"] == "grad-boost":
        est = fit_grad_boost(x_tr, y_tr)
        print(np.mean(np.abs(est.predict(x_te)-y_te)))

    scores = []
    excluded_features = []
    excluded_index = []
    for i in range(len(i_ref.columns)):
        x_te_masked = []
        for j in range(len(i_ref.columns)):
            if j!=i:
                x_te_masked.append(x_te[:, j])
            elif j==i:
                x_te_masked.append(np.zeros(x_te.shape[0]))
        x_te_masked = np.stack(x_te_masked).T
        excluded_features.append(i_ref.columns[i])
        excluded_index.append(i)
        try:
            scores.append(est.evaluate(x_te_masked, y_te)[1])
        except:
            scores.append(np.mean(np.abs(est.predict(x_te_masked) - y_te)))
    dur = time.time() - start
    print(f"Runtime: {dur}")
    feature_rank = pd.DataFrame({"features": excluded_features, "mae_score": scores, "feat_ind": excluded_index})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)
    print(feature_rank.head(30))

    ranking = feature_rank["feat_ind"].values

    scores = []

    for l in range(len(ranking)):
        feats = ranking[:l+1]
        x_te_masked = []
        for j in range(len(i_ref.columns)):
            if j in feats:
                x_te_masked.append(x_te[:, j])
            else:
                x_te_masked.append(np.zeros(x_te.shape[0]))
        x_te_masked = np.stack(x_te_masked).T
        try:
            scores.append(est.evaluate(x_te_masked, y_te)[1])
        except:
            scores.append(np.mean(np.abs(est.predict(x_te_masked) - y_te)))

    print(scores)

    plt.style.use("mystyle-2.mplstyle")
    plt.figure(figsize=(20, 7))
    plt.bar(feature_rank["features"], feature_rank["mae_score"]*o_ref.loc["test_std"].values)
    plt.xticks(rotation=90)
    label = time.time()
    plt.ylabel("MAE in fs")
    plt.xlabel("Excluded Feature")
    plt.savefig("feat_"+args["model"]+"_"+args["inp-folder"]+"_"+str(label)+".pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 7))
    plt.plot(np.array(scores)*o_ref.loc["test_std"].values)
    plt.ylabel("MAE in fs")
    plt.xlabel("Number of Features used")
    plt.savefig("feat-count_"+args["model"]+"_"+args["inp-folder"]+"_"+str(label)+".pdf", bbox_inches="tight")

if __name__ == "__main__":
    sys.exit(main())
