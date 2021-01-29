# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:41:59 2020

@author: Karim
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.estimator_checks import check_estimator

strategy = tf.distribute.MirroredStrategy()


class ANN(BaseEstimator):

    def __init__(self, shape=[20, 20, 20, 10], drop_out=0.0,
                 activation="relu", loss="mae", epochs=10000, verbose=0):

        self.shape = shape
        self.drop_out = drop_out
        self.activation = activation
        self.loss = loss
        self.epochs = epochs
        self.verbose = verbose

    def _more_tags(self):
        return {"multioutput": True}

    def set_up_model(self, inp_len, outp_len):
        """
        The function which sets up an individual ANN model (just takes the
        hyper-parameters and does not train it).

        Parameters
        ----------
        shape : int[]
            A list containing one value for each layer of the NN. The number is
            the number of neurons in the corresponding layer.
        drop_out_rate : float
            The drop out rate for the model, applied to each of the layers.
        activation : str
            String representation of the activation function.
        loss : str
            String representation of the loss function.

        Returns
        -------
        model : tf.keras.model
            The model compiled with the different parameters.

        """
        with strategy.scope():
            model = tf.keras.models.Sequential()
            model.add(InputLayer(inp_len))
            for i in self.shape[:-1]:
                model.add(Dense(i,
                                activation=self.activation,
                                kernel_regularizer="l2"))
                model.add(Dropout(self.drop_out))

            model.add(Dense(self.shape[-1], activation="sigmoid"))
            model.add(Dropout(self.drop_out))
            model.add(Dense(outp_len))

            optim = tf.keras.optimizers.Adagrad(learning_rate=0.0015)

            model.compile(optimizer=optim, loss=self.loss, metrics=['mae'])
            print(model.summary())
            self.ANN_ = model

    def fit(self, X, y):

        check_X_y(X, y, multi_output=True)

        # print(y)
        y = np.array(y)

        print(np.mean(y), np.std(y))

        if len(y.shape) == 2:
            o_shape = y.shape[1]
        else:
            o_shape = 1

        # print(y.shape, o_shape)

        with strategy.scope():
            self.set_up_model(X.shape[1], o_shape)

            self.ANN_.fit(X, y, 1000, self.epochs, verbose=self.verbose)

        return self

    def predict(self, X):

        check_is_fitted(self)
        check_array(X)

        with strategy.scope():
            return self.ANN_.predict(X)

    def score(self, X, y):
        check_is_fitted(self)
        check_X_y(X, y, multi_output=True)
        predi = self.predict(X).T[0]
        return -np.mean(np.abs(predi-y))


if __name__ == "__main__":
    check_estimator(ANN())
    print("Got here, check successful.")
