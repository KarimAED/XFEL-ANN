# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:41:59 2020

@author: Karim
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

strategy = tf.distribute.MirroredStrategy()


# Did not use dataclass due to weird __dict__ behaviour
class Layer:

    def __init__(self, kind=Dense, units=10, activation="relu", kernel_regularizer="l2", rate=0.0):
        self.kind = kind
        self.units = units
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.rate = rate

        if self.kind in (Dropout, BatchNormalization):
            self.units = None
            self.activation = None
            self.kernel_regularizer = None
            if self.kind is BatchNormalization:
                self.rate = None
        else:
            self.rate = None

    def get_attr(self):
        d = self.__dict__.copy()
        del d["kind"]
        return {k: v for k, v in d.items() if v is not None}


def ann(layer_list, out_shape, loss, opt):
    model = tf.keras.Sequential()
    for layer in layer_list:
        model.add(layer.kind(**layer.get_attr()))
    model.add(Dense(units=out_shape, activation="linear"))
    model.compile(opt, loss=loss, metrics=["mae"])
    return model

