# -*- coding: utf-8 -*-

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.compat.v1.keras.initializers import RandomNormal

def agent(item_count):
    """ 
    Requires the values of all the available items
    """
    init = RandomNormal(mean=0.0, stddev=0.05, seed=10)
    model = Sequential()
    model.add(Dense(item_count * 3, activation="relu", input_shape=(item_count,), kernel_initializer = init))
    model.add(Dense(item_count * 2, activation="relu", kernel_initializer = init))
    model.add(Dense(item_count, activation="relu", kernel_initializer = init))
    model.add(Dense(item_count, kernel_initializer = init))
    model.compile(Adam(1e-3), MeanSquaredError(), metrics=["accuracy"])
    
    return model

