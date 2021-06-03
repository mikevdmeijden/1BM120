# -*- coding: utf-8 -*-

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def agent(item_count):
    """ 
    Requires the values of all the available items
    """
    model = Sequential()
    model.add(Dense(item_count * 3, activation="relu", input_shape=(item_count,)))
    model.add(Dense(item_count * 2, activation="relu"))
    model.add(Dense(item_count, activation="relu"))
    model.compile(Adam(1e-3), MeanSquaredError(), metrics=["accuracy"])
    
    return model

