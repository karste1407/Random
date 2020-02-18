from myFunctions import *
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

def main():

    nTest = 200          #Number of samples for test data
    split = 0.8          #Ration of training compared to validation

    X = generateData()
    X_train,X_val,X_test = generateSubsets(X,nTest,split)

    '''
    nPerceptrons = 8
    nOut = 1

    inputs = keras.Input(shape=(X_train.shape[0],), name='time-shifts')
    x = layers.Dense(nPerceptrons, activation='sigmoid', name='dense_1')(inputs)
    x = layers.Dense(nPerceptrons, activation='sigmoid', name='dense_2')(x)
    outputs = layers.Dense(nOut, activation='sigmoid', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.SGD(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy()]
        )
    '''





if __name__ == "__main__":
    main()
