# file: CNN.py
# description: Implements a CNN that can be used to analyze DICOM images
# author: Ben Heil
# date: 4/27/17

import matplotlib.pyplot as plt
import argparse
import math
import random
import keras
import sys
import numpy
from utilityFunctions import imageCount, readScan, readValidationImages, readBatch, readTest
from keras.models import Sequential
from keras.layers import Dense, convolutional, Activation, Flatten, pooling 
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import keras.backend as K

def main():
    parser = argparse.ArgumentParser(description = "Convolutional Neural Net")
    parser.add_argument("mode", help = "Specify whether to train or test the model")
    args = parser.parse_args()

    #Define CNN architecture

    #Load model if it has a savefile, otherwise reinitialize
    try:
        print("Loading model")
        model = keras.models.load_model("./CNN.save")
    except:
        print("save file not found, reinstantiating...")
        #Initialize a model with five convolutional layers and one hidden layer
        model = Sequential([
            convolutional.Conv2D(16, 15, 
            input_shape = (512,512,1)),
            pooling.MaxPooling2D(pool_size = (4)),
            convolutional.Conv2D(16, 7),
            Activation('tanh'),
            BatchNormalization(),
            convolutional.Conv2D(16, 7),
            Activation('tanh'),
            BatchNormalization(),
            convolutional.Conv2D(16, 7),
            Activation('tanh'),
            BatchNormalization(),
            pooling.MaxPooling2D(pool_size = (4)),
            convolutional.Conv2D(16, 3),
            Activation('tanh'),
            BatchNormalization(),
            pooling.MaxPooling2D(pool_size = (4)),
        Flatten(),
            Dense(16, activation = "tanh"),
        BatchNormalization(),
            Dense(1, activation = "tanh")])

    #Compile the model and use stochastic gradient descent to minimize error
    model.compile(loss = 'mean_squared_error', 
    optimizer = optimizers.SGD(lr = .01), metrics = ['acc'])  


    #Create a callback function that logs the results of training 
    log = keras.callbacks.CSVLogger("KerasLog.csv", append = True)
    #END OF ARCHITECTURE

    if args.mode.lower() == "test":
        print("Beginning test")
        testImages, testLabels = readTest()
        print(model.evaluate(testImages.reshape(40,512,512,1), testLabels))

    elif args.mode.lower() == "train":
        print("Reading validation images")

        batchSize = 2 
        valImages, valLabels = readValidationImages()
        
        bestLoss = float('inf')
        for i in range(1000):
            batch, labels = readBatch(batchSize)

            history = model.fit(x = batch.reshape(batchSize,512,512,1), 
            y = labels, epochs = 5,
            validation_data = (valImages.reshape(16,512,512,1), valLabels),
            callbacks = [log])

            #Check to see whether the most recent round of training led to an
            #improved accuracy rate
            currLoss = history.history['val_loss'][-1]
            if currLoss < bestLoss:
                print("Saving model with loss of " + str(currLoss))
                model.save("./CNN.save")


if __name__ == "__main__":
    main()

