#Like the poolCNN, except with smaller filter sizes

from __future__ import division
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy
import dicom
import os 
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal.pool as pool
import theano
import pickle
import argparse
from utilityFunctions import parseLabels, maxPool, imageCount, readScan
from utilityFunctions import readValidationImages

def main():
    parser = argparse.ArgumentParser(description = "Convolutional Neural Net")
    parser.add_argument("mode", help = "Specify whether to train or test the model")
    args = parser.parse_args()

    #Build the network
    #Create a 5d tensor type because it's not a thing by default until theano 0.9
    tensor5 = T.TensorType('float64', (False,)*5)
    
    Img = tensor5(name="Img")
    Lab = T.dscalar()
    endSize = 100 
    
    #Layer 0 (shrink images)
    p0Factor = 1
    
    pool0 = pool.pool_2d(Img, (p0Factor, p0Factor))
    layer0 = theano.function([Img], pool0)
    
    #Layer 1
    print("Compiling layer 1")
    f1size = 3
    f1Depth = 3
    numFilters1 = 16
    p1Factor = 2
    p1Depth = 2
    learnVal = numpy.array(.05)
    learnRate = theano.shared(learnVal, 'learnRate')
    
    #Load the shared variables if possible, otherwise initialize them
    try:
        print("Loading weights...")
        f1File = open("F1_manyPooled.save" , "rb")
        F1 = pickle.load(f1File)
        f1File.close()
        b1File = open("b1_manyPooled.save", "rb")
        b1 = pickle.load(b1File)
        b1File.close()
    except:
        print("Save file not found, generating new weights")
        f1Arr = numpy.random.uniform(-.01,.01,(numFilters1,1,f1Depth,f1size,f1size)) 
	F1 = theano.shared(f1Arr, name = "F1")
        bias1 = numpy.ones((numFilters1,1))
        print(bias1.shape)
        b1 = theano.shared(bias1, name = "b1")
    
    #Output = batches x 64 - (f1size-1) x 64 - (f1size-1) x endSize/f1Depth - 1 x channels
    #Dimshuffle allows the bias to be broadcast along the filter dimension
    conv1 = T.tanh(nnet.conv3d(pool0, F1) + b1.dimshuffle(1,0,'x','x','x'))
    pool1 = pool.pool_3d(conv1, (p1Depth,p1Factor,p1Factor), ignore_border = True)
    layer1 = theano.function([Img], pool1)
    layer1PrePool = theano.function([Img], conv1)
    
    #Layer 2
    print("Compiling layer 2")
    f2size = 3
    f2Depth = 5
    numFilters2 = 16 
    pool2Factor = 2
    pool2Depth = 5
    
    try:
        f2File = open("F2_manyPooled.save", "rb")
        F2 = pickle.load(f2File)
        f2File.close()
        b2File = open("b2_manyPooled.save", "rb")
        b2 = pickle.load(b2File)
        b2File.close()
    except:
        f2Arr = numpy.random.uniform(-.01,.01,(numFilters2, numFilters1, f2Depth, f2size, f2size))
        F2 = theano.shared(f2Arr, name = "F2")
        bias2 = numpy.ones((numFilters2,1))
        print(bias2.shape)

        b2 = theano.shared(bias2, name = "b2")
    conv2 = T.tanh(nnet.conv3d(pool1, F2) + b2.dimshuffle(1,0,'x','x','x'))
    pool2 = pool.pool_3d(conv2, (pool2Depth,pool2Factor,pool2Factor), ignore_border = True)
    layer2PrePool = theano.function([Img], conv2)
    layer2 = theano.function([Img], pool2)
    
    #Calculate the size of the output of the second convolutional layer
    convOutLen = (((64//p0Factor - f1size) //p1Factor + 1) - f2size) // pool2Factor 
    print(convOutLen)
    convOutDepth = ((((endSize -f1Depth) // p1Depth + 1) - f2Depth) //pool2Depth) + 1 
    convOutLen = convOutLen * convOutLen * numFilters2 * convOutDepth
    print(convOutDepth)
    print(numFilters2)
    #Layer 3
    print("Compiling layer 3")
    layer3Size = 100
    try:
        b3File = open("b3_manyPooled.save", "rb")
        b3 = pickle.load(b3File)
        b3File.close()
        w3File = open("w3_manyPooled.save", "rb")
        w3 = pickle.load(w3File)
        w3File.close()
    except:
        b3arr = numpy.ones((layer3Size,1))
        b3 = theano.shared(b3arr, name = "b3")
        w3arr = numpy.random.uniform(-.01,.01,(convOutLen, layer3Size))
        w3 = theano.shared(w3arr, name = "w3")
    hidden3 = T.tanh(theano.dot(w3.T,pool2.flatten()) + b3.T)
    layer3 = theano.function([Img], hidden3)
    
    #Layer 4
    print("Compiling layer 4")
    try:
        w4File = open("w4_manyPooled.save", "rb")
        w4 = pickle.load(w4File)
        w4File.close()
        b4File = open("b4_manyPooled.save", "rb")
        b4 = pickle.load(b4File)
        b4File.close()
    except:
        w4arr = numpy.random.uniform(-.0001, .0001,(layer3Size, 1)) 
        w4 = theano.shared(w4arr, name = "w4")
        b4arr = numpy.ones((1))
        b4 = theano.shared(b4arr, name = "b4")
    hidden3 = T.tanh(hidden3)
    hidden4 = theano.dot(w4.T,hidden3.T) + b4
    layer4 = theano.function([Img], hidden4)

#    print(f1Arr.shape)
#    print(bias1.shape)
#    print(f2Arr.shape)
#    print(bias2.shape)
#    print(w3arr.shape)
#    print(b3arr.shape)
#    print(w4arr.shape)
#    print(b4arr.shape)

    #Output layer
    print("Compiling training and validation functions")
    output = nnet.sigmoid(hidden4)
    
    error = T.mean(T.sqr(abs(output - Lab)))
    F1Grad = T.grad(error, F1)
    F2Grad = T.grad(error, F2)
    w3Grad = T.grad(error, w3)
    w4Grad = T.grad(error, w4)
    b1Grad = T.grad(error, b1)
    b2Grad = T.grad(error, b2)
    b3Grad = T.grad(error, b3)
    b4Grad = T.grad(error, b4)
    
    gradPrint = theano.function([Img, Lab], F1Grad)
    validate = theano.function([Img, Lab], error)
    classify = theano.function([Img, Lab], abs(T.round(output) - Lab))
    train = theano.function([Img, Lab], error, updates = [(F1, F1 - F1Grad * learnRate),
             (F2, F2 - F2Grad * learnRate),
             (w3, w3 - w3Grad * learnRate),
             (w4, w4 - w4Grad * learnRate),
             (b1, b1 - b1Grad * learnRate),
             (b2, b2 - b2Grad * learnRate),
             (b3, b3 - b3Grad * learnRate),
             (b4, b4 - b4Grad * learnRate)])
    
    #END OF ARCHITECTURE
    

    if args.mode.lower() == "train":
        print("Reading validation images")
        valLabels, valImages = readValidationImages()
	try:
            errFile = open("manyPoolCNNErrorIteration.txt", 'r')
	    #Skip first line
	    errFile.readline()
            bestErr = float(errFile.readline())
	except:
	    bestErr = float("inf")
        posPatientCount = imageCount("posTrain")
        negPatientCount = imageCount("negTrain")
        
        logFile = open("manyPoolCNNLog.txt", "w")
        
        for i in range(10000):
            print(i)
            patientNum = int(math.floor(random.random() * posPatientCount ) + 1)
            posPatientData, posLabel = readScan(patientNum, "posTrain")
            patientNum = int(math.floor(random.random() * negPatientCount ) + 1)
            negPatientData, negLabel = readScan(patientNum, "negTrain")
            if posLabel == -1 or negLabel == -1:
                continue
        
            print("Pooling images")
            posImage = maxPool(posPatientData, endSize)
            negImage = maxPool(negPatientData, endSize)
            print("Pooling complete")
            
#	    print(layer1PrePool(posImage.reshape(1,1,endSize,64,64)).shape)
#	    print(layer1(posImage.reshape(1,1,endSize,64,64)).shape)
#	    plt.imshow(layer1(posImage.reshape(1,1,endSize,64,64))[0][0][20])
#	    plt.show()
#	    plt.imshow(layer1(posImage.reshape(1,1,endSize,64,64))[0][1][20])
#	    plt.show()
#	    plt.imshow(layer1(posImage.reshape(1,1,endSize,64,64))[0][2][20])
#	    plt.show()
#	    plt.imshow(layer1(posImage.reshape(1,1,endSize,64,64))[0][3][20])
#	    plt.show()
#	    plt.imshow(layer1(posImage.reshape(1,1,endSize,64,64))[0][4][20])
#	    plt.show()
#	    print(layer2PrePool(posImage.reshape(1,1,endSize,64,64)).shape)
#	    print(layer2(posImage.reshape(1,1,endSize,64,64)).shape)
#            print(layer3(posImage.reshape(1,1,endSize,64,64)).shape)
#            print(layer4(posImage.reshape(1,1,endSize,64,64)).shape)
            print("Training...")
	    posErr = train(posImage.reshape(1,1,endSize,64,64), int(posLabel))
            negErr = train(negImage.reshape(1,1,endSize,64,64), int(negLabel))
        
            logFile.write("Pos err = " + str(posErr))
            logFile.write("\tNeg err = " + str(negErr) + "\n")
	    print("Sum err = " + str(posErr + negErr))
            print("Pos err = " + str(posErr))
            print("Neg err = " + str(negErr))
        
            #Use validation set to test error
            if i%5 == 0:
                currErr = 0
        
                for j in range(len(valImages)):
                    label = valLabels[j]
                    print("Pooling image " + str(j))
                    valImage = maxPool(valImages[j], endSize)
                    currErr += validate(valImage.reshape(1,1,endSize,64,64), int(label))
        
                print("Validation err = " + str(currErr))
                if  currErr < bestErr:
                    bestErr = currErr
                    print("Saving weight data")
                    iterationFile = open("manyPoolCNNErrorIteration.txt", "w+")
                    iterationFile.write(str(i) + "\n" + str(bestErr))
                    iterationFile.close()
                    try:
                        f1File = open("F1_manyPooled.save" , "wb")
                        b1File = open("b1_manyPooled.save" , "wb")
                        f2File = open("F2_manyPooled.save", "wb")
                        b2File = open("b2_manyPooled.save", "wb")
                        w3File = open("w3_manyPooled.save", "wb")
                        b3File = open("b3_manyPooled.save", "wb")
                        w4File = open("w4_manyPooled.save", "wb")
                        b4File = open("b4_manyPooled.save", "wb")
            
                        pickle.dump(F1, f1File)
                        pickle.dump(b1, b1File)
                        pickle.dump(F2, f2File)
                        pickle.dump(b2, b2File)
                        pickle.dump(w3, w3File)
                        pickle.dump(b3, b3File)
                        pickle.dump(w4, w4File)
                        pickle.dump(b4, b4File)
                    
                        f1File.close()
                        b1File.close()
                        f2File.close()
                        b2File.close()
                        w3File.close()
                        b3File.close()
                        w4File.close()
                        b4File.close()
        
                    except:
                        print("Error dumping weight data")
    
    if args.mode.lower() == "test":
        print("Testing")
        testCount = imageCount("test")
        totalErr = 0
        for i in range(testCount):
            print(i)
            image, label = readScan(i+1, "test")
	    print(image.shape)
            print(label)
	    image = maxPool(image, endSize)
            totalErr = totalErr + classify(image.reshape(1,1,endSize,64,64),
                int(label))
            print(totalErr)
        
        outFile = open("manyPoolTestAccuracy.txt", 'w')

        avgErr = totalErr / testCount
        outFile.write("Accuracy: " + str(1 - avgErr))
        outFile.close()


    if args.mode.lower() == "visualize":
        image, label = readScan(2, "test")
        print(image.shape)
        print(label)
        image = maxPool(image, endSize)
        result = layer1(image.reshape(1,1,endSize,64,64))
        plt.subplot(2,2,1)
        print(result.shape)
        plt.imshow(image[40])
        plt.subplot(2,2,2)
        plt.imshow(result[0][1][15])
        plt.subplot(2,2,3)
        plt.imshow(result[0][2][15])
        plt.subplot(2,2,4)
        plt.imshow(result[0][3][15])
	plt.show()

if __name__ == "__main__":
    main()


#result = layer1(patientData[0][0].reshape(1,1,64,64))
#plt.subplot(1,3,1)
#plt.imshow(patientData[0][1])
#plt.gray()
#plt.subplot(1,3,2)
#plt.imshow(result[0][0])
#plt.gray()
#result = layer2(patientData[0][0].reshape(1,1,64,64))
#print(result[0][0].shape)
#plt.subplot(1,3,3)
#plt.imshow(result[0][0])
#plt.gray()
#plt.show()
