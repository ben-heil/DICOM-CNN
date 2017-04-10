#A modification of the smallPool CNN that uses relu activation units

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

#This function creates a dictionary that maps patient IDs to cancer incidence
def parseLabels(labelFile):
    #Throw away header
    labelFile.readline()
    patientLabels = {}
    
    #Read each line and add it to the dictionary
    for line in labelFile:
        line = line.strip().split(',')
        #print(line)
        #Throw out malformed input and empty lines
        if len(line) != 2:
            continue
        patientLabels[line[0]] = line[1]
    return patientLabels

def maxPool(images, endSize):
    totalImages = len(images)
    poolSize = totalImages // endSize
    extras = totalImages % endSize
    if extras > 0:
        extraIteration = totalImages // extras
    else:
        extraIteration = 0
    extraCount = 0
    i = 0
    iteration = 0
    output = numpy.zeros((endSize,512,512))

    while iteration < endSize:
    #while i < images.shape[0]:
        size = poolSize
        if iteration % (extraIteration - 1) == 0 and extraCount < extras:
            extraCount += 1
            size += 1

        choice = i + int(math.floor(random.random() * size))
        #print(output.shape)
        #print(images.shape)
        #print(iteration)
        #print(i)
        output[iteration] = images[choice]
        
        i += size 
        iteration += 1
    return output 
        

def imageCount(dirName):
    dirs = os.walk("./" + dirName)
    #Skip the directory itself, only look at subdirectories
    return len([dir for dir in dirs]) - 1

def readScan(scanNum, dirName):
    print("Reading image " + str(scanNum) + " from " + dirName)
    dirGen = os.walk("./" + dirName)
    dirs = [dir for dir in dirGen]
    labelFile = open("./labels.csv")
    idLabels = parseLabels(labelFile)
    
    patientData = []
    fileCount = 0
    currDir = os.path.basename(dirs[scanNum][0])
    dataList = []
    #Add each array to a list
    for image in dirs[scanNum][2]:
        imageHandle = dicom.read_file(dirs[scanNum][0] + '/' + image)
        imgData = imageHandle.pixel_array
        dataList.append(imageHandle.pixel_array)
            
        
        #Add the 3d array of all images from a patient to the list
        patientData = numpy.stack(dataList).astype(float)
    patientData -= numpy.mean(patientData)
    label = -1
    try:
        label = idLabels[currDir]
    except:
        label = -1
    return patientData, label 

def readValidationImages():
    labelFile = open("./labels.csv")
    idLabels = parseLabels(labelFile)
    dirs = os.walk("./validation")
    #Skip the directory itself, only look at subdirectories
    dirs.next()
    
    labels = []
    patientData = []
    for directory in dirs:
        currDir = os.path.basename(directory[0])
	if currDir in idLabels:
            dataList = []
            #Add each array to a list
            for image in directory[2]:
                if image == "desktop.ini":
                    continue
                imageHandle = dicom.read_file(directory[0] + '/' + image)
                imgData = imageHandle.pixel_array.astype(float)
                dataList.append(imageHandle.pixel_array)
            
            #Preprocess all images from the patient to set a zero mean
            dataList -= numpy.mean(dataList)
        
            #Add the 3d array of all images from a patient to the list
            patientData.append(numpy.stack(dataList))
            labels.append(idLabels[currDir])
    return patientData, labels

def main():
    parser = argparse.ArgumentParser(description = "Convolutional Neural Net")
    parser.add_argument("mode", help = "Specify whether to train or test the model")
    args = parser.parse_args()

    #Build the network
    #Create a 5d tensor type because it's not a thing by default until theano 0.9
    tensor5 = T.TensorType('float64', (False,)*5)
    
    Img = tensor5(name="Img")
    Lab = T.dscalar()
    endSize = 60 
    
    #Layer 0 (shrink images)
    p0Factor = 4
    
    pool0 = pool.pool_2d(Img, (p0Factor, p0Factor))
    layer0 = theano.function([Img], pool0)
    
    #Layer 1
    print("Compiling layer 1")
    f1size = 3
    f1Depth = 3
    numFilters1 = 2
    p1Factor = 4
    p1Depth = 2
    learnVal = numpy.array(.01)
    learnRate = theano.shared(learnVal, 'learnRate')
    
    #Load the shared variables if possible, otherwise initialize them
    try:
        f1File = open("F1_smallPooled.save" , "rb")
        F1 = pickle.load(f1File)
        f1File.close()
        b1File = open("b1_smallPooled.save", "rb")
        b1 = pickle.load(f1File)
        b1File.close()
    except:
        f1Arr = numpy.random.randn(numFilters1,1,f1Depth,f1size,f1size) 
        F1 = theano.shared(f1Arr, name = "F1")
        bias1 = numpy.random.randn()
        b1 = theano.shared(bias1, name = "b1")
    
    #Output = batches x 512 - (f1size-1) x 512 - (f1size-1) x endSize/f1Depth - 1 x channels
    conv1 = nnet.relu(nnet.conv3d(pool0, F1) + b1)
    pool1 = pool.pool_3d(conv1, (p1Depth,p1Factor,p1Factor), ignore_border = True)
    layer1 = theano.function([Img], pool1)
    
    #Layer 2
    print("Compiling layer 2")
    f2size = 3
    f2Depth = 3
    numFilters2 = 2 
    pool2Factor = 4
    pool2Depth = 5
    
    try:
        f2File = open("F2_smallPooled.save", "rb")
        F2 = pickle.load(f2File)
        f2File.close()
        b2File = open("b2_smallPooled.save", "rb")
        b2 = pickle.load(b2File)
        b2File.close()
    except:
        f2Arr = numpy.random.randn(numFilters2, numFilters1, f2Depth, f2size, f2size)
        F2 = theano.shared(f2Arr, name = "F2")
        bias2 = numpy.random.randn()
        b2 = theano.shared(bias2, name = "b2")
    conv2 = nnet.relu(nnet.conv3d(pool1, F2) + b2)
    pool2 = pool.pool_3d(conv2, (pool2Depth,pool2Factor,pool2Factor), ignore_border = True)
    layer2 = theano.function([Img], pool2)
    
    #Calculate the size of the output of the second convolutional layer
    convOutLen = (((512//p0Factor - f1size) //p1Factor + 1) - f2size) // pool2Factor 
    convOutDepth = (((endSize -f1Depth) // p1Depth + 1) - f2Depth) //pool2Depth 
    convOutLen = convOutLen * convOutLen * numFilters2 * convOutDepth
    
    #Layer 3
    print("Compiling layer 3")
    try:
        b3File = open("b3_smallPooled.save", "rb")
        b3 = pickle.load(b3File)
        b3File.close()
        w3File = open("w3_smallPooled.save", "rb")
        w3 = pickle.load(w3File)
        w3File.close()
    except:
        b3arr = numpy.random.randn()
        b3 = theano.shared(b3arr, name = "b3")
        w3arr = numpy.random.randn(convOutLen, convOutLen // numFilters2)
        w3 = theano.shared(w3arr, name = "w3")
    hidden3 = theano.dot(pool2.flatten(), w3) + b3
    layer3 = theano.function([Img], hidden3)
    
    #Layer 4
    print("Compiling layer 4")
    try:
        w4File = open("w4_smallPooled.save", "rb")
        w4 = pickle.load(w4File)
        w4File.close()
        b4File = open("b4_smallPooled.save", "rb")
        b4 = pickle.load(b4File)
        b4File.close()
    except:
        w4arr = numpy.random.randn(convOutLen // numFilters2)
        w4 = theano.shared(w4arr, name = "w4")
        b4arr = numpy.random.randn()
        b4 = theano.shared(b4arr, name = "b4")
    hidden4In = nnet.relu(hidden3)
    hidden4 = theano.dot(hidden4In, w4) + b4
    layer4 = theano.function([Img], hidden4)
    
    #Output layer
    print("Compiling training and validation functions")
    output = nnet.sigmoid(hidden4)
    
    error = T.sqr(abs(output - Lab))
    F1Grad = T.grad(error, F1)
    F2Grad = T.grad(error, F2)
    w3Grad = T.grad(error, w3)
    w4Grad = T.grad(error, w4)
    b1Grad = T.grad(error, b1)
    b2Grad = T.grad(error, b2)
    b3Grad = T.grad(error, b3)
    b4Grad = T.grad(error, b4)
    
    validate = theano.function([Img, Lab], error)
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
        valImages, valLabels = readValidationImages()
        bestErr = float("inf")
        posPatientCount = imageCount("posTrain")
        negPatientCount = imageCount("negTrain")
        
        logFile = open("smallPoolCNNLog.txt", "w")
        
        bestErr = float("inf")
        
        for i in range(1000):
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
            #print(layer2(posImage.reshape(1,1,endSize,512,512)).shape)
            print("Training...")
            posErr = train(posImage.reshape(1,1,endSize,512,512), int(posLabel))
            negErr = train(negImage.reshape(1,1,endSize,512,512), int(negLabel))
            
        
            logFile.write("Pos err = " + str(posErr))
            logFile.write("\tNeg err = " + str(negErr) + "\n")
            print("Pos err = " + str(posErr))
            print("Neg err = " + str(negErr))
        
            #Use validation set to test error
            if i%5 == 0:
                currErr = 0
        
                for j in range(len(valImages)):
                    label = valLabels[j]
                    print("Pooling image " + str(j))
                    valImage = maxPool(valImages[j], endSize)
                    currErr += validate(valImage.reshape(1,1,endSize,512,512), int(label))
        
                print("Validation err = " + str(currErr))
                if  currErr < bestErr:
                    bestErr = currErr
                    print("Saving weight data")
                    iterationFile = open("smallPoolCNNErrorIteration.txt", "w+")
                    iterationFile.write(str(i) + "\n" + str(bestErr))
                    iterationFile.close()
                    try:
                        f1File = open("F1_smallPooled.save" , "wb")
                        b1File = open("b1_smallPooled.save" , "wb")
                        f2File = open("F2_smallPooled.save", "wb")
                        b2File = open("b2_smallPooled.save", "wb")
                        w3File = open("w3_smallPooled.save", "wb")
                        b3File = open("b3_smallPooled.save", "wb")
                        w4File = open("w4_smallPooled.save", "wb")
                        b4File = open("b4_smallPooled.save", "wb")
            
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
            totalErr = totalErr + validate(image.reshape(1,1,endSize,512,512),
                int(label))
            print(totalErr)
        
        outFile = open("smallPoolTestAccuracy.txt", 'w')

        accuracy = totalErr / testCount
        outFile.write("Accuracy: " + str(accuracy))
        outFile.close()

if __name__ == "__main__":
    main()


#result = layer1(patientData[0][0].reshape(1,1,512,512))
#plt.subplot(1,3,1)
#plt.imshow(patientData[0][1])
#plt.gray()
#plt.subplot(1,3,2)
#plt.imshow(result[0][0])
#plt.gray()
#result = layer2(patientData[0][0].reshape(1,1,512,512))
#print(result[0][0].shape)
#plt.subplot(1,3,3)
#plt.imshow(result[0][0])
#plt.gray()
#plt.show()
