#An example of a few useful things that can be done with DICOM files


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

def imageCount():
    dirs = os.walk("./images")
    #Skip the directory itself, only look at subdirectories
    return len([dir for dir in dirs]) - 1

def readScan(scanNum):
    print("Reading " + str(scanNum))
    dirGen = os.walk("./images")
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
        patientData = numpy.stack(dataList)
    patientData -= numpy.mean(patientData)
    return patientData, idLabels[currDir]

def readImages():
    labelFile = open("./labels.csv")
    idLabels = parseLabels(labelFile)
    dirs = os.walk("./images")
    #Skip the directory itself, only look at subdirectories
    dirs.next()
    
    #Create a list of tuples mapping 3d numpy arrays of image data to their label
    labels = []
    patientData = []
    fileCount = 0
    for directory in dirs:
        fileCount += 1
        if fileCount %30 == 1:
            print("Read " + str(fileCount) + " files")
        currDir = os.path.basename(directory[0])
        if currDir in idLabels:
            dataList = []
            #Add each array to a list
            for image in directory[2]:
                imageHandle = dicom.read_file(directory[0] + '/' + image)
                imgData = imageHandle.pixel_array
                dataList.append(imageHandle.pixel_array)
            
            #Preprocess all images from the patient to set a zero mean
            dataList -= numpy.mean(dataList)
        
            #Add the 3d array of all images from a patient to the list
            patientData.append(numpy.stack(dataList))
            labels.append(idLabels[currDir])
    return patientData, labels


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
                imgData = imageHandle.pixel_array
                dataList.append(imageHandle.pixel_array)
            
            #Preprocess all images from the patient to set a zero mean
            dataList -= numpy.mean(dataList)
        
            #Add the 3d array of all images from a patient to the list
            patientData.append(numpy.stack(dataList))
            labels.append(idLabels[currDir])
    return patientData, labels



Img = T.tensor4(name="Img")
Lab = T.dscalar()

#Layer 1
f1size = 8
numFilters1 = 5
p1Factor = 3
learnRate = .1
#Load the shared variables if possible, otherwise initialize them
try:
    f1File = open("F1_vanilla.save" , "rb")
    F1 = pickle.load(f1File)
    f1File.close()
    b1File = open("b1_vanilla.save", "rb")
    b1 = pickle.load(f1File)
    b1File.close()
except:
    f1Arr = numpy.random.randn(numFilters1, 1, f1size ,f1size) 
    F1 = theano.shared(f1Arr, name = "F1")
    bias1 = numpy.random.randn()
    b1 = theano.shared(bias1, name = "b1")

#Output = batches x channels x 512 - f1size x 512 - f1size
conv1 = nnet.sigmoid(nnet.conv2d(Img, F1) + b1)
pool1 = pool.pool_2d(conv1, (p1Factor,p1Factor), ignore_border = True)
layer1 = theano.function([Img], pool1)

#Layer 2
f2size = 7
numFilters2 = 10
pool2Factor = 6
try:
    f2File = open("F2_vanilla.save", "rb")
    F2 = pickle.load(f2File)
    f2File.close()
    b2File = open("b2_vanilla.save", "rb")
    b2 = pickle.load(b2File)
    b2File.close()
except:
    f2Arr = numpy.random.randn(numFilters2, numFilters1, f2size, f2size)
    F2 = theano.shared(f2Arr, name = "F2")
    bias2 = numpy.random.randn()
    b2 = theano.shared(bias2, name = "b2")
conv2 = nnet.sigmoid(nnet.conv2d(pool1, F2) + b2)
pool2 = pool.pool_2d(conv2, (pool2Factor,pool2Factor), ignore_border = True)
layer2 = theano.function([Img], pool2)

#Calculate the size of the output of the second convolutional layer
convOutLen = (((512 - numFilters1) //p1Factor + 1) - numFilters2) // pool2Factor + 1
convOutLen = convOutLen * convOutLen * numFilters2

#Layer 3
try:
    b3File = open("b3_vanilla.save", "rb")
    b3 = pickle.load(b3File)
    b3File.close()
    w3File = open("w3_vanilla.save", "rb")
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
try:
    w4File = open("w4_vanilla.save", "rb")
    w4 = pickle.load(w4File)
    w4File.close()
    b4File = open("b4_vanilla.save", "rb")
    b4 = pickle.load(b4File)
    b4File.close()
except:
    w4arr = numpy.random.randn(convOutLen // numFilters2)
    w4 = theano.shared(w4arr, name = "w4")
    b4arr = numpy.random.randn()
    b4 = theano.shared(b4arr, name = "b4")
hidden4In = nnet.sigmoid(hidden3)
hidden4 = theano.dot(hidden4In, w4) + b4
layer4 = theano.function([Img], hidden4)

#Output layer
output = nnet.sigmoid(hidden4)

error = T.sqr(abs(output - Lab))
F1Grad = T.grad(error, F1)
F2Grad = T.grad(error, F2)
w3Grad = T.grad(error, w3)
w4Grad = T.grad(error, w4)

train = theano.function([Img, Lab], error, updates = [(F1, F1 - F1Grad * learnRate),
         (F2, F2 - F2Grad * learnRate),
         (w3, w3 - w3Grad * learnRate),
         (w4, w4 - w4Grad * learnRate)])


#END OF ARCHITECTURE

valImages, valLabels = readValidationImages()
besterr = float("inf")
patientCount = imageCount()
for i in range(10000):
    patientNum = int(math.floor(random.random() * patientCount ) + 1)
    patientData, label = readScan(patientNum)  
    for j in range(patientData.shape[0]):
        image = patientData[j]
        print(train(image.reshape(1,1,512,512), int(label)))
    
    #Use validation set to test error every 30 iterations
    if i%30 == 29:
        currErr = 0

        for j in range(len(valImages)):
            label = valLabels[j]
            for k in range(valImages[j].shape[0]):
                image = valImages[j][k]
                currErr += error(image.reshape(1,1,512,512), int(label))

        print("Validation err = " + str(currErr))
        if  currErr < besterr:
            besterr = currErr
            print("Saving weight data")
            try:
                f1File = open("F1_vanilla.save" , "wb")
                b1File = open("b1_vanilla.save" , "wb")
                f2File = open("F2_vanilla.save", "wb")
                b2File = open("b2_vanilla.save", "wb")
                w3File = open("r3_vanilla.save", "wb")
                b3File = open("b3_vanilla.save", "wb")
                w4File = open("w4_vanilla.save", "wb")
                b4File = open("b4_vanilla.save", "wb")
    
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
