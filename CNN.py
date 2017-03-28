#An example of a few useful things that can be done with DICOM files

import matplotlib
import matplotlib.pyplot as plt
import numpy
import dicom
import os 
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal.pool as pool
import theano

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



labelFile = open("./labels.csv")
idLabels = parseLabels(labelFile)

dirs = os.walk("./images")
#Skip the directory itself, only look at subdirectories
dirs.next()

#Create a list of tuples mapping 3d numpy arrays of image data to their label
for directory in dirs:
    currDir = os.path.basename(directory[0])
    patientData = []
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
        patientData.append((numpy.stack(dataList), idLabels[currDir]))
        print(patientData[0][0].shape)

Img = T.tensor4(name="Img")
images = patientData[0][0][1].reshape(1,1,512,512)
print(images.shape)
f1Arr = numpy.random.randn(1, 1, 512, 512) 

F1 = theano.shared(f1Arr, name = "F1")
bias1 = numpy.random.randn()
b1 = theano.shared(bias1, name = "b1")

conv1 = nnet.conv2d(Img, F1)

#pool1 = pool.pool_2d(conv1,(2,2), ignore_border = True)

layer1Func = theano.function([Img], nnet.sigmoid(conv1 + b1))

result = layer1Func(images)
print(result)
plt.subplot(1,2,1)
plt.imshow(patientData[0][0][1])
plt.gray()
plt.subplot(1,2,2)
plt.imshow(result[0][0])
plt.gray()
plt.show()
