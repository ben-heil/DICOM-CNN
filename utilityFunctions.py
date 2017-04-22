#File containing functions used across different CNN architectures

import os
import dicom
import math
import random
import numpy
import matplotlib.pyplot as plt
import time

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
    output = numpy.zeros((endSize,64,64))

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
    currDir = os.path.basename(dirs[scanNum][0])
    images = []
    #Add each array to a list
    for image in dirs[scanNum][2]:
	if image != "desktop.ini":
            imageData = numpy.load(dirs[scanNum][0] + '/' + image)
	    images.append(imageData)
    
    #Add the 3d array of all images from a patient to the list
    patientData = numpy.stack(images).astype(float)
    patientData -= patientData.mean()
    patientData = patientData / patientData.max()
    label = -1
    try:
        label = idLabels[currDir]
    except:
        label = -1
    return patientData, label
    

def readDicomScan(scanNum, dirName):
    print("Reading image " + str(scanNum) + " from " + dirName)
    dirGen = os.walk("./" + dirName)
    dirs = [dir for dir in dirGen]
    labelFile = open("./labels.csv")
    idLabels = parseLabels(labelFile)

    patientData = []
    currDir = os.path.basename(dirs[scanNum][0])
    dataList = []
    images = []
    #Add each array to a list
    for image in dirs[scanNum][2]:
        imageHandle = dicom.read_file(dirs[scanNum][0] + '/' + image)
	images.append(imageHandle)
    
    images.sort(key=lambda image: image.ImagePositionPatient)
    for image in images:
	plt.imshow(image.pixel_array)
	plt.draw()
	plt.pause(.01)
	print("Figure")
        dataList.append(image.pixel_array)
    #Add the 3d array of all images from a patient to the list
    patientData = numpy.stack(dataList).astype(float)
    patientData = patientData / patientData.max()
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
    
    for i in range(1,imageCount("validation") + 1):
        label, data = readScan(i, "validation")
	labels.append(label)
	patientData.append(data)

    return patientData, labels
