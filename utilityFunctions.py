#File containing functions used across different CNN architectures

import os
import dicom
import math
import random
import numpy

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
