#File containing functions used across different CNN architectures

from skimage import exposure
import os
import dicom
import math
import random
import numpy
import matplotlib.pyplot as plt
import time

def imageCount(dirName):
    
    for root, dir, files in os.walk("./" + dirName):
        scans = [file  for file in files if file != "desktop.ini"]
        return len(scans)
    print(dirName + " is not a valid directory")
    return -1

def readScan(scanNum, dirName):
    
    for root, dir, files in os.walk("./" + dirName):
        scans = [file  for file in files if file != "desktop.ini"]
        print("Reading image " + scans[scanNum] + " from " + dirName)
        data = numpy.load("./" + dirName + "/" + scans[scanNum])
        #plt.imshow(data)
        #plt.show()
        data = exposure.equalize_adapthist(data)        
        #plt.imshow(data)
        #plt.show()
        return data
    print(dirName + " is not a valid directory")
    return -1
    
def readValidationImages():
    labels = []
    patientData = []
    
    for i in range(0,imageCount("tumorPosVal")):
        data = readScan(i, "tumorPosVal")
        labels.append(1)
        patientData.append(data)
    for i in range(0,imageCount("tumorNegVal")):
        data = readScan(i, "tumorNegVal")
        labels.append(0)
        patientData.append(data)

    patientData = numpy.stack(patientData).astype(float)

    return patientData, numpy.stack(labels)

def readBatch(batchSize):
    labels = []
    patientData = []
    
    posLen = imageCount("tumorPosTrain")
    negLen = imageCount("tumorNegTrain")

    posStart = random.randint(0,(posLen- (batchSize//2)) - 1)
    negStart = random.randint(0,(negLen- (batchSize//2)) - 1)

    for i in range(batchSize //2):
        data = readScan(posStart + i, "tumorPosTrain")
        labels.append(1)
        patientData.append(data)
        data = readScan(negStart + i, "tumorNegTrain")
        labels.append(0)
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    return patientData, numpy.stack(labels)

def readAll():
    labels = []
    patientData = []
    
    posLen = imageCount("tumorPosTrain")
    negLen = imageCount("tumorNegTrain")

    for i in range(posLen):
        data = readScan(i, "tumorPosTrain")
        labels.append(1)
        patientData.append(data)
    for i in range(negLen):
        data = readScan(i, "tumorNegTrain")
        labels.append(0)
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    patientData -= patientData.mean()
    patientData = patientData / patientData.max()
    return patientData, numpy.stack(labels)

def normReadAll():
    labels = []
    patientData = []
    
    posLen = imageCount("tumorPosTrain")
    negLen = imageCount("tumorNegTrain")

    for i in range(posLen):
        data = readScan(i, "tumorPosTrain")
        labels.append(1)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    for i in range(negLen):
        data = readScan(i, "tumorNegTrain")
        labels.append(0)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    return patientData, numpy.stack(labels)

def readTest():
    labels = []
    patientData = []
    
    posLen = imageCount("tumorPosTest")
    negLen = imageCount("tumorNegTest")

    for i in range(posLen):
        data = readScan(i, "tumorPosTest")
        labels.append(1)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    for i in range(negLen):
        data = readScan(i, "tumorNegTest")
        labels.append(0)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    return patientData, numpy.stack(labels)
    
