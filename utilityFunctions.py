""" file: utilityFunctions.py
 description: contains functions that are used in manipulating DICOM images and reading data from files
 author: Ben Heil
 date: 4/27/17
"""

from skimage import exposure
import os
import dicom
import math
import random
import numpy
import matplotlib.pyplot as plt
import time

def imageCount(dirName):
    """Counts the number of files in a directory

    Counts the number of files in a given directory and returns the result. Ignores desktop.ini to allow for Windows compatibility

    Args:
        dirName: A string representation of the name of the directory to be opened. This directory should be a child of the working directory. EX: "trainingImages"

    Returns:
        An integer value for the number of files in the directory
    """

    for root, dir, files in os.walk("./" + dirName):
        scans = [file  for file in files if file != "desktop.ini"]
        return len(scans)
    print(dirName + " is not a valid directory")
    exit(-1)

def readScan(scanNum, dirName):
    """ Reads a DICOM file from a directory

    Given a directory name and the number of the file containing the image to be read, reads in and returns a numpy representation of the image. Also normalizes the pixel values of the image.

    Args:
        scanNum: An integer containing the one-based index of the file to be read
        dirName: A string representation of the name of the directory to be opened. This directory should be a child of the working directory. Ex: "trainingImages"

    Returns: 
        A numpy array containing the pixel data for the image
    """

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
    exit(-1)

def readValidationImages():
    """ Reads in all images in the validation directories

    Creates a numpy array containing the pixel data of all the images in the directories of validation images. Assumes that the subdirectories "PosVal" and "NegVal" exit in the working directory
    
    Args: None

    Returns: 
        A numpy array containing the pixel values for all validation images. Also returns a numpy array containing whether each image has a positive or negative label
    """
    
    labels = []
    patientData = []
    
    for i in range(0,imageCount("PosVal")):
        data = readScan(i, "PosVal")
        labels.append(1)
        patientData.append(data)
    for i in range(0,imageCount("NegVal")):
        data = readScan(i, "NegVal")
        labels.append(0)
        patientData.append(data)

    patientData = numpy.stack(patientData).astype(float)

    return patientData, numpy.stack(labels)

def readBatch(batchSize):
    """ Reads in a batch of images

    Given a batch size, reads a group of images at random containing an eqaul number of positive and negative examples. Assumes the directories "posTrain" and "negTrain" exist

    Args: 
        batchSize: An integer containing the total number of images to read. batchSize // 2 images will be read from the positive and negative training directories

    Returns:
        A numpy array containing the pixel values for the batch of images. Also returns a numpy array containing whether each image has a positive or negative label
    """

    labels = []
    patientData = []
    
    posLen = imageCount("PosTrain")
    negLen = imageCount("NegTrain")

    posStart = random.randint(0,(posLen- (batchSize//2)) - 1)
    negStart = random.randint(0,(negLen- (batchSize//2)) - 1)

    for i in range(batchSize //2):
        data = readScan(posStart + i, "PosTrain")
        labels.append(1)
        patientData.append(data)
        data = readScan(negStart + i, "NegTrain")
        labels.append(0)
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    return patientData, numpy.stack(labels)

def normReadAll():
    """ Reads all images in the training set

    Reads all images from the training set. Assumes that the working directory has subdirectories "PosTrain" and "NegTrain"

    Returns:
        A numpy array containing the pixel values for the batch of images. Also returns a numpy array containing whether each image has a positive or negative label
    """
    
    labels = []
    patientData = []
    
    posLen = imageCount("PosTrain")
    negLen = imageCount("NegTrain")

    for i in range(posLen):
        data = readScan(i, "PosTrain")
        labels.append(1)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    for i in range(negLen):
        data = readScan(i, "NegTrain")
        labels.append(0)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    return patientData, numpy.stack(labels)

def readTest():
    """ Reads all images in the test set

    Reads all images from the test set. Assumes that the working directory has subdirectories "PosTest" and "NegTest"

    Returns:
        A numpy array containing the pixel values for the batch of images. Also returns a numpy array containing whether each image has a positive or negative label
    """
    labels = []
    patientData = []
    
    posLen = imageCount("PosTest")
    negLen = imageCount("NegTest")

    for i in range(posLen):
        data = readScan(i, "PosTest")
        labels.append(1)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    for i in range(negLen):
        data = readScan(i, "NegTest")
        labels.append(0)
        data = exposure.equalize_adapthist(data)        
        patientData.append(data)
    
    patientData = numpy.stack(patientData).astype(float)
    return patientData, numpy.stack(labels)
    
