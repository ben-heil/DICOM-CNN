#An example of a few useful things that can be done with DICOM files

import numpy
import dicom
import os 

#This function creates a dictionary that maps patient IDs to cancer incidence
def parseLabels(labelFile):
    #Throw away header
    labelFile.readline()
    patientLabels = {}

    #Read each line and add it to the dictionary
    for line in labelFile:
        line = line.strip().split(',')
        print(line)
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
#Print information about each file
#for directory in dirs:
#    currDir = os.path.basename(directory[0])
#    if currDir in idLabels:
#        print(idLabels[currDir])
#    for image in directory[2]:
#        imageHandle = dicom.read_file(directory[0] + '/' + image)
#        print(imageHandle)
#        print(imageHandle.pixel_array)

#Create a list of tuples mapping 3d numpy arrays of image data to their label
for directory in dirs:
    currDir = os.path.basename(directory[0])
    patientData = []
    if currDir in idLabels:
        dataList = []
        #Add each array to a list
        for image in directory[2]:
            imageHandle = dicom.read_file(directory[0] + '/' + image)
            dataList.append(imageHandle.pixel_array)
        #Add the 3d array of all images from a patient to the list
        patientData.append((numpy.stack(dataList), idLabels[currDir]))
        print(patientData)
print(patientData)
