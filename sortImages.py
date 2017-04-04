#Script for sorting training images into a positive and negative training set

import os

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


dirs = os.walk("./images")

for directory in dirs:
    labelFile = open("labels.csv")
    idLabels = parseLabels(labelFile)
    currDir = os.path.basename(directory[0])
    label = -1
    try:
        label = idLabels[currDir]
    except:
	continue
    if label == "0":
        os.renames(directory[0], "./negTrain/" + currDir)
    else:
        os.renames(directory[0], "./posTrain/" + currDir)
