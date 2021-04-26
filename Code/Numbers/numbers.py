import numpy as np
import matplotlib.pyplot as plt

def importNumbers(imageFile):

    #rawDataImage = np.fromfile(filename)
    #rawDataLabels = np.fromfile(labelFile)
    trainingSet = open(imageFile, "rb")

    return trainingSet

#def NearestNeighbour(trainingData, ):

trainingSet = importNumbers('Code\mnist\train_images.bin')
print(trainingSet[:10])
