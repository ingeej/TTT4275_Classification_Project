
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

def importNumbers(imageFile, labelFile): #28*28 = 748, de 748 første bør tilhøre første bilde
    rawDataImage = np.fromfile(imageFile)
    rawDataLabels = np.fromfile(labelFile)
    #trainingSet = open(imageFile, "rb")

    return rawDataImage, rawDataLabels

#def NearestNeighbour(trainingData, ):
(train_X, train_y), (test_X, test_y) = mnist.load_data()

trainingSet, trainingLabel = importNumbers('Code\mnist\\train_images.bin', 'Code\mnist\\train_labels.bin')
print(trainingSet[:10])
