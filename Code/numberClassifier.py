
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time

from multiprocessing import Process
t1 = time.time()



def importBIN():
    nSamples = 60000
    nTests = 1000
    #nClasses = 10

    with open('Code/mnist/train_images.bin','rb') as binaryFile:
        imgB = binaryFile.read()
    with open('Code/mnist/train_labels.bin','rb') as binaryFile:
        lbB = binaryFile.read()
    with open('Code/mnist/test_images.bin','rb') as binaryFile:
        tstimgB = binaryFile.read()
    with open('Code/mnist/test_labels.bin','rb') as binaryFile:
        tstlbB = binaryFile.read()

    train_x = np.reshape(np.frombuffer(imgB[16:16+784*nSamples], dtype=np.uint8), (nSamples,784))
    train_y = np.frombuffer(lbB[8:nSamples+8], dtype=np.uint8)
    test_x = np.reshape(np.frombuffer(tstimgB[16:16+784*nTests], dtype=np.uint8), (nTests,784))
    test_y = np.frombuffer(tstlbB[8:nTests+8], dtype=np.uint8)

    return train_x, train_y, test_x, test_y



def printNumber(first_image):
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray', vmin=0, vmax=255)
    plt.show()

def eucledianDistance(img1, img2):
    return np.sum(differenceImage(img1, img2))
    #diffPic = np.array((28,28))
    #for row in range(28):
    #   for pixel in row:
    #       diffPic[row,pixel] = abs(img1[row, pixel]] - img2[row, pixel])
    #return diffPic
    #return np.sum(np.multiply((img1 - img2).T, (img1 - img2)))

def NNpredictor(test, ref, refLables, trueLables):  # No need for teslabels (not without clustering)

    predictedLables = []
    #i = 0
    for testPicture in test:
        distance = []  # Distance array, from testpic to all refpics
        #correctLable = trueLables[i]
        for refPicture in ref:
            distance.append(eucledianDistance(refPicture, testPicture))
       
        testLable = refLables[np.argmin(distance)]  # Index to lowest distance
        predictedLables.append(testLable)  # List with the
        """
        if correctLable != testLable:
            print("Correct label", correctLable, " predicted lable: ", testLable)
            printNumber(testPicture)
            printNumber(ref[np.argmin(distance)])
            printNumber(plotDiffPics(testPicture,ref[np.argmin(distance)]))
        
        
        i+=1
        """
    return predictedLables


def confusionMatrix(test, trueLables, ref, refLables):
    predictedLables = NNpredictor(test, ref, refLables, trueLables)
    #print("actual label: %s,\n   predicted: %s \n" % (trueLables, np.array(predictedLables)))
    confuMatrix = np.zeros((10, 10))
    for i in range(len(predictedLables)):
        confuMatrix[predictedLables[i], trueLables[i]] += 1
    correct = 0
    for el in range(10):
        correct += confuMatrix[el][el]
    errorRate = (len(predictedLables) - correct) / len(predictedLables)

    return confuMatrix, errorRate


def plotDiffPics(img1, img2):
    #tempArr = np.zeros((28,28))
    temp1 = differenceImage(img1, img2)
    #printNumber(temp1)
    #diffPic = np.sqrt(np.square(temp1))
    #diffPic = np.multiply((img1 - img2).T, (img1- img2))
    return temp1

def differenceImage(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return a * b

#Load dataset (60000x28x28) 
(train_x, train_y), (test_x, test_y) = mnist.load_data()
#train_x, train_y, test_x, test_y = importBIN()
N = 1000 #number of pictures we want to use in each training 
M = 1

conMatrix, ER= confusionMatrix(test_x[:10], test_y[:10],  train_x, train_y)
print(conMatrix, "\n ER: ", ER)

"""
imgblack = np.zeros((28,28))
imgWhite = np.full((28,28),255.0)
imgWhite[5][5] = 0
imgblack[10][10] = 255

printNumber(plotDiffPics(imgWhite, imgblack))
"""
t2 = time.time()
t = t2 - t1
print("Runtime: %.20f" % t)
