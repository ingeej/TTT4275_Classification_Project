
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.cluster import KMeans
import time
from scipy import stats
from multiprocessing import Process
t1 = time.time()



def importBIN():
    nSamples = 60000
    nTests = 1000

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
    i = 0
    for testPicture in test:
        distance = []  # Distance array, from testpic to all refpics
        correctLable = trueLables[i]
        for refPicture in ref:
            distance.append(eucledianDistance(refPicture, testPicture))
       
        testLable = refLables[np.argmin(distance)]  # Index to lowest distance
        predictedLables.append(testLable)  # List with the
        """
        if correctLable != testLable:
            print("Correct label", correctLable, " predicted lable: ", testLable)
            print(eucledianDistance(testPicture, ref[np.argmin(distance)]))
            printNumber(testPicture)
            printNumber(ref[np.argmin(distance)])
            printNumber(differenceImage(testPicture,ref[np.argmin(distance)]))

        i+=1
        """
    return predictedLables


def confusionMatrix(test, testLables, ref, refLables, k=0):
    if k==0:
        predictedLables = NNpredictor(test, ref, refLables, testLables)
    else:
        predictedLables = KNN(test, ref, testLables, refLables, k)
    #print("actual label: %s,\n   predicted: %s \n" % (trueLables, np.array(predictedLables)))
    confuMatrix = np.zeros((10, 10))
    for i in range(len(predictedLables)):
        confuMatrix[predictedLables[i], testLables[i]] += 1
    correct = 0
    for el in range(10):
        correct += confuMatrix[el][el]
    errorRate = (len(predictedLables) - correct) / len(predictedLables)

    return confuMatrix, errorRate



def differenceImage(img1, img2):
    posDiff = img1-img2 # Since we import uint8, this will give zero for img1[i]-img2[i] < 0
    negDiff = np.uint8(img1<img2) * 254 + 1 # Making array with
    #negDiff = img2-img1
    #return posDiff + negDiff 
    return posDiff * negDiff

#def sort()



def getClusters(ref, refLables, nClusters=64):
    
    #clusters = [] # [(label, kmeans[])] # 10*64*28*28
    clusterData = [] # [kmeans] 10*64*28*28
    classes = range(len(np.unique(refLables))) #creates a list [0:9]

    for c in classes: # For every number 0-9
        i = np.where(refLables == c)[0] #indices for the number
        data = ref[i] # All pictures of number c
        data = data.reshape((data.shape[0], -1)) #make it 1d
        kmeans = KMeans(n_clusters=nClusters, n_init=20, n_jobs=1) # mean-picture

        kmeans.fit_predict(data) # cluster centers and index 
        cluster = kmeans.cluster_centers_ #
        cluster = cluster.reshape((64,28,28))
        clusterData.extend(cluster)

    clusterData = np.array(clusterData)
    clusterData = np.uint8(clusterData)
    #create a 10*64 array with 0-9

    clusterLables = []
    for c in range(10):
        clusterLables.extend([c]*nClusters)
    clusterLables = np.array(clusterLables)

    return clusterData, clusterLables


def KNN(test, ref, testLables, refLables, k):
    predictedLables = []
    for testPicture in test:
        distance = []  # Distance array, from testpic to all refpics
        #correctLable = trueLables[i]
        for refPicture in ref:
            distance.append(eucledianDistance(refPicture, testPicture)) #distances
        knnIndex = np.argpartition(distance,k)[:k] # List with index to k lowest distances
        knnValue = np.take(refLables, knnIndex)
        #print(stats.mode(knnValue))
        predictedLables.append(np.bincount(knnValue).argmax())  # Mode of KNN-list
    return predictedLables

#Saves 3d array to file

def saveFile(arr, name):
    np.save(name, arr)

def loadFile(name):
    loaded_arr = np.load(name)
    return loaded_arr

#Load dataset (60000x28x28) 
(train_x, train_y), (test_x, test_y) = mnist.load_data()
#train_x, train_y, test_x, test_y = importBIN()

#task 1
#conMatrix, ER= confusionMatrix(test_x[:10], test_y[:10],  train_x, train_y,0)
#print(conMatrix, "\n ER: ", ER)

#task 2

N = 6000 # Number of pictures we want to use in each training 
M = 10 # Number of classes
nClusters = 64 # Number of clusters for each class

#clusterData, clusterLables = getClusters(train_x, train_y, nClusters)

clusterData = loadFile('clusterData.npy')
clusterLables = loadFile('clusterLables.npy')




conMatrix, ER= confusionMatrix(test_x, test_y,  clusterData, clusterLables,0)
print(conMatrix)
print("Error",ER)


#task 2c 

#conMatrix, ER= confusionMatrix(test_x[10:30], test_y[10:30],  train_x, train_y,7)
#print(conMatrix, "\n ER: ", ER)

t2 = time.time()
t = t2 - t1
print("Runtime: %.20f" % t)
#saveFile(clusterLables, 'clusterLables.npy')