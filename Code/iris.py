import numpy as np
import matplotlib.pyplot as plt




def importIris(filename,slice1, slice2):
    rawData = np.genfromtxt(filename, dtype=str, delimiter=",") #m*n-matrix, m = features, n = observations
    data = np.empty((3,50,4)) # 3*50-matrix, each column is one class
    indices = [0,0,0] #

    for row in rawData:
        if row[len(row)-1] == "Iris-setosa":
            data[0,indices[0]] = row[0:4]
            indices[0] +=1
        elif row[len(row)-1] == "Iris-versicolor":
            data[1,indices[1]] = row[0:4]
            indices[1] +=1
        elif row[len(row)-1] == "Iris-virginica":
            data[2,indices[2]] = row[0:4]
            indices[2] +=1
    
    #add 1 to end of features 
    classes, samples, features = data.shape
    data = np.dstack([data, np.ones([classes, samples, 1])])

    ts = data[:,:slice1]
    vs = data[:, slice1:(slice1+slice2)]
    return ts, vs, data

def sigmoid(zik):
    return 1/(1+np.exp(-zik))

def trainModel(data, N, lr): #X = data set, N = iterations, lr = alpha = learning rate

    classes, samples, features = data.shape

    #data = np.dstack([data, np.ones([classes, samples, 1])])


    W = np.zeros((classes, features))
    #w_0 = np.zeros((classes))
    #t_k = np.zeros((classes, 1)) 
    t_k = np.array([[1,0,0],[0,1,0],[0,0,1]]) # What we want to ajust to get low MSE
    g_k = np.zeros((classes))
    g_k[0] = 1 #remember 1

    MSEset = []

    for k in range(N):

        MSE = 0
        gMSE = 0
        for i in range(samples):
            x_k = data[:,i]
            #print(x_k)
            #Det jeg tror:
            #Cost function: MSE J(W, w0), skal deriveres
            #g_k er det faktiske punket (i settet) man ønsker å minimere avstanden til
            #t_k er punktet/linjen man ønsker å aproksimere verdien til best mulig
            #sigmoid er en funksjon vi bruker for å få sansynligheten for at hvilken blomst det er 
            #z_k blir hva man gjetter 
            #Ahh, så hvert datasett er jo en 4d vektor
            z_k = np.matmul(W,x_k.T) #+ w_0
            g_k = sigmoid(z_k) 

            firstFactor = np.multiply((g_k-t_k), g_k)
            secondFactor = (1-g_k)
            #grad =  np.multiply(np.multiply(((g_k-t_k),g_k)),(1-g_k))*x_k.T
            grad1 = np.multiply(firstFactor, secondFactor)
            gwz_k = x_k

            gMSE += np.matmul(grad1,  gwz_k)

            MSE += 1/2 * np.multiply((g_k-t_k).T,(g_k-t_k))
        MSEset.append(MSE)    

        W -= lr*gMSE
    MSEset = np.array(MSEset)
    return W #, MSEset

def classifier(W, test_set): # W must be the already trained matrix
    # Her skal hvert element i testsettet manipuleres med W?
    for s in test_set[2,:]:
        omega = int(np.argmax(np.matmul(W,s.T)))
        #print(np.matmul(W,test_set[4,0,:].T))
    return omega

def confuMatrix(W, test_set):
    classes, samples, features = test_set.shape
    confusion_matrix = np.zeros((classes, classes)) # n*m, n = prediction, m = true
    #Husk! Finn feil som funksjon av alpha
    for flower in range(classes):
        for s in test_set[flower,:]:
            omega = int(np.argmax(np.matmul(W,s.T)))
            confusion_matrix[omega][flower] += 1

    errorRate = (confusion_matrix[0,1]+confusion_matrix[0,2]+confusion_matrix[1,0]+confusion_matrix[1,2]+confusion_matrix[2,0]+confusion_matrix[2,1]) / (confusion_matrix.sum())
    return confusion_matrix, errorRate

def calculateAlpha(start, end, n, ts, trainSize):
    alphaSet = np.linspace(start, end, n)
    errorSet = []

    for a in alphaSet:
        W = trainModel(ts,trainSize, a)
        CF, ER = confuMatrix(W, ts) # Bruker training set, burde vi bruke test?
        errorSet.append(ER)

    errorSet = np.array(errorSet)
    alpha = np.argmin(errorSet)
    
    plt.plot(alphaSet, errorSet)
    plt.xlabel("alpha")
    plt.ylabel("Error rate")
    plt.show()
    print("Alpha: ", alpha)
    return errorSet, alpha

def plotHistogram(set, feature):
    bins = np.arange(0, 8, 1/7)
    featureNames = ["Sepal length", "Sepal Width", "Petal Length", "Petal Width"]
    fig, axs = plt.subplots(3,1)
    for (i, ax) in enumerate(axs.flat):
        ax.hist(set[i,:,feature].tolist(),bins=bins)
        ax.set_ylabel("Class %d" %(i))
        #ax.set_ylim((0,13))
    plt.xlabel("%s [cm]" % (featureNames[feature]))
    plt.show()

def plotHistogramAll(set):
    bins = np.arange(0, 8, 1/7)
    featureNames = ["Sepal length", "Sepal Width", "Petal Length", "Petal Width"]
    fig, axs = plt.subplots(2,2)

    for (i, ax) in enumerate(axs.flat):
        
        ax.hist(set[0,:,i].tolist(),bins=bins, alpha=0.6,)
        ax.hist(set[1,:,i].tolist(),bins=bins, alpha=0.6)
        ax.hist(set[2,:,i].tolist(),bins=bins, alpha=0.6)

        ax.set_xlabel("%s [cm]" % (featureNames[i]))
        
        #ax.set_ylabel("Feature %d" %(i))
        #ax.set_ylim((0,13))
    plt.tight_layout()
    
    plt.show()

def removeFeature(set, feature):
    return np.delete(set, feature, axis = 2)
        


#task 1a
ts1, vs1, data1 = importIris('Code\iris\iris.data', 30, 20)
print("Task 1a)")
print("Training samples:",ts1.shape)
print("Verification samples:",vs1.shape)

#1b
#calculateAlpha(0.01,0.1,30,ts1,1000) # valgt 0.009
#calculateAlpha(0.0001,0.05,30,ts1,500) # valgt 0.009
#ts1 = removeFeature(ts1, 1)
#vs1 = removeFeature(vs1, 1)
#ts1 = removeFeature(ts1, 0)
#vs1 = removeFeature(vs1, 0)

#1c
"""
W1= trainModel(ts1, 1000, 0.009)
CMts1, ERts1 = confuMatrix(W1,ts1) 
CMvs1, ERvs1 = confuMatrix(W1,vs1) 
print("All features")
print("ts = 30, vs = 20, ts used for training")
print("Training set error rate: ", ERts1)
print(CMts1)
print("Verification set error rate: ", ERvs1)
print(CMvs1)


plt.plot(MSEset[:,0,0])
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("MSE as a function of itereations with feature 1 removed, alpha = 0.009")
plt.show()

#1d
vs2, ts2, data2 = importIris('Code\iris\iris.data', 20, 30)

W2 = trainModel(ts2, 1000, 0.009)
CMts2, ERts2 = confuMatrix(W2,ts2) 
CMvs2, ERvs2 = confuMatrix(W2,vs2) 
print("\nts = 20, vs = 30, last 30 used for training")
print("Training set error rate: ", ERts2, "\n",CMts2)
print("Verification set error rate: ", ERvs2, "\n",CMvs2)

"""

#2a
plotHistogramAll(data1)
#plotHistogram(data1, 0)
#plotHistogram(data1, 1)
#plotHistogram(data1, 2)
#plotHistogram(data1, 3)
#
"""
ts1 = removeFeature(ts1, 1)
vs1 = removeFeature(vs1, 1)

W1 = trainModel(ts1, 1000, 0.009)

CMts1, ERts1 = confuMatrix(W1,ts1) 
CMvs1, ERvs1 = confuMatrix(W1,vs1) 
print("\nRemoved feature 1")
print("Training set error rate: ", ERts1)
print(CMts1)
print("Verification set error rate: ", ERvs1)


ts1 = removeFeature(ts1, 0)
vs1 = removeFeature(vs1, 0)

W1 = trainModel(ts1, 1000, 0.009)

CMts1, ERts1 = confuMatrix(W1,ts1) 
CMvs1, ERvs1 = confuMatrix(W1,vs1) 
print("\nRemoved feature 0")
print("Training set error rate: ", ERts1)
print(CMts1)
print("Verification set error rate: ", ERvs1)
print(CMvs1)

ts1 = removeFeature(ts1, 1)
vs1 = removeFeature(vs1, 1)

W1 = trainModel(ts1, 1000, 0.009)

CMts1, ERts1 = confuMatrix(W1,ts1) 
CMvs1, ERvs1 = confuMatrix(W1,vs1) 
print("\nRemoved feature 3")
print("Training set error rate: ", ERts1)
print(CMts1)
print("Verification set error rate: ", ERvs1)
print(CMvs1)
"""