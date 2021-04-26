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
    ts = data[:,:slice1]
    vs = data[:, slice1:(slice1+slice2)]
    return ts, vs, data

def sigmoid(zik):
    return 1/(1+np.exp(-zik))

def trainModel(data, N, lr): #X = data set, N = iterations, lr = alpha = learning rate

    classes, samples, features = data.shape
    
    W = np.zeros((classes, features))
    w_0 = np.zeros((classes))
    t_k = np.zeros((classes, 1))

    g_k = np.zeros((classes))
    g_k[0] = 1 #remember 1

    for k in range(N):
        #1/2 * (g_k-t_k).T*(g_k-t_k) #1/N?
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

            t_k = np.array([[1,0,0],[0,1,0],[0,0,1]])
            

            firstFactor = np.multiply((g_k-t_k), g_k)
            secondFactor = (1-g_k)
            #grad =  np.multiply(np.multiply(((g_k-t_k),g_k)),(1-g_k))*x_k.T
            grad1 = np.multiply(firstFactor, secondFactor)
            gwz_k = x_k

            gMSE += np.matmul(grad1,  gwz_k)
            """
            dg_k_MSE = g_k -t_k
            dz_k_g = np.multiply(g_k,(1-g_k))
            dwz_k = (x_k).T
            dMSE = np.matmul(np.matmul(dg_k_MSE,dz_k_g),dwz_k)"""
           
            MSE += 1/2 * np.matmul((g_k-t_k).T,(g_k-t_k))
            

        W -= lr*gMSE
    return W

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
    

    plt.scatter(alphaSet, errorSet)
    plt.xlabel("alpha")
    plt.ylabel("Error rate")
    plt.show()
    print(alpha)
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
       
        


#task 1a
ts1, vs1, data1 = importIris('Code\iris\iris.data', 30, 20)
print("Task 1a)")
print("Training samples:",ts1.shape)
print("Verification samples:",vs1.shape)

#1b
#calculateAlpha(0.01,0.1,30,ts,1000) # valgt 0.009
#calculateAlpha(0.001,0.02,30,ts,1000) # valgt 0.009
"""
W1 = trainModel(ts1, 1000, 0.009)
CMts1, ERts1 = confuMatrix(W1,ts1) 
CMvs1, ERvs1 = confuMatrix(W1,vs1) 
print("ts = 30, vs = 20, ts used for training")
print("Training set error rate: ", ERts1, "\n",CMts1)
print("Verification set error rate: ", ERvs1, "\n",CMvs1)

#1d
ts2, vs2, data2 = importIris('Code\iris\iris.data', 20, 30)

W2 = trainModel(vs2, 1000, 0.009)
CMts2, ERts2 = confuMatrix(W2,ts2) 
CMvs2, ERvs2 = confuMatrix(W2,vs2) 
print("\nts = 20, vs = 30, vs used for training")
print("Training set error rate: ", ERts2, "\n",CMts2)
print("Verification set error rate: ", ERvs2, "\n",CMvs2)
"""
#2a
plotHistogram(data1, 0)
plotHistogram(data1, 1)
plotHistogram(data1, 2)
plotHistogram(data1, 3)