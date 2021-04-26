import numpy as np
import matplotlib.pyplot as plt



def importIris(filename):
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
    ts = data[:,:30]
    vs = data[:, 30:50]
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
def confuMarix(W, test_set):
    classes, samples, features = test_set.shape
    confusion_matrix = np.zeros((classes, classes)) # n*m, n = prediction, m = true
    #Husk! Finn feil som funksjon av alpha
    for flower in range(classes):
        for s in test_set[flower,:]:
            omega = int(np.argmax(np.matmul(W,s.T)))
            confusion_matrix[omega][flower] += 1
    print(confusion_matrix)
"""
def grad_W_MSE(W,ts): # W = matrix to train, ts = training set
    #Noe kode som finner g_k og t_k
    grad =  np.multiply(np.multiply((g_k-t_k),g_k)),(1-g_k))) #elementwise mult
    return grad
def trainW(ts, N, alpha): 
    classes, samples, features = ts.shape

    W = np.zeros((classes, features)) #Initializing untrained W

    for i in range(N): # This loop is what does the training of W
        grad = grad_W_MSE(W, ts)
        W -= alpha*grad # Do we have to transpose?
        #Need some function to stop if it takes to much time, 
        #and to check if alpha is good enough?
    return W # The trained matrix, which we can use for classification



"""
#task 1a
ts, vs, data = importIris('Kode\iris\iris.data')
print("Task 1a)")
print("Training samples:",ts.shape)
print("Verification samples:",vs.shape)

#task 2a
#print(ts[:,0]) #3x4
#print(ts[:,1])
W = trainModel(ts, 1000, 0.0092)
print(W) 
confuMarix(W,data) 