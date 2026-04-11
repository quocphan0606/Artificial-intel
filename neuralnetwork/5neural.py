import numpy as np
import matplotlib.pyplot as plt
Char = np.matrix([[0,0,1,0,
                    0,1,1,0,
                    0,0,1,0,
                    0,0,1,0,
                    0,1,1,1], 
                    [1,1,1,1,
                    0,0,0,1,
                    1,1,1,1,
                    1,0,0,0,
                    1,1,1,1],
                    [1,1,1,1,
                    0,0,0,1,
                    1,1,1,1,
                    0,0,0,1,
                    1,1,1,1],
                    [1,0,0,1,
                    1,0,0,1,
                    1,1,1,1,
                    0,0,0,1,
                    0,0,0,1],
                    [1,1,1,1,
                    1,0,0,0,
                    1,1,1,1,
                    0,0,0,1,
                    1,1,1,1]],dtype = np.int8)
Target = np.matrix([[1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1]])
plt.subplot(151)
imgplot = plt.imshow(Char[0,:].reshape(5,4))
plt.subplot(152)
imgplot = plt.imshow(Char[1,:].reshape(5,4))
plt.subplot(153)
imgplot = plt.imshow(Char[2,:].reshape(5,4))
plt.subplot(154)
imgplot = plt.imshow(Char[3,:].reshape(5,4))
plt.subplot(155)
imgplot = plt.imshow(Char[4,:].reshape(5,4))
plt.show()
def sigmoid(x):
    return 1./(1.+np.exp(-x))
Cost = np.zeros(100)
W = np.matrix(np.random.uniform(-0.1,0.1,(5,20)))
n=0.1 # learning rate
for j in range(100):
    dW = np.zeros_like(W)
    for i in range (5):
        X = Char[i,:]
        t= Target[i,:]
        a=np.dot(X,W.transpose())
        o=sigmoid(a)
        d = np.multiply(np.multiply((t-o),o),(1-o))
        dW += d.T*X
    #update
    W=W+n*dW
    #Check the output, calculate cost
    A= sigmoid(Char*W.T)
    Cost[j]=np.mean(np.power((Target-A),2))
plt.plot(Cost,)
plt.ylabel('Cost function')
plt.xlabel('Iteration')
plt.show()