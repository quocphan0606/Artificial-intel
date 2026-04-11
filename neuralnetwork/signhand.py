import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#load dataset
print("Load MNIST Dataset")
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=np.reshape(x_train,(60000,784))/255.0
x_test= np.reshape(x_test,(10000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")
print(x_train.shape)
print(y_train.shape)
import datetime as dt
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def AccTest(outN,labels): 
    OutMaxArg=np.argmax(outN,axis=1)
    LabelMaxArg=np.argmax(labels,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy
def feedforward(samples,Wh,bh,Wo,bo):
    OutH1=sigmoid(np.dot(samples,Wh.T)+bh)
    OutN=sigmoid(np.dot(OutH1,Wo.T)+bo)
    return OutN
# define network parameters
learningRate = 0.1
BatchSize= 100
Epoch=10
NumOfTrainSample=60000
NumOfTestSample=10000
NumInput=784
NumHidden=512
NumOutput=10; # number of classes
#Hidden layer
Wh=np.matrix(np.random.uniform(-
0.5,0.5,(NumHidden,NumInput)))
bh= np.random.uniform(0,0.5,(1,NumHidden))
dWh= np.zeros((NumHidden,NumInput))
dbh= np.zeros((1,NumHidden))
#Output layer
Wo=np.random.uniform(-0.5,0.5,(NumOutput,NumHidden))
bo= np.random.uniform(0,0.5,(1,NumOutput))
dWo= np.zeros((NumOutput,NumHidden))
dbo= np.zeros((1,NumOutput))
# Train the network with back propagation,SGD
SampleIdx=np.arange(NumOfTrainSample)
t_start=t1=dt.datetime.now()
Acc=np.zeros(Epoch)
for ep in range(Epoch):
    t1=dt.datetime.now()
 # Shuffle the training samples
    np.random.shuffle(SampleIdx)
    for i in range(0,NumOfTrainSample):
        x=np.matrix(x_train[SampleIdx[i],:])
        y=np.matrix(y_train [SampleIdx[i],:])
        # Feedforward propagation
        a=sigmoid(np.dot(x,Wh.T)+bh)
        o =sigmoid(np.dot(a,Wo.T)+bo)
        do=np.multiply(np.multiply(o,(1-o)),(y-o))
        dWo=np.matrix(np.dot(do.T,a))
        dbo=np.mean(do,0)
        Wo=Wo + learningRate*dWo
        bo=bo + learningRate*dbo
        #back propagate error
        dh=np.multiply(np.dot(do,Wo),np.multiply(a,(1-a)))
        dWh=np.dot(dh.T,x)
        dbh=np.mean(dh,0)
    # Update weights
        Wh=Wh + learningRate*dWh
        bh=bh + learningRate*dbh 
    t2=dt.datetime.now()
    print(t2-t1)
    print("Training epoch %i" % ep)
#test the model 
    RealOutN=feedforward(x_test,Wh,bh,Wo,bo)
    Accuracy=AccTest(RealOutN,y_test)
    Acc[ep]=Accuracy
    print("Accuracy: %f" % Accuracy)
t_end=t1=dt.datetime.now()
print("Total time : ", t_end)
plt.plot(Acc,"dr-")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()