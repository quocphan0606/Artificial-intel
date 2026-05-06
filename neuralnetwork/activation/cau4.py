import Point as point
import numpy as np
import matplotlib.pyplot as plt 
import Dense as dense
import Activation as activation
import Loss as loss
np.random.seed(42)
data = point.Spiral(1000, 3, 2)
P, L = data.generate()

plt.scatter(P[:,0], P[:,1], c=L, cmap='viridis')
plt.show()

layer1 = dense.Dense(2, 32)
layer1.forward(P)
activation1 = activation.ReLU()
output1 = activation1.forward(layer1.output)
print(output1)
plt.scatter(output1[:,0], output1[:,1], c=L, cmap='viridis')
plt.show()

layer2 = dense.Dense(32, 16)
layer2.forward(output1)
activation2 = activation.ReLU()
output2 = activation2.forward(layer2.output)
print(output2)
plt.scatter(output2[:,0], output2[:,1], c=L, cmap='viridis')
plt.show()

layer3  = dense.Dense(16, 3)
layer3.forward(output2)
activation3 = activation.Softmax()
output3 = activation3.forward(layer3.output)
print(output3)
plt.scatter(output3[:,0], output3[:,1], c=L, cmap='viridis')
plt.show()
loss_function = loss.CrossEntropy() 
loss = loss_function.calculate_loss(activation3.output,L) 
predictions = np.argmax(activation3.output, axis =1)
print(loss)
accuracy = np.mean(predictions == L)
print(accuracy)

