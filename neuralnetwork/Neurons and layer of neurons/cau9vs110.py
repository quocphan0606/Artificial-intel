# cau 9 khoi tao 1 lop neuron 3 dau vao 2 neuron
import numpy as np
from Dense import Dense
input = [1,2,4]
dense1=Dense(3,2)
dense1.forward(input)
print(dense1.output)
# cau 10 khoi tao 2 lop neuron 4 dau vao 3 neuron 2 neuron
input2 = [1, 2, 3, 4]
dense2= Dense(4,3)
dense2.forward(input2)
print(dense2.output)
dense3 = Dense(3,3)
dense3.forward(dense2.output)
print(dense3.output)
