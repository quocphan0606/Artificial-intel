import numpy as np

class Linear:
    def forward(self, inputs):
        self.output = inputs
        return self.output

class Sigmoid:    
    def forward(self, inputs):
        self.output = 1/(1+np.exp(-inputs))
        return self.output

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0    
class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
