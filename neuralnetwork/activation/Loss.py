import numpy as np

class Loss:
    def calculate_loss(self, output, y_true):
        samples_losses = self.forward(output, y_true)
        data_loss = np.mean(samples_losses)
        
        return data_loss

class MSE:
        @staticmethod
        def function(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        @staticmethod
        def derivative(y_true, y_pred):
            return 2 * (y_pred - y_true) / y_true.size
class CrossEntropy(Loss):

    def forward(self, y_pred , y_true):
        samples = len(y_pred)

        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_true = y_true

        if len(y_true.shape) == 1:
            correct_confidences = self.y_pred[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_true * self.y_pred, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


'''       
softmax_output = np.array([[0.7, 0.1, 0.2], 
[0.1, 0.5, 0.4], 
[0.02, 0.9, 0.08]]) 
#class_target = np.array([0, 1, 1]) # cat, dog, dog 
class_target = np.array([[1, 0, 0], 
[0, 1, 0], 
[0, 1, 0]])  
loss_function = CrossEntropy() 
loss = loss_function.calculate_loss(softmax_output,class_target) 
print(loss) '''