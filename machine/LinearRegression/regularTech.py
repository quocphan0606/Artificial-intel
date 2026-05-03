import numpy as np
'''==================ridge regression=================='''
class RidgeRegression:
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (2/n) * np.dot(X.T, (y_pred - y)) + 2*self.lambda_*self.w
            db = (2/n) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
'''==================lasso regression=================='''
class LassoRegression:
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (2/n) * np.dot(X.T, (y_pred - y)) + self.lambda_ * np.sign(self.w)
            db = (2/n) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
'''==================elastic net regression=================='''
class ElasticNetRegression:
    def __init__(self, lr=0.01, epochs=1000, lambda1=0.1, lambda2=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lambda1 = lambda1  # L1
        self.lambda2 = lambda2  # L2

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (2/n) * np.dot(X.T, (y_pred - y)) \
                 + self.lambda1 * np.sign(self.w) \
                 + 2*self.lambda2 * self.w

            db = (2/n) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

class LassoRegressionGD():
    def __init__(self, learning_rate, iterations, l1_penalty, verbose=True):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.verbose = verbose 
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.losses = []  
        for i in range(self.iterations):
            self.update_weights()
        
                # ---- tính loss ----
            Y_pred = self.predict(self.X)
            loss = np.mean((self.Y - Y_pred) ** 2) + self.l1_penalty * np.sum(np.abs(self.W))
            self.losses.append(loss)

            # ---- in từng bước ----
            if self.verbose and i % 100 == 0:
                print(f"Iter {i}: Loss={loss:.4f}, W={self.W}, b={self.b:.4f}")
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X) # y= wx + b
      
        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) + self.l1_penalty) / self.m
            else:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) - self.l1_penalty) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.W) + self.b
    