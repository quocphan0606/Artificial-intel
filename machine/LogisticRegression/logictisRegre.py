import numpy as np

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss
def compute_loss(y, y_hat):
    m = len(y)
    epsilon = 1e-9  # tránh log(0)
    return - (1/m) * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))

# Train
def train(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    
    # thêm bias
    X_b = np.c_[np.ones((m, 1)), X]
    
    # khởi tạo theta
    theta = np.zeros((n + 1, 1))
    
    y = y.reshape(-1, 1)
    
    for epoch in range(epochs):
        z = X_b @ theta
        y_hat = sigmoid(z)
        
        # gradient
        gradient = (1/m) * X_b.T @ (y_hat - y)
        
        # update
        theta -= lr * gradient
        
        if epoch % 100 == 0:
            loss = compute_loss(y, y_hat)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return theta

# Predict
def predict(X, theta):
    m = X.shape[0]
    X_b = np.c_[np.ones((m, 1)), X]
    probs = sigmoid(X_b @ theta)
    return (probs >= 0.5).astype(int)

# tạo data đơn giản
np.random.seed(42)
X = np.random.randn(100, 2)

# tạo nhãn
y = (X[:, 0] + X[:, 1] > 0).astype(int)

theta = train(X, y, lr=0.1, epochs=1000)

y_pred = predict(X, theta)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)