'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.random.rand(50, 1) * 100  
Y = 3.5 * X + np.random.randn(50, 1) * 20

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

plt.figure(figsize=(8,6)) 
plt.scatter(X, Y, color='blue', label='Data Points') 
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Data
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
y = y.reshape(-1, 1)
m = X.shape[0]

X_b = np.c_[np.ones((m, 1)), X]

# Initial theta
theta = np.array([[2.0], [3.0]])

# ---- Tạo figure có 2 subplot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ===== Hình 1: Before GD =====
axes[0].scatter(X, y)
axes[0].plot(X, X_b.dot(theta))
axes[0].set_title("Before Gradient Descent")
axes[0].set_xlabel("Feature")
axes[0].set_ylabel("Target")

# ---- Gradient Descent ----
learning_rate = 0.1
n_iterations = 100

for _ in range(n_iterations):
    y_pred = X_b.dot(theta)
    gradients = (2 / m) * X_b.T.dot(y_pred - y)
    theta -= learning_rate * gradients

# ===== Hình 2: After GD =====
axes[1].scatter(X, y)
axes[1].plot(X, X_b.dot(theta))
axes[1].set_title("After Gradient Descent")
axes[1].set_xlabel("Feature")
axes[1].set_ylabel("Target")

plt.tight_layout()
plt.show()