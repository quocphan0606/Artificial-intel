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
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_regression

# ===== DATA =====
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
y = y.reshape(-1, 1)
m = X.shape[0]

X_b = np.c_[np.ones((m, 1)), X]

# Initial theta
theta = np.array([[2.0], [3.0]])

learning_rate = 0.1
n_iterations = 100

# ===== FIGURE =====
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X, y, color='blue')

line, = ax.plot(X, X_b.dot(theta), color='red')
ax.set_title("Gradient Descent Animation")
ax.set_xlabel("Feature")
ax.set_ylabel("Target")

# ===== UPDATE FUNCTION =====
def update(frame):
    global theta
    
    # Gradient Descent step
    y_pred = X_b.dot(theta)
    gradients = (2 / m) * X_b.T.dot(y_pred - y)
    theta -= learning_rate * gradients
    
    # Update line
    line.set_ydata(X_b.dot(theta))
    ax.set_title(f"Iteration {frame}")
    
    return line,

# ===== ANIMATION =====
ani = FuncAnimation(fig, update, frames=n_iterations, interval=100, blit=True)

plt.show()