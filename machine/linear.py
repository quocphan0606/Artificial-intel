import numpy as np
import matplotlib.pyplot as plt
import torch
import time

np.random.seed(42)

# ===== Bật / tắt GPU =====
USE_GPU = True   # đổi True / False để test

device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== Tạo dữ liệu =====
X = np.random.rand(50,1)*100
Y = 3.5*X + np.random.randn(50,1)*20

X_t = torch.tensor(X, dtype=torch.float32).to(device)
Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

# ===== Model =====
model = torch.nn.Linear(1,1).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# ===== Train =====
start = time.time()

for i in range(1000):
    pred = model(X_t)
    loss = loss_fn(pred, Y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end = time.time()

print("Training time:", end-start)

# ===== Predict =====
with torch.no_grad():
    Y_pred = model(X_t).cpu().numpy()

# ===== Plot =====
plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()