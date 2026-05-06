import Point as point
import numpy as np
import matplotlib.pyplot as plt 
import Dense as dense
import Activation as activation
import Loss as loss
np.random.seed(42)
def plot_layer(X, y, title):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
    plt.title(title)
    plt.show()
# ===== DATA =====
data = point.Spiral(100, 3, 2)
P, L = data.generate()

plt.scatter(P[:,0], P[:,1], c=L, cmap='viridis')
plt.title("Original Data")
plt.show()

# ===== MODEL =====
layer1 = dense.Dense(2, 32)
activation1 = activation.ReLU()

layer2 = dense.Dense(32, 16)
activation2 = activation.ReLU()

layer3  = dense.Dense(16, 3)


# ===== LOSS =====
from Loss import CrossEntropy   # dùng class bạn đã viết
loss_fn = CrossEntropy()

# ===== TRAIN =====
for epoch in range(1000):

    # ----- FORWARD -----
# ===== FORWARD SAU TRAIN =====
    layer1.forward(P)
    output1 = activation1.forward(layer1.output)

    layer2.forward(output1)
    output2 = activation2.forward(layer2.output)

    layer3.forward(output2)



    logits = layer3.output
    loss = loss_fn.calculate_loss(logits, L)

    # accuracy
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == L)

    if epoch % 100 == 0:
        print(f"epoch {epoch} | loss {loss:.4f} | acc {acc:.4f}")

    # ----- BACKWARD -----
    loss_fn.backward()   # 🔥 dùng class bạn đã tạo

    layer3.backward(loss_fn.dinputs)
    activation2.backward(layer3.dinputs)

    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)

    layer1.backward(activation1.dinputs)

    # ----- UPDATE -----
    lr = 0.01
    layer1.weights -= lr * layer1.dweights
    layer1.biases -= lr * layer1.dbiases

    layer2.weights -= lr * layer2.dweights
    layer2.biases -= lr * layer2.dbiases

    layer3.weights -= lr * layer3.dweights
    layer3.biases -= lr * layer3.dbiases
# ===== FORWARD LẠI SAU TRAIN =====
layer1.forward(P)
output1 = activation1.forward(layer1.output)

layer2.forward(output1)
output2 = activation2.forward(layer2.output)

layer3.forward(output2)


# ===== PLOT =====
plot_layer(P, L, "Input Space")

plot_layer(output1[:, :2], L, "After Layer 1")
plot_layer(output2[:, :2], L, "After Layer 2")

pred = np.argmax(layer3.output, axis=1)
plot_layer(P, pred, "Final Prediction (After Training)")