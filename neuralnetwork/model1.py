import numpy as np
# ============== thong so ====================
epochs = 5000
lr = 0.1         #learning rate

#==================ACTIVATION===================
def sigmoid(x):
    return 1/(1+np.exp(-x))


def Relu(x):
    return np.maximum(0,x)

def Relu_dir(x):
    return (x>0).astype(int)

# ==================== LOSS FUNCTION =====================
def mse(y_true, y_fred):
    return np.mean((y_true- y_fred)**2)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1-1e-9)
    return -np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

# =============== CONSTANT ===============
x = np.array([1, 0, 1, 1, 0, 1])
'''w1 = np.array([
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1]
])
b1 = np.array([-5, 0, 1,-3])

w2 = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 1, 1]
])

b2 = np.array([0, 1,0])

w3 = np.array([
    [1, 0, 1],
    [0, 1, 1]
])

b3 = np.array([0, 0])
'''
w1 = np.random.randn(4,6) * 0.1
w2 = np.random.randn(3,4) * 0.1
w3 = np.random.randn(2,3) * 0.1


b1 = np.zeros(4)
b2 = np.zeros(3)
b3 = np.zeros(2)


w1 = w1.astype(float)
w2 = w2.astype(float)
w3 = w3.astype(float)

b1 = b1.astype(float)
b2 = b2.astype(float)
b3 = b3.astype(float)
#====================== output need ===============
y_true = np.array([1, 0])
    #============== FOWARD PROPAGATION ====================
for epoch in range(epochs):
    # ======== layer1 ============
    z1 = np.dot(w1, x) + b1
    A1 = Relu(z1)

    # ======== layer2 ============
    z2 = np.dot(w2, A1) + b2
    A2 = Relu(z2)
    # ======== ouput============
    z3 = np.dot(w3, A2) + b3

    A3= sigmoid(z3)


    y_pred = A3

    loss = cross_entropy(y_true, y_pred)
    
    # ======================= BACK PROPAGATION===================
    dZ3 = (A3 - y_true)
    # layer 3
    dw3 = np.outer(dZ3,A2)
    db3 = dZ3

    dA2 = np.dot(w3.T, dZ3)
    dZ2 = dA2 * Relu_dir(z2)
    # layer 2
    dw2 = np.outer(dZ2, A1)            # outer la tich ngoai  2 vector
    db2 = dZ2

    dA1 = np.dot(w2.T, dZ2)
    dZ1 = dA1 * Relu_dir(z1)

    dw1 = np.outer(dZ1, x)
    db1 = dZ1

    # ================update ==================
    w3 -= lr * dw3
    b3 -= lr * db3

    w2 -= lr * dw2
    b2 -= lr * db2

    w1 -= lr * dw1
    b1 -= lr * db1
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss = {loss:.6f}")
        print("==================================")
#==================PRINT ======================
print("Final prediction:", A3)
print("Target:", y_true)