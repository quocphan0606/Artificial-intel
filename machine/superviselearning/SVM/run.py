from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# import đúng (fix lỗi của bạn)
from SVM import SVM


# ===== DATA =====
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

# ===== TRAIN =====
model = SVM()
model.fit(X, y)
predictions = model.predict(X)

print("Accuracy:", np.mean(predictions == y))


# ===== VẼ =====
def plot_svm(X, y, model):
    plt.figure()

    # vẽ điểm dữ liệu
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

    w = model.w
    b = model.b

    # tránh lỗi chia 0
    if abs(w[1]) < 1e-6:
        print("Không vẽ được vì w[1] ≈ 0")
        return

    # tạo trục x
    x0 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

    # decision boundary
    x1 = (-w[0]*x0 - b) / w[1]

    # margin
    x1_pos = (-w[0]*x0 - b + 1) / w[1]
    x1_neg = (-w[0]*x0 - b - 1) / w[1]

    # vẽ
    plt.plot(x0, x1, 'y--', label='Decision boundary')
    plt.plot(x0, x1_pos, 'k', label='Margin')
    plt.plot(x0, x1_neg, 'k')

    plt.legend()
    plt.title("SVM Decision Boundary")
    plt.show()


# ===== VẼ VÙNG =====
def plot_svm_region(X, y, model):
    plt.figure()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

    plt.title("SVM Decision Region")
    plt.show()


# ===== RUN =====
plot_svm(X, y, model)
plot_svm_region(X, y, model)