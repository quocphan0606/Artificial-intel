import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # convert label về -1, +1
        y = np.where(y == 0, -1, 1)

        # khởi tạo trọng số
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # base model: decision stump
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)

            y_pred = model.predict(X)

            # tính error
            error = np.sum(w * (y != y_pred))

            # tránh chia 0
            error = max(error, 1e-10)

            # tính alpha
            alpha = 0.5 * np.log((1 - error) / error)

            # cập nhật trọng số
            w *= np.exp(-alpha * y * y_pred)

            # chuẩn hóa
            w /= np.sum(w)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])

        for model, alpha in zip(self.models, self.alphas):
            pred = model.predict(X)
            final_pred += alpha * pred

        return np.sign(final_pred)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = AdaBoost(n_estimators=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# convert lại về 0/1
y_pred = np.where(y_pred == -1, 0, 1)

print("Accuracy:", accuracy_score(y_test, y_pred))