import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BaggingClassifier:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.models = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)

            model = self.base_model()
            model.fit(X_sample, y_sample)

            self.models.append(model)

    def predict(self, X):
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # shape: (n_estimators, n_samples)

        # majority vote
        final_pred = []
        for i in range(predictions.shape[1]):
            counts = np.bincount(predictions[:, i])
            final_pred.append(np.argmax(counts))

        return np.array(final_pred)
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
base_model = DecisionTreeClassifier
model = BaggingClassifier(base_model=base_model, n_estimators=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

for i, clf in enumerate(model.models):
    y_pred_i = clf.predict(X_test)
    acc_i = accuracy_score(y_test, y_pred_i)
    print(f"Accuracy of classifier {i+1}: {acc_i:.4f}")