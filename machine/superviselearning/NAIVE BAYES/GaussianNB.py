import numpy as np
class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # lưu mean, var, prior
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]

            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        numerator = np.exp(- (x - mean) ** 2 / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)

        return numerator / denominator

    def _predict(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._gaussian(c, x)))

            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = np.mean(preds == y_test)
print("Accuracy:", accuracy)