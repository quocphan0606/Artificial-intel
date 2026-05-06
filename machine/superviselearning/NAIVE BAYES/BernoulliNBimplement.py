import numpy as np
class BernoulliNB:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.feature_probs = {}  # P(x_i = 1 | y)

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes:
            X_c = X[y == c]

            # prior
            self.priors[c] = X_c.shape[0] / n_samples

            # Laplace smoothing
            self.feature_probs[c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)

    def _predict(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            probs = self.feature_probs[c]

            # Bernoulli likelihood
            likelihood = np.sum(
                x * np.log(probs) + (1 - x) * np.log(1 - probs)
            )

            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "free money now",
    "call me tonight",
    "win cash prize",
    "hello friend"
]

labels = [1, 0, 1, 0]

cv = CountVectorizer(binary=True)
X = cv.fit_transform(texts).toarray()
y = np.array(labels)
model = BernoulliNB()
model.fit(X, y)
test = ["free cash now"]
test_vec = cv.transform(test).toarray()

print("Prediction:", model.predict(test_vec))