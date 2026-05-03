import numpy as np

# =========================
# Utils
# =========================
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))


def most_common_label(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


# =========================
# Information Gain (binary split)
# =========================
def information_gain(y, left_idx, right_idx):
    parent_entropy = entropy(y)

    n = len(y)
    n_l = len(left_idx)
    n_r = len(right_idx)

    if n_l == 0 or n_r == 0:
        return 0

    e_left = entropy(y[left_idx])
    e_right = entropy(y[right_idx])

    child_entropy = (n_l / n) * e_left + (n_r / n) * e_right

    return parent_entropy - child_entropy


# =========================
# Find best split
# =========================
def best_split(X, y):
    best_gain = -1
    split_idx, split_threshold = None, None

    n_samples, n_features = X.shape

    for feature in range(n_features):
        X_column = X[:, feature]
        thresholds = np.unique(X_column)

        for t in thresholds:
            left_idx = np.where(X_column <= t)[0]
            right_idx = np.where(X_column > t)[0]

            gain = information_gain(y, left_idx, right_idx)

            if gain > best_gain:
                best_gain = gain
                split_idx = feature
                split_threshold = t

    return split_idx, split_threshold, best_gain


# =========================
# Node
# =========================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


# =========================
# Build Tree
# =========================
def build_tree(X, y, depth=0, max_depth=5):
    # Stop conditions
    if len(np.unique(y)) == 1:
        return Node(label=y[0])

    if depth >= max_depth:
        return Node(label=most_common_label(y))

    feature, threshold, gain = best_split(X, y)

    if gain == 0 or feature is None:
        return Node(label=most_common_label(y))

    left_idx = np.where(X[:, feature] <= threshold)[0]
    right_idx = np.where(X[:, feature] > threshold)[0]

    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left, right=right)


# =========================
# Predict
# =========================
def predict_sample(node, x):
    if node.label is not None:
        return node.label

    if x[node.feature] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)


def predict(tree, X):
    return np.array([predict_sample(tree, x) for x in X])


# =========================
# Test thử
# =========================
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    tree = build_tree(X_train, y_train, max_depth=5)

    # Predict
    y_pred = predict(tree, X_test)

    print("Accuracy (from scratch):", accuracy_score(y_test, y_pred))

    # So sánh sklearn
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    y_pred2 = model.predict(X_test)

    print("Accuracy (sklearn):", accuracy_score(y_test, y_pred2))