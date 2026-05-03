'''class sklearn.tree.DecisionTreeClassifier(
    *,
    criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None
)'''
#1: Importing Libraries
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# 2: Load data
data = load_iris()
X = data.data  
y = data.target
# 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = 99)
# 4: Defining the Model
clf = DecisionTreeClassifier(random_state=1)
# Training the Model
clf.fit(X_train, y_train)
# 6: Making Predictions
y_pred = clf.predict(X_test)
# 7: Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#7: Hyperparameter Tuning with Decision Tree Classifier using GridSearchCV
param_grid = {
    'max_depth': range(1, 10, 1),
    'min_samples_leaf': range(1, 20, 2),
    'min_samples_split': range(2, 20, 2),
    'criterion': ["entropy", "gini"]
}

tree = DecisionTreeClassifier(random_state=1)

grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, 
                           cv=5, verbose=True)
grid_search.fit(X_train, y_train)

print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)
# Visualize the best decision tree
tree_clf = grid_search.best_estimator_

plt.figure(figsize=(18, 15))
plot_tree(tree_clf, filled=True, feature_names=data.feature_names,
          class_names=data.target_names)
plt.show()