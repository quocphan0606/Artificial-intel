import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV 

# loading and print dataset
cancer = load_breast_cancer()

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])

print("Feature Variables: ")
print(df_feat.info())
print("Dataframe looks like : ")
print(df_feat.head())
# splitting data 

X_train, X_test, y_train, y_test = train_test_split( 
						df_feat, np.ravel(df_target), 
				test_size = 0.30, random_state = 101)

# traing the model 
model = SVC() 
model.fit(X_train, y_train) 

predictions = model.predict(X_test) 

print(classification_report(y_test, predictions))

# hyperparameter tuning 

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
			'kernel': ['rbf']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
 
grid.fit(X_train, y_train)
print("================================")
print(grid.best_params_) 
print(grid.best_estimator_)


grid_predictions = grid.predict(X_test) 
print(classification_report(y_test, grid_predictions))