import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from regularTech import LassoRegressionGD

df = pd.read_csv("machine/Experience-Salary.csv")
df.head()
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/3, random_state=0)


model = LassoRegressionGD(iterations=1000, learning_rate=0.01, l1_penalty=500)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


print("Predicted values: ", np.round(Y_pred[:3], 2))
print("Real values:      ", Y_test[:3])
print("Trained W:        ", round(model.W[0], 2))
print("Trained b:        ", round(model.b, 2))



plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
plt.plot(X_test, Y_pred, color='yellow', label='Lasso Regression Line')
plt.title('Salary vs Experience (Lasso Regression)')
plt.xlabel('Years of Experience (Standardized)')
plt.ylabel('Salary')
plt.legend()
plt.show()