
from regularTech import LassoRegression
from regularTech import RidgeRegression
from regularTech import ElasticNetRegression
X = [[1,2],[2,3],[3,4]]
y = [5,7,9]

ridge = RidgeRegression()
ridge.fit(X, y)

lasso = LassoRegression()
lasso.fit(X, y)

elastic = ElasticNetRegression()
elastic.fit(X, y)

print(ridge.predict(X))
print(lasso.predict(X))
print(elastic.predict(X))