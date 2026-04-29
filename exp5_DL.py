

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)

X = data[['sepal length (cm)']]
y = data['sepal width (cm)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.show()
