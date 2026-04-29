

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(0)

X = np.sort(np.random.rand(30))
y = np.cos(1.5 * np.pi * X) + np.random.randn(30) * 0.1

degrees = [1, 4, 15]

for i in degrees:
    plt.figure(figsize=(6,4))

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=i)),
        ("linear", LinearRegression())
    ])

    model.fit(X[:, np.newaxis], y)

    X_test = np.linspace(0,1,100)
    plt.plot(X_test, model.predict(X_test[:, np.newaxis]))
    plt.scatter(X, y)
    plt.title("Degree " + str(i))
    plt.show()
