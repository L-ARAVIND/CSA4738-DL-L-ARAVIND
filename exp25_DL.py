import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

X, y = make_moons(n_samples=200, noise=0.2, random_state=1)

model = MLPClassifier(
    hidden_layer_sizes=(1,),   # very small network
    activation='identity',
    learning_rate_init=0.01,
    max_iter=1000
)

model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("Exp 25: Underfitting (Simple Model)")
plt.show()

print("Accuracy:", model.score(X, y))
