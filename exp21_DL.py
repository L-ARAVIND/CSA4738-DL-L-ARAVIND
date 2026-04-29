import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

X, y = make_circles(n_samples=200, noise=0.1, factor=0.5)

model = MLPClassifier(
    hidden_layer_sizes=(2, 2, 2),   # 3 layers
    activation='relu',
    learning_rate_init=0.1,
    max_iter=1000
)

model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("Exp 21: Circular (ReLU)")
plt.show()

print("Accuracy:", model.score(X, y))
