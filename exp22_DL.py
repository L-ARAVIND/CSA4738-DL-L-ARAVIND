import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=1
)

model = MLPClassifier(
    hidden_layer_sizes=(2, 2, 2),
    activation='relu',
    learning_rate_init=0.1,
    max_iter=1000
)

model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("Exp 22: Two Class (ReLU)")
plt.show()

print("Accuracy:", model.score(X, y))
