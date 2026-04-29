import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=1
)

model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    learning_rate_init=0.01,
    max_iter=1000
)

model.fit(X, y)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='viridis')
plt.title("NN - Multi Class (Linear Activation)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Accuracy:", model.score(X, y))
