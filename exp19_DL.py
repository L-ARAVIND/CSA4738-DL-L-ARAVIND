import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

# Generate circular dataset
X, y = make_circles(
    n_samples=200,
    noise=0.1,
    factor=0.5
)

# Neural Network model
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',   # Linear activation
    learning_rate_init=0.03,
    max_iter=1000
)

model.fit(X, y)

# Plot output
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("NN - Circular Data (Linear Activation)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Accuracy:", model.score(X, y))
