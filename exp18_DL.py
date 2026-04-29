import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=1
)

model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',     # Linear activation
    learning_rate_init=0.03,
    max_iter=1000
)

model.fit(X, y)

# Plot output
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("NN - Two Class (Linear Activation)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Accuracy:", model.score(X, y))
